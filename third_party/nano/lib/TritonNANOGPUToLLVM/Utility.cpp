#include "Utility.h"
#include "TritonNANOGPUToLLVM/TargetUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
namespace tt = mlir::triton;
using mlir::triton::ModuleAxisInfoAnalysis;
using mlir::triton::NANO::DppCtrl;
using mlir::triton::NANO::ISAFamily;
using mlir::triton::gpu::appendOrGetExternFuncOp;

namespace {
enum class ShflKind : uint32_t {
  bfly = 0,
  up = 1,
  down = 2,
  idx = 3,
};
} // namespace

namespace mlir::LLVM::NANO {
static Value shuffleCommonImpl(Location loc, RewriterBase &rewriter,
                               ISAFamily isaFamily, Value val, Value i,
                               int strideInt, ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  // On AMD, the ds_swizzle_b32 and ds_permute_b32 instructions work on
  // 32bit/dwords so we need promote to 32 here.
  auto valType = val.getType();
  if (!valType.isInteger(32) && bits <= 32) {
    if (!valType.isIntOrIndex())
      val = b.bitcast(val, int_ty(bits));
    if (bits < 32)
      val = b.sext(i32_ty, val);

    val = shuffleCommonImpl(loc, rewriter, isaFamily, val, i, strideInt, mode,
                            clamp);

    if (bits < 32)
      val = b.trunc(int_ty(bits), val);
    if (!valType.isIntOrIndex())
      val = b.bitcast(val, valType);
    return val;
  }

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = b.bitcast(val, vecTy);
    Value val0 = b.extract_element(f32_ty, vec, b.i32_val(0));
    Value val1 = b.extract_element(f32_ty, vec, b.i32_val(1));
    val0 = shuffleCommonImpl(loc, rewriter, isaFamily, val0, i, strideInt, mode,
                             clamp);
    val1 = shuffleCommonImpl(loc, rewriter, isaFamily, val1, i, strideInt, mode,
                             clamp);
    vec = b.undef(vecTy);
    vec = b.insert_element(vecTy, vec, val0, b.i32_val(0));
    vec = b.insert_element(vecTy, vec, val1, b.i32_val(1));
    return b.bitcast(vec, val.getType());
  }

  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId = getThreadId(rewriter, loc);

  unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value warpSize = b.i32_val(iWarpSize);
  Value laneId = b.urem(threadId, warpSize);
  auto bpermute = [&](Value lane) {
    // Multiple lineId by 4. (More on permute instruction semantics:
    // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf#page=180
    Value byteOffset = b.i32_val(2);
    Value permuteAddr = b.shl(lane, byteOffset);
    return ROCDL::DsBpermuteOp::create(rewriter, loc, valType, permuteAddr,
                                       val);
  };

  switch (mode) {
  case ShflKind::bfly:
    if (strideInt > 16) {
      Value stride = b.i32_val(32);
      Value lineId = b.xor_(threadId, stride);
      return bpermute(lineId);
    } else if (strideInt == 16) {
      if (isRDNA(isaFamily)) {
        // Lane i in the upper 16 lanes reads the value from lane i in the lower
        // 16 lanes and vice versa.
        Value select_lo = b.i32_val(0x76543210);
        Value select_hi = b.i32_val(0xfedcba98);
        return ROCDL::PermlaneX16Op::create(rewriter, loc, valType, val, val,
                                            select_lo, select_hi, true, false);
      } else {
        Value offset = b.i32_val(0x401F);
        return ROCDL::DsSwizzleOp::create(rewriter, loc, valType, val, offset);
      }
    } else {
      if (!llvm::is_contained({ISAFamily::CDNA2, ISAFamily::CDNA3,
                               ISAFamily::CDNA4, ISAFamily::RDNA3,
                               ISAFamily::RDNA4},
                              isaFamily)) {
        // DPP is only supported for CDNA2/CDNA3/CDNA4/RDNA3/RDNA4 right now, so
        // we fallback to ds_swizzle for other architectures.
        //
        // This map facilates the butterfly shuffle pattern for a stride less
        // than 16. The pattern stride is the key of the map.
        DenseMap<short, unsigned int> masks{
            {16, 0x401F}, {8, 0x201F}, {4, 0x101F}, {2, 0x081F}, {1, 0x041F}};
        Value offset = b.i32_val(masks[strideInt]);
        return ROCDL::DsSwizzleOp::create(rewriter, loc, valType, val, offset);
      }

      auto createDppOpWithoutBoundCtrl = [&](Value &old, Value &src,
                                             uint32_t dppCtrl, uint32_t rowMask,
                                             uint32_t bankMask) {
        return ROCDL::DPPUpdateOp::create(rewriter, loc, valType, old, src,
                                          rewriter.getI32IntegerAttr(dppCtrl),
                                          rewriter.getI32IntegerAttr(rowMask),
                                          rewriter.getI32IntegerAttr(bankMask),
                                          rewriter.getBoolAttr(false));
      };

      const int allRows = 0xf;
      const int allBanks = 0xf;

      switch (strideInt) {
      case 1: {
        // quad_perm: 1, 0, 3, 2
        uint32_t dppCtrl = static_cast<uint32_t>(DppCtrl::QUAD_PERM_FIRST);
        std::array<uint32_t, 4> mask = {1, 0, 3, 2};
        for (int i = 0; i < mask.size(); i++) {
          dppCtrl |= mask[i] << (i * 2);
        }
        return createDppOpWithoutBoundCtrl(val, val, dppCtrl, allRows,
                                           allBanks);
      }
      case 2: {
        // quad_perm: 2, 3, 0, 1
        uint32_t dppCtrl = static_cast<uint32_t>(DppCtrl::QUAD_PERM_FIRST);
        std::array<uint32_t, 4> mask = {2, 3, 0, 1};
        for (int i = 0; i < mask.size(); i++) {
          dppCtrl |= mask[i] << (i * 2);
        }
        return createDppOpWithoutBoundCtrl(val, val, dppCtrl, allRows,
                                           allBanks);
      }
      case 4: {
        // row_shr:4 bank_mask: 0xa
        auto ret = createDppOpWithoutBoundCtrl(
                       val, val, 4 + static_cast<uint32_t>(DppCtrl::ROW_SHR0),
                       allRows, 0xa)
                       .getRes();

        // row_shl:4 bank_mask: 0x5
        return createDppOpWithoutBoundCtrl(
            ret, val, 4 + static_cast<uint32_t>(DppCtrl::ROW_SHL0), allRows,
            0x5);
      }
      case 8: {
        // row_shr:8 bank_mask: 0xc
        auto ret = createDppOpWithoutBoundCtrl(
                       val, val, 8 + static_cast<uint32_t>(DppCtrl::ROW_SHR0),
                       allRows, 0xc)
                       .getRes();

        // row_shl:8 bank_mask: 0x3
        return createDppOpWithoutBoundCtrl(
            ret, val, 8 + static_cast<uint32_t>(DppCtrl::ROW_SHL0), allRows,
            0x3);
      }
      default:
        assert(false &&
               "bfly shfl with stride >= 16 should not be handled by dpp.");
      }
    }
    break;
  case ShflKind::up: {
    Value mask = b.icmp_slt(laneId, i);
    Value delta = b.sub(laneId, i);
    Value index = b.select(mask, laneId, delta);
    return bpermute(index);
  }
  case ShflKind::idx:
    return bpermute(i);
  default:
    assert(false && "Unsupported ShflKind");
    break;
  }
  return Value();
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter,
                           ISAFamily isaFamily, Value val, Value i,
                           int strideInt, ShflKind mode, Value clamp) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // To shuffle pointers, convert them to i64.
  Type valTy = val.getType();
  if (isa<LLVM::LLVMPointerType>(valTy))
    val = b.ptrtoint(i64_ty, val);
  Value result = shuffleCommonImpl(loc, rewriter, isaFamily, val, i, strideInt,
                                   mode, clamp);
  if (isa<LLVM::LLVMPointerType>(valTy))
    result = b.inttoptr(valTy, result);
  return result;
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, b.i32_val(i), i,
                       ShflKind::bfly, b.i32_val(0x1f));
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i,
                ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, b.i32_val(i), i,
                       ShflKind::up, b.i32_val(0x0));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleIdx(loc, rewriter, val, b.i32_val(i), isaFamily);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i,
                 ISAFamily isaFamily) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return shuffleCommon(loc, rewriter, isaFamily, val, i, 0, ShflKind::idx,
                       b.i32_val(0x1f));
}

Value permute(Location loc, RewriterBase &rewriter, Value x, Value y,
              Value selector) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value prmt_mask = selector;
  // convert from nybble mask to byte mask:
  prmt_mask =
      b.or_(b.and_(prmt_mask, b.i32_val(0x000000ff)),
            b.shl(b.and_(prmt_mask, b.i32_val(0x0000ff00)), b.i32_val(8)));
  prmt_mask =
      b.or_(b.and_(prmt_mask, b.i32_val(0x000f000f)),
            b.shl(b.and_(prmt_mask, b.i32_val(0x00f000f0)), b.i32_val(4)));
  Value args[] = {x, y, prmt_mask};
  auto op = createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.perm", i32_ty,
                                      args);
  return op.getResult(0);
}

// Utility function that returns flags <volatile, nontemporal> for a predicated
// Load or Store
// ---------------------------------
// Op   | cm  | volatile | NT
// -----+-----+---------------------
// Load | .ca |   F      | F
//      | .cg |   F      | T
//      | .cs |   F      | T
//      | .cv |   T      | X
// -----+-----+----------+---------
// Store| .wb |   F      | F
//      | .cg |   F      | F
//      | .cs |   F      | T
//      | .wt |   T      | X
// -----+-----+----------+---------
std::pair<bool, bool>
getCacheModifierFlagsForLoadStore(const triton::CacheModifier &cm,
                                  MemoryOp op) {
  switch (op) {
  case MemoryOp::Load: {
    switch (cm) {
    case triton::CacheModifier::CA:
      return std::make_pair(false, false);
    case triton::CacheModifier::CG:
      return std::make_pair(false, true);
    case triton::CacheModifier::CS:
      return std::make_pair(false, true);
    case triton::CacheModifier::CV:
      return std::make_pair(true, true);
    default:
      return std::make_pair(false, false);
    }
  }
  case MemoryOp::Store: {
    switch (cm) {
    case triton::CacheModifier::CG:
      return std::make_pair(false, false);
    case triton::CacheModifier::CS:
      return std::make_pair(false, true);
    case triton::CacheModifier::WT:
      return std::make_pair(true, true);
    default:
      return std::make_pair(false, false);
    }
  }
  }
  return std::make_pair(false, false);
}

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis) {
  assert(moduleOp);

  int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);
  if (numCTAs == 1) {
    // For single CTA the block id is the program id
    Value blockId = ::mlir::gpu::BlockIdOp::create(rewriter, loc,
                                                   mlir::gpu::Dimension(axis));
    return arith::IndexCastOp::create(rewriter, loc, i32_ty, blockId);
  }
  // For multiple CTAs the cluster id is the program id
  Type resType = rewriter.getI32Type();
  Value clusterIdx = nullptr;
  switch (axis) {
  case ProgramIDDim::X: {
    clusterIdx = ROCDL::ClusterIdXOp::create(rewriter, loc, resType);
    break;
  }
  case ProgramIDDim::Y: {
    clusterIdx = ROCDL::ClusterIdYOp::create(rewriter, loc, resType);
    break;
  }
  case ProgramIDDim::Z: {
    clusterIdx = ROCDL::ClusterIdZOp::create(rewriter, loc, resType);
    break;
  }
  }
  return clusterIdx;
}

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, Value multicastMask,
             triton::CacheModifier cm, bool forceNoAliasAsyncLoads) {
  // Direct LLVM load - MaskedLoadOp removed with TritonNANOGPU dialect
  TritonLLVMOpBuilder b(loc, rewriter);

  // Simple case: unconditional load
  auto load = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr, /*alignment*/ 0);
  // addLocalLoadNoAliasScope removed - not needed for minimal nano backend
  return load.getResult();
}

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm,
             bool forceNoAliasAsyncLoads) {
  // Direct LLVM store - MaskedStoreOp removed with TritonNANOGPU dialect
  auto store = LLVM::StoreOp::create(rewriter, loc, val, ptr, /*alignment*/ 0);
  // addLocalLoadNoAliasScope removed - not needed for minimal nano backend
}

static int32_t getDefaultCtrlBitsForCacheModifier(triton::CacheModifier cm) {
  return 0;
}

Value cvtFp32ToFp16RTNE_oneValue(Location loc, RewriterBase &rewriter,
                                 const Value &v) {
  LLVM::RoundingMode rm = LLVM::RoundingMode::NearestTiesToEven;
  return LLVM::FPTruncOp::create(rewriter, loc, f16_ty, v);
}

Type getPointerTypeWithShape(Value basePtr, Value offset) {
  Type basePtrType = basePtr.getType();
  auto offsetType = cast<RankedTensorType>(offset.getType());
  return offsetType.cloneWith(std::nullopt, basePtrType);
}

unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  return axisAnalysisPass.getContiguity(ptr);
}

unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {

  Type type = getPointerTypeWithShape(ptr, offset);
  RankedTensorType tensorTy = cast<RankedTensorType>(type);

  // To compute the contiguity of the scalar/warp-uniform ptr and offset pair we
  // need to look at the contiguity of the offsets and the alignment of the ptr
  auto elemNumBits = triton::getPointeeBitWidth(tensorTy);
  auto contiguity = axisAnalysisPass.getContiguity(offset, elemNumBits);

  // To get the alignment of the scalar ptr we need to look at the divisibility
  auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
  auto maxMultipleBytes = axisInfo->getDivisibility(0);
  auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
  auto align = std::max<unsigned>(maxMultipleBytes / elemNumBytes, 1);

  // FIXME (Alex): this should not be needed anymore because it's done inside
  // getContiguity, but we have an order issues with LL, so we keep this
  // until the LL order issue is fixed
  auto linearLayout = triton::gpu::toLinearLayout(tensorTy);
  auto llAttr = triton::gpu::LinearEncodingAttr::get(tensorTy.getContext(),
                                                     std::move(linearLayout));
  auto order = triton::gpu::getOrder(tensorTy);
  auto contigPerThread = llAttr.getContigPerThread();
  assert(order[0] < contigPerThread.size() &&
         "Unexpected contigPerThread size");
  contiguity = std::min(contiguity, contigPerThread[order[0]]);

  // Final contiguity is a min of the offset contiguity and pointer alignment
  return std::min(align, contiguity);
}

unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  auto contiguity = getContiguity(ptr, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

unsigned getVectorSize(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto contiguity = getContiguity(ptr, offset, axisAnalysisPass);
  auto pointeeBitWidth = triton::getPointeeBitWidth(ptr.getType());
  return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
}

Type scaleDotElemTypeToMLIRType(MLIRContext *ctx, triton::ScaleDotElemType t) {
  switch (t) {
  case triton::ScaleDotElemType::FP16:
    return Float16Type::get(ctx);
  case triton::ScaleDotElemType::BF16:
    return BFloat16Type::get(ctx);
  case triton::ScaleDotElemType::E4M3:
    return Float8E4M3FNType::get(ctx);
  case triton::ScaleDotElemType::E5M2:
    return Float8E5M2Type::get(ctx);
  case triton::ScaleDotElemType::E3M2:
    return Float6E3M2FNType::get(ctx);
  case triton::ScaleDotElemType::E2M3:
    return Float6E2M3FNType::get(ctx);
  case triton::ScaleDotElemType::E2M1:
    return Float4E2M1FNType::get(ctx);
  default:
    llvm_unreachable("unsupported ScaleDotElemType!");
  }
}

bool isChainDotHead(tt::DotOpInterface dotOp, unsigned opIdx) {
  auto isInSameRegion = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  ForwardSliceOptions fwdOpt;
  fwdOpt.filter = isInSameRegion;
  SetVector<mlir::Operation *> fwdSlices;
  getForwardSlice(dotOp, &fwdSlices, fwdOpt);
  for (Operation *op : fwdSlices) {
    if (auto dOp = dyn_cast<tt::DotOpInterface>(op)) {
      assert(dOp != dotOp);
      Operation *dotOperand = (opIdx == 0) ? dOp.getA().getDefiningOp()
                                           : dOp.getB().getDefiningOp();
      if (dotOperand && fwdSlices.contains(dotOperand)) {
        return true;
      }
    }
  }
  return false;
}

bool isChainDotTail(tt::DotOpInterface dotOp) {
  auto isInSameRegion = [&dotOp](Operation *op) {
    return op->getParentRegion() == dotOp->getParentRegion();
  };
  BackwardSliceOptions bwdOpt;
  bwdOpt.omitBlockArguments = true;
  bwdOpt.filter = isInSameRegion;
  SetVector<Operation *> bwdSlices;
  Operation *opA = dotOp.getA().getDefiningOp();
  if (!opA)
    return false;
  (void)getBackwardSlice(opA, &bwdSlices, bwdOpt);
  if (llvm::find_if(bwdSlices, [](Operation *op) {
        return isa<tt::DotOpInterface>(op);
      }) != bwdSlices.end())
    return true;
  return false;
}

} // namespace mlir::LLVM::NANO
