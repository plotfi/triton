// TritonNANOGPU dialect removed - not needed for minimal nano backend
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNANOGPUToLLVM/TargetUtils.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

using ::mlir::triton::gpu::MemDescType;

namespace {
class TransLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
public:
  TransLocalLoadOpConversion(const LLVMTypeConverter &converter,
                             const NANO::TargetInfo &targetInfo,
                             PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::LocalLoadOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    auto typeConverter = this->getTypeConverter();
    auto llvmElemTy = typeConverter->convertType(dstTy.getElementType());
    unsigned bitWidth = llvmElemTy.getIntOrFloatBitWidth();

    // FP4 is represented as i8 and, when packed along K, can be
    // transposed using ds_read_tr8 which doesn't change packing.
    if (bitWidth != 16 && bitWidth != 8) {
      return failure();
    }
    auto ldsParams = targetInfo.queryLDSTransLoadParams(bitWidth);
    if (!ldsParams)
      return failure();

    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcTy.getEncoding());
    LinearLayout cvtDstLL = LinearLayout::empty();
    if (paddedEnc) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvtDstLL = triton::gpu::toLinearLayout(dstTy).invertAndCompose(sharedLL);
      if (paddedEnc.getMinInterval() < ldsParams->tileSize)
        return failure();
    } else {
      auto sharedLL = triton::gpu::toLinearLayout(srcTy);
      cvtDstLL = triton::gpu::toLinearLayout(dstTy).invertAndCompose(sharedLL);
    }
    auto kBlock = StringAttr::get(ctx, "block");
    auto maybeSublayout = cvtDstLL.quotient({kBlock});
    if (!maybeSublayout) {
      return failure();
    }
    cvtDstLL = maybeSublayout.value();
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    auto smemBase = smemObj.getBase();
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, srcTy);
    auto maskSpanAffineOffset = smemObj.getMaskSpanOffsets(srcTy);
    auto paddingShifts = getPaddedSharedShifts(srcTy.getEncoding(),
                                               srcTy.getElementTypeBitWidth(),
                                               /*offsetInBytes=*/true);

    llvm::SmallVector<Value> values;
    auto result = lowerDsReadTr(
        op, ldsParams.value(), loc, cvtDstLL, values, smemBase, affineOffset,
        maskSpanAffineOffset, paddingShifts, llvmElemTy, rewriter, targetInfo);
    if (failed(result)) {
      return failure();
    }

    auto structTy = LLVM::LLVMStructType::getLiteral(
        ctx, SmallVector<Type>(values.size(), llvmElemTy));
    auto value = packLLElements(loc, typeConverter, values, rewriter, structTy);

    rewriter.replaceOp(op, value);
    return success();
  }

private:
  LogicalResult lowerDsReadTr(
      triton::gpu::LocalLoadOp op,
      ::triton::NANO::TargetInfo::LDSTransLoadParams ldsParams, Location loc,
      LinearLayout cvt,
      SmallVector<Value> &vals, // Input for stmatrix, output for ldmatrix
      Value smemBase, Value affineOffset, uint64_t maskSpanAffineOffset,
      ArrayRef<std::pair<unsigned, unsigned>> paddingShifts, Type llvmElemTy,
      ConversionPatternRewriter &rewriter,
      const ::triton::NANO::TargetInfo &targetInfo) const {

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();

    auto S = [ctx](StringRef v) { return StringAttr::get(ctx, v); };
    auto kReg = S("register");
    auto kLane = S("lane");
    auto kWarp = S("warp");
    auto kOffset = S("offset");
    auto kAddr = S("addr");
    auto smemPtrTy = ptr_ty(ctx, 3);
    auto bitWidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);

    // Map onto offsets (contiguous part) and addr (non-contiguous part)
    LinearLayout fullTile;
    // Contiguous tile
    LinearLayout tile;
    // ds_read_tr*_b64 performs a cooperative transposed load across 16
    // threads. The instruction processes an Nx16 tile (N=4 for 16-bit, N=8 for
    // 8-bit). The loaded tile is re-packed/transposed where lane i will
    // receive the i-th column.
    //
    // Loaded tile layout (input):     Register layout (output after transpose):
    //     K0  K1  ... K15               R0  R1  R2  R3
    // M0[ ............... ]    =>  T0 [ .   .   .   . ]
    // M1[ ............... ]        T1 [ .   .   .   . ]
    // M2[ ............... ]        ...
    // M3[ ............... ]        T15[ .   .   .   . ]
    //
    // Each lane loads 64 contiguous bits from LDS. After the transpose,
    // lane i receives column i from the input (elements strided by 16
    // the loaded tile).
    //
    // For example with N=4 (16-bit):
    // - Lane 0 receives elements from column 0: originally at [t0,t4,t8,t12]
    // - Lane 1 receives elements from column 1: originally at [t0,t4,t8,t12]
    //   These are the second 16 bits loaded by the same lanes before repacking
    // - Lane 4 receives elements from column 4: originally at [t1,t5,t9,t13]
    //
    // Note that there is no restriction on where elements are loaded
    // from, only that each lane needs to load 64 contiguous bits from shared
    // memory. We require N number of lanes to be contiguous since they read
    // consecutive 64 bits loaded from the same lanes.
    tile = LinearLayout::identity1D(ldsParams.tileSize, kLane, kOffset);

    const auto isaFamily = targetInfo.getISAFamily();
    // B8 types on gfx1250 require a different tile with double the contiguity
    bool doubleB8Contiguity =
        isaFamily == NANO::ISAFamily::GFX1250 && bitWidth == 8;
    const unsigned missingLanes =
        targetInfo.getWarpSize() / tile.getInDimSize(kLane);
    unsigned otherLanes = 1;
    if (isaFamily == NANO::ISAFamily::CDNA4) {
      otherLanes = (bitWidth == 8) ? 2 : 4;
    } else if (doubleB8Contiguity) {
      otherLanes = 2;
    }

    if (doubleB8Contiguity) {
      fullTile =
          tile * LinearLayout::identity1D(ldsParams.tileSize / 2, kReg, kAddr) *
          LinearLayout::identity1D(otherLanes, kLane, kAddr) *
          LinearLayout::identity1D(2, kReg, kAddr) *
          LinearLayout::identity1D(missingLanes / otherLanes, kLane, kAddr);
    } else {
      fullTile =
          tile * LinearLayout::identity1D(otherLanes, kLane, kAddr) *
          LinearLayout::identity1D(ldsParams.tileSize, kReg, kAddr) *
          LinearLayout::identity1D(missingLanes / otherLanes, kLane, kAddr);
    }
    // Add warp dimension so we can invert and compose with reps later
    fullTile *= LinearLayout::identity1D(1, kWarp, kAddr);

    if (cvt.getInDimSize(kReg) < fullTile.getInDimSize(kReg)) {
      return failure();
    }

    auto maybeQuot = divideLeft(cvt, tile);
    if (!maybeQuot.has_value()) {
      return failure();
    }

    // From here on we perform the lowering
    auto reps = zerosLike(tile) * maybeQuot.value();

    // Sanity check
    assert(fullTile.getInDimSize(kReg) * bitWidth == ldsParams.instBitWidth);

    // If we are lowering a subslice, the subslice offsets shall not touch the
    // contiguous part of the tile
    if (maskSpanAffineOffset & (tile.getOutDimSizeLog2(kOffset) - 1)) {
      return failure();
    }

    // fullTile.invert() is a map from kOffset, kAddr into kReg, kLane, kWarp
    // addrToOffset gives us a map from kAddr into kOffset, which is the map of
    // the addresses each lane should hold
    auto addrToOffset = fullTile.invert().compose(reps);
    // sanity check
    assert(addrToOffset.getInDimSizeLog2(kAddr) >= 3 &&
           addrToOffset.getInDimSizeLog2(kAddr) <= 6);

    LinearLayout addrLayout =
        LinearLayout({{kLane, addrToOffset.getBases().lookup(kAddr)},
                      {kWarp, reps.getBases().lookup(kWarp)}},
                     {{kOffset, reps.getOutDimSize(kOffset)}}, false);

    // Compute the bits that are moved by one instruction
    // Compute elements for which we can swap the xor by an add
    auto [nAdditive, permStrides] =
        actionAdditiveStrides(reps, addrLayout, maskSpanAffineOffset);
    reps = permStrides.apply(reps);

    // Perform computation in bytes, LLVM optimises this better
    assert(bitWidth >= 8);
    auto i8Tile =
        zerosLike(LinearLayout::identity1D(bitWidth / 8, kReg, kOffset));
    auto i8AddrLayout = i8Tile * addrLayout;

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto regBase =
        applyLinearLayout(
            loc, rewriter, i8AddrLayout,
            {{kReg, b.i32_val(0)}, {kLane, laneId}, {kWarp, warpId}})[0]
            .second;

    // It's fine that we don't compute the offset in bytes as affineOffset
    // will be folded into a constant
    auto affineOffsetI8 = b.mul(affineOffset, b.i32_val(bitWidth / 8));
    regBase = b.xor_(regBase, affineOffsetI8);

    auto lowerInst = [&](RewriterBase &rewriter, Location loc, Value vecAddr,
                         int idx, VectorType vTy) -> SmallVector<Value> {
      assert(bitWidth == 16 || bitWidth == 8);
      Value dsReadTr;
      // tr16 instructions return vectors of bf16/f16 while "tr8" instructions
      // return vectors of i32. Generate the corresponding i32 vector
      auto numElemsI32 = (vTy.getNumElements() * bitWidth / 32);
      auto vTyI32 = VectorType::get(numElemsI32, i32_ty);
      switch (targetInfo.getISAFamily()) {
      case NANO::ISAFamily::GFX1250: {
        if (bitWidth == 16) {
          dsReadTr = LLVM::createLLVMIntrinsicCallOp(
                         rewriter, loc, "llvm.amdgcn.ds.load.tr16.b128", {vTy},
                         {vecAddr})
                         .getResult(0);
        } else
          dsReadTr = LLVM::createLLVMIntrinsicCallOp(
                         rewriter, loc, "llvm.amdgcn.ds.load.tr8.b64", {vTyI32},
                         {vecAddr})
                         .getResult(0);
        break;
      }
      case NANO::ISAFamily::CDNA4: {
        if (bitWidth == 16) {
          dsReadTr =
              ROCDL::ds_read_tr16_b64::create(rewriter, loc, vTy, vecAddr);
        } else {
          dsReadTr =
              ROCDL::ds_read_tr8_b64::create(rewriter, loc, vTyI32, vecAddr);
        }
        break;
      }
      default:
        return {};
      }
      // addLocalLoadNoAliasScope removed - not needed for minimal nano backend
      Value vecVal = b.bitcast(dsReadTr, vTy);
      SmallVector<Value> loadedVals;
      for (int v = 0; v < vTy.getNumElements(); v++) {
        loadedVals.push_back(
            b.extract_element(llvmElemTy, vecVal, b.i32_val(v)));
      }

      return loadedVals;
    };

    // Elements per op
    auto elemsPerInstr = fullTile.getInDimSize(kReg);
    auto elemsPerVec = ldsParams.instBitWidth / bitWidth;
    auto vecTy = vec_ty(llvmElemTy, elemsPerVec);
    for (int i = 0; i < cvt.getInDimSize(kReg); i += nAdditive) {
      auto regIdx = reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}})[0].second;
      auto regIdxI8 = regIdx * (bitWidth / 8);
      Value offset = b.xor_(regBase, b.i32_val(regIdxI8));
      offset = applyPadding(loc, rewriter, offset, paddingShifts);
      for (int i2 = 0; i2 < nAdditive; i2 += elemsPerInstr) {
        // all these constants will go as immediate values to ds_read_tr
        auto regIdxAdd =
            reps.apply({{kReg, i2}, {kLane, 0}, {kWarp, 0}})[0].second;
        auto regIdxAddI8 = regIdxAdd * (bitWidth / 8);
        // `actionAdditiveStrides` forces `regIdxAddI8` and `offset` to be
        // bitwise disjoint, so we can calculate their padding contributions
        // separately.
        regIdxAddI8 = applyPadding(regIdxAddI8, paddingShifts);
        Value innerOffset = b.add(offset, b.i32_val(regIdxAddI8));
        auto vecAddr = b.gep(smemPtrTy, i8_ty, smemBase, innerOffset,
                             LLVM::GEPNoWrapFlags::inbounds);
        llvm::append_range(vals,
                           lowerInst(rewriter, loc, vecAddr, i + i2, vecTy));
      }
    }
    // apply all the inverse permutations in the reverse order
    assert(vals.size() == cvt.getInDimSize(kReg));
    vals = permStrides.inverse().apply(vals);

    return success();
  }

private:
  const NANO::TargetInfo &targetInfo;
};

// LocalLoadPackedTransposedOpConversion removed - TritonNANOGPU dialect not available

class BarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
public:
  BarrierOpConversion(const LLVMTypeConverter &converter,
                      const NANO::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::BarrierOp>(converter, benefit),
        targetInfo(targetInfo) {}
  using OpAdaptor = typename triton::gpu::BarrierOp::Adaptor;

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCDNA(targetInfo.getISAFamily()))
      return failure();
    // Check no other memory addrspaces are selected.
    // TensorRead/Write are allowed but noop.
    auto mask = triton::gpu::AddrSpace::Local |
                triton::gpu::AddrSpace::GlobalRead |
                triton::gpu::AddrSpace::GlobalWrite |
                triton::gpu::AddrSpace::TensorRead |
                triton::gpu::AddrSpace::TensorWrite;
    if ((op.getAddrSpace() & ~mask) != triton::gpu::AddrSpace::None)
      return failure();
    bool localBarrier = op.hasLocal();
    bool globalBarrier = op.hasGlobalRead() || op.hasGlobalWrite();
    if (globalBarrier)
      return failure();
    if (localBarrier) {
      // MemoryCounterWaitOp removed - use ROCDL wait directly
      // Wait for LDS operations to complete before barrier
      auto isaVersion = targetInfo.getIsaVersion();
      if (isaVersion.Major >= 12) {
        ROCDL::WaitDscntOp::create(rewriter, op->getLoc(), 0);
      } else {
        // Use s_waitcnt lgkmcnt(0) for pre-gfx12
        unsigned waitValue = 0xF70F; // lgkmcnt=0, vmcnt=max, expcnt=max
        ROCDL::SWaitcntOp::create(rewriter, op->getLoc(), waitValue);
      }
    }
    rewriter.replaceOpWithNewOp<ROCDL::SBarrierOp>(op);

    return success();
  }

private:
  const NANO::TargetInfo &targetInfo;
};

// MemoryCounterWaitOpConversion removed - TritonNANOGPU dialect not available

} // namespace

void mlir::triton::NANO::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  PatternBenefit transBenefit = PatternBenefit(benefit.getBenefit() + 1);
  PatternBenefit barrierBenefit = PatternBenefit(benefit.getBenefit() + 1);

  patterns.add<TransLocalLoadOpConversion>(typeConverter, targetInfo,
                                           transBenefit);
  // LocalLoadPackedTransposedOpConversion, MemoryCounterWaitOpConversion removed
  // - TritonNANOGPU dialect not available
  patterns.add<BarrierOpConversion>(typeConverter, targetInfo, barrierBenefit);
}
