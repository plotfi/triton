#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
using mlir::triton::ModuleAxisInfoAnalysis;

namespace mlir::LLVM::NANO {

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

} // namespace mlir::LLVM::NANO
