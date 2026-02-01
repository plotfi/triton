// Minimal TargetInfo.cpp - Only what's needed for vector add
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::NANO {

// AMD SPECIFIC TargetInfo code:
int TargetInfo::getWarpSize() const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return 64;
  case ISAFamily::GFX1250:
    return 32;
  default:
    return 32;
  }
}
void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.wave.barrier", {},
                                  {});
}
std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  return resultElementTy.isInteger(32) ? "llvm.amdgcn.mul.hi.u32"
                                       : "llvm.amdgcn.mul.hi.u64";
}

bool TargetInfo::supportMaximumMinimum() const { return false; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  llvm_unreachable("ballot not supported");
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  TritonLLVMOpBuilder(loc, rewriter).barrier(targets);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value())
    llvm::report_fatal_error("Cross-CTA shared memory not supported");
  mlir::LLVM::NANO::llStore(rewriter, loc, ptr, val, pred);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *) const {
  if (ctaId.has_value())
    llvm::report_fatal_error("Cross-CTA shared memory not supported");
  Value falseVal = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                            rewriter.getZeroAttr(elemTy));
  return mlir::LLVM::NANO::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal,
                                  {}, triton::CacheModifier::NONE, false);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("shuffleXor not supported");
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  llvm_unreachable("shuffleUp not supported");
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("shuffleIdx not supported");
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  llvm_unreachable("shuffleIdx not supported");
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  llvm_unreachable("permute not supported");
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::NANO::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  return false;
}

void TargetInfo::printf(RewriterBase &, Value, int, ValueRange,
                        ArrayRef<bool>) const {}

void TargetInfo::printf(RewriterBase &, StringRef, ValueRange,
                        ArrayRef<bool>) const {}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc, StringRef,
                            StringRef, StringRef, int) const {
  LLVM::Trap::create(rewriter, loc);
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace))
    return 3;
  llvm::report_fatal_error("Only SharedMemorySpace supported");
}

bool TargetInfo::supportVectorizedAtomics() const { return true; }

bool TargetInfo::supportsMultiCTALaunch() const {
  return getISAFamily() == ISAFamily::GFX1250;
}

} // namespace mlir::triton::NANO
