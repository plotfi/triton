// Minimal TargetInfo.cpp - Only what's needed for vector add
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::NANO {

int TargetInfo::getWarpSize() const { return 64; }
void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  llvm_unreachable("warpSync not supported in nano backend");
}
std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  llvm_unreachable("MulhiFuncName not supported in nano backend");
}

bool TargetInfo::supportMaximumMinimum() const { return false; }

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  llvm_unreachable("ballot not supported in nano backend");
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  llvm::report_fatal_error("Barrier not supported in nano backend");
}

void TargetInfo::storeDShared(RewriterBase &, Location, Value,
                              std::optional<Value>, Value, Value) const {
  llvm::report_fatal_error("Shared memory not supported in nano backend");
}

Value TargetInfo::loadDShared(RewriterBase &, Location, Value,
                              std::optional<Value>, Type, Value,
                              Operation *) const {
  llvm::report_fatal_error("Shared memory not supported in nano backend");
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("shuffleXor not supported in nano backend");
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  llvm_unreachable("shuffleUp not supported in nano backend");
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("shuffleIdx not supported in nano backend");
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  llvm_unreachable("shuffleIdx not supported in nano backend");
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  llvm_unreachable("permute not supported in nano backend");
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
  llvm::report_fatal_error("getAddressSpace not supported in nano backend");
}

bool TargetInfo::supportVectorizedAtomics() const { return false; }

bool TargetInfo::supportsMultiCTALaunch() const {
  return false;
}

} // namespace mlir::triton::NANO
