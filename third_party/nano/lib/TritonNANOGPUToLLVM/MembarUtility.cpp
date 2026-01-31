#include "TritonNANOGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::NANO {
namespace {
// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2,
                                       Allocation *allocation) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
  };
  auto getMemdescValue = [](Operation *op) -> Value {
    return llvm::TypeSwitch<Operation *, Value>(op)
        .Case<triton::gpu::AsyncCopyGlobalToLocalOp>(
            [](auto op) { return op.getResult(); })
        .Case<triton::gpu::LocalLoadOp>([](auto op) { return op.getSrc(); })
        .Default([](Operation *) { return Value(); });
  };

  // Early return if neither or both operands are an AsyncLoad
  if (isAsyncLoad(op1) == isAsyncLoad(op2)) {
    return false;
  }

  Value op1Memdesc = getMemdescValue(op1);
  Value op2Memdesc = getMemdescValue(op2);
  if (!op1Memdesc || !op2Memdesc)
    return false;
  auto op1BufferIds = allocation->getBufferIds(op1Memdesc);
  auto op2BufferIds = allocation->getBufferIds(op2Memdesc);

  // Check if operations access the same buffer
  bool sameBuffer = llvm::any_of(
      op1BufferIds, [&](auto id) { return op2BufferIds.count(id); });

  if (!sameBuffer)
    return false;

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
}

// TritonNANOGPU dialect barrier operations removed
bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  // No nanogpu barrier operations to filter
  return false;
}
} // namespace

bool membarFilter(Operation *op1, Operation *op2, Allocation *allocation) {
  return (filterAsyncLocalLoadsDependencies(op1, op2, allocation) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}
} // namespace mlir::triton::NANO
