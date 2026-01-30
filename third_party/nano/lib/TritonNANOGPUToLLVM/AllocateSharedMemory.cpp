#include "Analysis/NANOGPUAllocation.h"
#include "TritonNANOGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::NANO;

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATENANOGPUSHAREDMEMORY
#include "TritonNANOGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

struct AllocateNANOGPUSharedMemory
    : public mlir::triton::impl::AllocateNANOGPUSharedMemoryBase<
          AllocateNANOGPUSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod, AMDAllocationAnalysisScratchSizeFn);

    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace
