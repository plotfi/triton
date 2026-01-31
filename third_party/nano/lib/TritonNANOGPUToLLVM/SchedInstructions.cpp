#include "TritonNANOGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONNANOGPUINSERTINSTRUCTIONSCHEDHINTS
#define GEN_PASS_DEF_TRITONNANOGPULOWERINSTRUCTIONSCHEDHINTS
#include "TritonNANOGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

#undef DEBUG_TYPE
#define DEBUG_TYPE "lower-insert-instruction-sched-hints"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

// TritonNANOGPU dialect InstructionSchedHint removed - passes are no-ops

namespace {

struct TritonNANOGPULowerInstructionSchedHints
    : public triton::impl::TritonNANOGPULowerInstructionSchedHintsBase<
          TritonNANOGPULowerInstructionSchedHints> {

  explicit TritonNANOGPULowerInstructionSchedHints(StringRef arch,
                                                  int32_t numStages) {
    this->arch = arch.str();
    this->numStages = numStages;
  }

  void runOnOperation() override {
    // No-op: InstructionSchedHint removed with TritonNANOGPU dialect
  }
};

struct TritonNANOGPUInsertInstructionSchedHints
    : public triton::impl::TritonNANOGPUInsertInstructionSchedHintsBase<
          TritonNANOGPUInsertInstructionSchedHints> {

  explicit TritonNANOGPUInsertInstructionSchedHints(StringRef variant) {
    this->variant = variant.str();
  }

  void runOnOperation() override {
    // No-op: InstructionSchedHint removed with TritonNANOGPU dialect
  }
};
} // namespace

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createTritonNANOGPULowerInstructionSchedHintsPass(StringRef arch,
                                                 int32_t numStages) {
  return std::make_unique<TritonNANOGPULowerInstructionSchedHints>(arch,
                                                                  numStages);
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonNANOGPUInsertInstructionSchedHintsPass(StringRef variant) {
  return std::make_unique<TritonNANOGPUInsertInstructionSchedHints>(variant);
}
} // namespace mlir::triton
