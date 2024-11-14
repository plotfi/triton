#include <memory>
#include <stack>

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

class TTIRPeepHolePass : public TritonTTIRPeepHoleBase<TTIRPeepHolePass> {
private:
  // DenseMap<Value, RewritedInfo> rewritedInfo;

public:
  using peepFunc = Operation *(Operation *op, std::stack<Operation *> &eraser);

  static Operation *peepCaptureReshapeOp(Operation *op,
                                         std::stack<Operation *> &eraser) {
    OpBuilder builder(op);
    if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(op)) {
      if (auto defOp =
              dyn_cast<triton::ReshapeOp>(reshapeOp.getSrc().getDefiningOp())) {
        llvm::errs() << "Found reshape of a RESHAPE OP:\n";
        defOp->dump();
        reshapeOp->dump();
      }
    }
    // Otherwise return the original one
    return op;
  }

  void visitOperation(Operation *op, std::stack<Operation *> &eraser,
                      peepFunc peep) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &nestedOp : llvm::make_early_inc_range(block)) {
          // llvm::errs() << "DO STUFF\n";
          if (auto newOp = peep(&nestedOp, eraser)) {
            visitOperation(newOp, eraser, peep);
          }
        }
      }
    }
  }

  void runOnOperation() override {

    std::stack<Operation *> eraser;
    visitOperation(getOperation(), eraser, peepCaptureReshapeOp);

    // rewritedInfo.clear();
    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }
  }
};

std::unique_ptr<Pass> triton::createTTIRPeepHolePass() {
  return std::make_unique<TTIRPeepHolePass>();
}
