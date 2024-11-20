#include <memory>
#include <stack>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Casting.h"
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
    if (auto reshapeRootCandOp = dyn_cast<triton::ReshapeOp>(op)) {

      // Find a root for a reshape chain
      if (!isa<triton::ReshapeOp>(reshapeRootCandOp.getSrc().getDefiningOp())) {
        std::stack<Operation*> stack;
        stack.push(reshapeRootCandOp);

        while (stack.size()) {

          auto curr = cast<triton::ReshapeOp>(stack.top());
          stack.pop();

          bool matchTypes =
            cast<RankedTensorType>(curr->getResult(0).getType()) ==
            cast<RankedTensorType>(reshapeRootCandOp.getOperand().getType());

          bool hitNonReshape = false;
          for (auto user : curr.getResult().getUsers()) {
            auto reshapeOp = dyn_cast<triton::ReshapeOp>(user);
            if (!reshapeOp) {
              hitNonReshape = true;
              continue;
            }
            stack.push(reshapeOp);
          }

          if (!matchTypes || !hitNonReshape)
            continue;

          curr->getResult(0).replaceAllUsesWith(reshapeRootCandOp.getSrc());
        }
      }
    }

    // Otherwise return the original one
    return op;
  }

  static Operation *peepCaptureDeadSelectOp(Operation *op,
                                            std::stack<Operation *> &eraser) {
    OpBuilder builder(op);
    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      auto condOpnd = selectOp.getCondition();
      auto trueOpnd = selectOp.getTrueValue();
      auto falseOpnd = selectOp.getFalseValue();

      bool isFalseValZero = false;

      if (auto constOp = llvm::dyn_cast_or_null<arith::ConstantOp>(falseOpnd.getDefiningOp())) {
        TypedAttr val = constOp.getValue();
        if (auto intOrFpEltAttr = llvm::dyn_cast<DenseIntOrFPElementsAttr>(val)) {
          auto type = intOrFpEltAttr.getType();
          auto elementType = intOrFpEltAttr.getElementType();

          if (ComplexType complexTy = llvm::dyn_cast<ComplexType>(elementType)) {
          } else if (elementType.isIntOrIndex()) {
          } else {
            auto values_or_nil = intOrFpEltAttr.tryGetValues<APFloat>();
            if (succeeded(values_or_nil)) {
              isFalseValZero = true;
              for (auto [inputIndex, inputValue] : llvm::enumerate(*values_or_nil)) {
                isFalseValZero = isFalseValZero && inputValue.isZero();
              }
            }
          }
        }
      }

      if (isFalseValZero) {
        llvm::errs() << "Found: REPLACE WITH MUL: ";
        selectOp->dump();
        auto condAsFloat =
          builder.create<arith::SIToFPOp>(selectOp.getLoc(),
                                          selectOp.getTrueValue().getType(),
                                          selectOp.getCondition());
        auto mulReplace = builder.create<arith::MulFOp>(selectOp.getLoc(), condAsFloat, selectOp.getTrueValue());

        op->getResult(0).replaceAllUsesWith(mulReplace);
        eraser.push(selectOp);
        return mulReplace;

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

  void visitOperation(Operation *op, peepFunc peep) {
    std::stack<Operation *> eraser;
    visitOperation(getOperation(), eraser, peep);

    // rewritedInfo.clear();
    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }
  }

  void runOnOperation() override {
    visitOperation(getOperation(), peepCaptureReshapeOp);
    visitOperation(getOperation(), peepCaptureDeadSelectOp);
  }
};

std::unique_ptr<Pass> triton::createTTIRPeepHolePass() {
  return std::make_unique<TTIRPeepHolePass>();
}
