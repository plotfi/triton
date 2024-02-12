#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"
#include <iterator>
#include <numeric>
#include <stack>

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {
  void
  setCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       llvm::MapVector<Operation *, Attribute> &layoutMap) {
    Value ptr = getMemAccessPtr(op);
    auto refTensorType = ptr.getType().cast<RankedTensorType>();

    LDBG("Considering op: " << *op);
    LLVM_DEBUG({
      DBGS() << "axis info of pointer: ";
      axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
    SmallVector<unsigned> order = argSort(contiguity);
    LDBG("order=[" << triton::join(order, ", ") << "]");

    auto matchesShape = [&refTensorType](const Value &val) {
      auto rttType = val.getType().dyn_cast<RankedTensorType>();
      return rttType && rttType.getShape() == refTensorType.getShape();
    };

    // The desired divisibility is the maximum divisibility among all dependent
    // pointers which have the same shape and order as `ptr`.
    llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
    memAccessesSameOrder.insert(op);
    if (ptr.getDefiningOp()) {
      for (Operation *use : mlir::multiRootGetSlice(op)) {
        Value val = getMemAccessPtr(use);
        if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
          continue;
        auto currOrder =
            argSort(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
        if (order == currOrder) {
          LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
          memAccessesSameOrder.insert(use);
        }
      }
    }

    auto shapePerCTA = triton::gpu::getShapePerCTA(refTensorType);
    LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;

    unsigned perThread = getNumElementsPerThread(op, order, axisInfoAnalysis);
    LDBG("perThread for op: " << perThread);

    for (Operation *opSameOrder : memAccessesSameOrder) {
      if (opSameOrder == op)
        continue;
      unsigned currPerThread =
          getNumElementsPerThread(opSameOrder, order, axisInfoAnalysis);
      LDBG("perThread for opSameOrder: " << currPerThread);
      perThread = std::max(perThread, currPerThread);
    }

    perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
    LDBG("perThread: " << perThread);

    if (!dyn_cast<triton::LoadOp>(op)) {
      // For ops that can result in a global memory write, we should enforce
      // that each thread handles at most 128 bits, which is the widest
      // available vectorized store op; otherwise, the store will have "gaps"
      // in the memory write at the warp level, resulting in worse performance.
      // For loads, we can expect that the gaps won't matter due to the L1
      // cache.
      unsigned elemNumBits = getElementBitWidth(refTensorType);
      perThread = std::min<int>(
          perThread, getNumElementsPerThread(op, order, axisInfoAnalysis));
    }
    SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[order[0]] = perThread;

    auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    layoutMap[op] = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = type.cast<RankedTensorType>();
    return RankedTensorType::get(tensorType.getShape(),
                                 tensorType.getElementType(), encoding);
  }

  void coalesceOp(Attribute encoding, Operation *op) {
    OpBuilder builder(op);
    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    SmallVector<Value, 4> newArgs;
    for (auto operand : op->getOperands()) {
      auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
      if (tensorType &&
          !tensorType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
        Type newType = getNewType(tensorType, encoding);
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), newType, operand));
      } else {
        newArgs.push_back(operand);
      }
    }

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = isa<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
      newTypes.push_back(isAsync ? t : getNewType(t, encoding));
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create(op->getLoc(), op->getName().getIdentifier(), newArgs,
                       newTypes, op->getAttrs());

    // Cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  void runOnOperation() override {
    expandLocalLoads();

    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    llvm::MapVector<Operation *, Attribute> layoutMap;
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;
      // We only convert `tensor<tt.ptr<>>` or `tt.ptr<tensor<>>` load/store
      bool isPtrTensor = false, isTensorPointer = false;
      if (auto tensorType = ptr.getType().dyn_cast<RankedTensorType>())
        isPtrTensor = tensorType.getElementType().isa<PointerType>();
      if (auto ptrType = ptr.getType().dyn_cast<PointerType>())
        isTensorPointer = ptrType.getPointeeType().isa<RankedTensorType>();
      if (!isPtrTensor && !isTensorPointer)
        return;
      auto mod = curr->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      setCoalescedEncoding(axisInfoAnalysis, curr, numWarps, threadsPerWarp,
                           layoutMap);
    });

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    for (auto &kv : layoutMap) {
      coalesceOp(kv.second, kv.first);
    }
  }

  void expandLocalLoads() {
    ModuleOp moduleOp = getOperation();

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    std::stack<Operation *> eraser;
    moduleOp.walk([&](Operation *op) {
      auto loadOp = dyn_cast<triton::LoadOp>(op);
      if (!loadOp)
        return;

      auto ptr = op->getOperand(0);
      auto loadOpPtr = dyn_cast<triton::LoadOp>(ptr.getDefiningOp());
      if (!loadOpPtr || !loadOpPtr.getIsSharedMem())
        return;

      OpBuilder builder(op);

      // Convert the tensor operand of the tt.load(isShared = true) into a
      // MemDescType
      auto tensorType =
          loadOpPtr->getResult(0).getType().cast<RankedTensorType>();
      auto elementType = tensorType.getElementType();

      auto srcBlocked =
          tensorType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();

      auto oldOrder = mlir::triton::gpu::getOrder(srcBlocked);
      std::vector<unsigned> newOrder = {1, oldOrder[0]};

      auto oldShape = tensorType.getShape();
      std::vector<int64_t> newShape = {1, oldShape[0]};

      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(moduleOp);
      int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);

      SmallVector<unsigned, 4> sizePerThread(tensorType.getRank() + 1, 1);
      auto newBlockedLayout = triton::gpu::BlockedEncodingAttr::get(
          tensorType.getContext(), newShape, sizePerThread, newOrder, numWarps,
          threadsPerWarp, numCTAs);

      auto reshapeOp = builder.create<triton::ReshapeOp>(
          loadOp.getLoc(),
          RankedTensorType::get(newShape, elementType, newBlockedLayout),
          loadOpPtr.getResult(), false /* allowReorder */);

      auto CTALayout = newBlockedLayout.getCTALayout();
      auto newLayout = mlir::triton::gpu::SharedEncodingAttr::get(
          tensorType.getContext(), newShape, newOrder, CTALayout, elementType);

      auto memDescType =
          triton::MemDescType::get(newShape, elementType, newLayout);
      loadOpPtr.setIsSharedMem(false);

      // Replace the original tt.load(isShared = true) with the
      // triton_gpu.local_alloc(tt.load(isShared = false)) instruction sequence
      auto localAllocOp = builder.create<triton::gpu::LocalAllocOp>(
          loadOpPtr.getLoc(), memDescType, reshapeOp.getResult());

      auto loadOpType = cast<RankedTensorType>(loadOp.getType());
      auto shape = memDescType.getShape();
      auto enc = newBlockedLayout;

      // Replace the tt.load from SMEM with a triton_gpu.local_load
      auto localLoadOp = builder.create<triton::gpu::LocalLoadOp>(
          loadOp.getLoc(),
          RankedTensorType::get(shape, memDescType.getElementType(), enc),
          localAllocOp);

      auto reshapeOp2 = builder.create<triton::ReshapeOp>(
          loadOp.getLoc(),
          RankedTensorType::get(oldShape, elementType, srcBlocked),
          localLoadOp.getResult(), false /* allowReorder */);

      op->getResult(0).replaceAllUsesWith(reshapeOp2);

      // Erase the original operation
      eraser.push(op);
    });

    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }
  }
};

std::unique_ptr<Pass> mlir::triton::gpu::createCoalescePass() {
  return std::make_unique<CoalescePass>();
}
