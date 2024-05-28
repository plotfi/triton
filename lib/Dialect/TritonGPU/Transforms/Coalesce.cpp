#include <iterator>
#include <numeric>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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
    auto refTensorType = cast<RankedTensorType>(ptr.getType());

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
      auto rttType = dyn_cast<RankedTensorType>(val.getType());
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
    RankedTensorType tensorType = cast<RankedTensorType>(type);
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
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (tensorType &&
          !isa<triton::gpu::SharedEncodingAttr>(tensorType.getEncoding())) {
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
      // We only convert `tensor<tt.ptr<>>` load/store
      bool isPtrTensor = false;
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
        isPtrTensor = isa<PointerType>(tensorType.getElementType());
      if (!isPtrTensor)
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

    static unsigned loadCount = 0;

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    std::stack<Operation *> eraser;
    moduleOp.walk([&](Operation *op) {
      auto loadOp = dyn_cast<triton::LoadOp>(op);
      if (!loadOp)
        return;

      if (loadCount == 8) {
        llvm::errs() << "last load before crash!\n";
      }

      llvm::errs() << "FOUND LOAD " << loadCount++ << ": ";
      loadOp->dump();

      if (loadCount != 9) {
        return;
      }

      if (!op->getNumOperands() || !op->getOperand(0).getDefiningOp())
        return;

      auto ptr = op->getOperand(0);
      auto prefetchLoad = dyn_cast<triton::LoadOp>(ptr.getDefiningOp());

      Value offset = nullptr;

      // Handle the base + offset case
      auto addIOp = dyn_cast<mlir::arith::AddIOp>(ptr.getDefiningOp());
      auto addFOp = dyn_cast<mlir::arith::AddFOp>(ptr.getDefiningOp());
      Type resultType;
      Operation *tailAllocOp = nullptr;
      Operation *addOp = addIOp ? addIOp : addFOp;
      bool hasBroadcast = false;

      // Handle the base + offset case
      if (!prefetchLoad && addOp) {

        triton::BroadcastOp broadcast =
            dyn_cast<triton::BroadcastOp>(addOp->getOperand(0).getDefiningOp());
        if (broadcast)
          hasBroadcast = true;

        mlir::triton::SplatOp earlySplat =
          broadcast ?
          dyn_cast<mlir::triton::SplatOp>(broadcast->getOperand(0).getDefiningOp()) :
          dyn_cast<mlir::triton::SplatOp>(addOp->getOperand(0).getDefiningOp());


        arith::ExtSIOp ext = broadcast
                                 ? dyn_cast<arith::ExtSIOp>(
                                       broadcast->getOperand(0).getDefiningOp())
                                 : dyn_cast<arith::ExtSIOp>(
                                       addOp->getOperand(0).getDefiningOp());

        arith::ExtFOp extf = broadcast
                                 ? dyn_cast<arith::ExtFOp>(
                                       broadcast->getOperand(0).getDefiningOp())
                                 : dyn_cast<arith::ExtFOp>(
                                       addOp->getOperand(0).getDefiningOp());

        if (auto splat = dyn_cast<mlir::triton::SplatOp>(
                addOp->getOperand(1).getDefiningOp())) {
          auto sitofp =
              dyn_cast<arith::SIToFPOp>(splat.getOperand().getDefiningOp());
          offset = sitofp ? sitofp.getOperand() : nullptr;
        }

        offset = offset ? offset : addOp->getOperand(1);

        if (auto offsetFP = dyn_cast<arith::SIToFPOp>(offset.getDefiningOp())) {
          offset = offsetFP.getOperand();
        }

        prefetchLoad =
            extf ? dyn_cast<triton::LoadOp>(extf.getOperand().getDefiningOp()->
                getOperands()[0].getDefiningOp()->
                getOperands()[0].getDefiningOp()->getOperands()[0].getDefiningOp()) : (
            ext ? dyn_cast<triton::LoadOp>(ext.getOperand().getDefiningOp())
                : dyn_cast<triton::LoadOp>(
                      addOp->getOperand(0).getDefiningOp()));
        prefetchLoad = dyn_cast<triton::LoadOp>(
            addOp->
            getOperands()[0].getDefiningOp()->
            getOperands()[0].getDefiningOp()->
            getOperands()[0].getDefiningOp()->
            getOperands()[0].getDefiningOp());
        tailAllocOp = broadcast ? broadcast : (ext ? ext : prefetchLoad);
        tailAllocOp = earlySplat ? earlySplat : tailAllocOp;
        resultType = tailAllocOp->getResult(0).getType();
      }

      if (!prefetchLoad || !offset)
        return;

      llvm::errs() << "FOUND PRE-LOAD!!\n";

      OpBuilder builder(op);
      OpBuilder builder_prefetch(prefetchLoad);

      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(moduleOp);
      int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);

      auto prefetchLoadTensorType = mlir::dyn_cast<RankedTensorType>(
          prefetchLoad->getResult(0).getType());
      auto tensorType = mlir::dyn_cast<RankedTensorType>(resultType);
      auto elementType = tensorType.getElementType();

      auto prefetchLoadBlocked =
          mlir::dyn_cast<triton::gpu::BlockedEncodingAttr>(
              prefetchLoadTensorType.getEncoding());
      auto srcBlocked = mlir::dyn_cast<triton::gpu::BlockedEncodingAttr>(
          tensorType.getEncoding());
      auto oldOrder = mlir::triton::gpu::getOrder(srcBlocked);
      auto oldShape = tensorType.getShape();

#define LOCAL_GATHER 1
#define RESHAPE !LOCAL_GATHER

#if LOCAL_GATHER
      unsigned newOrderDim1 = 0;
      int64_t newShapeDim1 = oldShape[1];
      std::vector<int64_t> newShape = {newShapeDim1, oldShape[0]};
      auto tensorRank = tensorType.getRank();
      std::vector<unsigned> newOrder = {newOrderDim1, oldOrder[0]};
      auto blockedOrder = oldOrder;
      auto sharedMemoryOrder = newOrder;
      auto sharedMemoryShape = newShape;
#else
      unsigned newOrderDim1 = 1;
      int64_t newShapeDim1 = oldShape.size() < 2 ? 1 : oldShape[1];
      std::vector<int64_t> newShape = {newShapeDim1, oldShape[0]};
      auto tensorRank = tensorType.getRank() + 1;
      std::vector<unsigned> newOrder = {newOrderDim1, oldOrder[0]};
      auto blockedOrder = newOrder;
      auto sharedMemoryOrder = newOrder;
      auto sharedMemoryShape = newShape;
#endif

      SmallVector<unsigned, 4> sizePerThread(tensorRank, 1);
      auto sharedBlockedLayout = triton::gpu::BlockedEncodingAttr::get(
          tensorType.getContext(), sharedMemoryShape, sizePerThread,
          blockedOrder, numWarps, threadsPerWarp, numCTAs);

      if (hasBroadcast) {
        auto a = sharedMemoryShape[0];
        sharedMemoryShape[0] = sharedMemoryShape[1];
        sharedMemoryShape[1] = a;
      }

      auto newLayout = mlir::triton::gpu::SharedEncodingAttr::get(
          tensorType.getContext(), sharedMemoryShape, sharedMemoryOrder,
          sharedBlockedLayout.getCTALayout(), elementType);

      auto memDescType =
          triton::MemDescType::get(sharedMemoryShape, elementType, newLayout);

      Operation *nextOp = tailAllocOp;

#if RESHAPE
      auto reshapeOp = builder.create<triton::ReshapeOp>(
          prefetchLoad.getLoc(),
          RankedTensorType::get(sharedMemoryShape, elementType,
                                sharedBlockedLayout),
          nextOp->getResults()[0], false /* allowReorder */);
      reshapeOp.getOperation()->moveAfter(nextOp);
      nextOp = reshapeOp;

      auto localGatherShape =
        RankedTensorType::get(memDescType.getShape(),
                              memDescType.getElementType(),
                              sharedBlockedLayout);
#else
      auto localGatherShape = tensorType;
#endif

      auto localAllocOp = builder_prefetch.create<triton::gpu::LocalAllocOp>(
          prefetchLoad.getLoc(), memDescType, nextOp->getResults()[0]);
      localAllocOp.getOperation()->moveAfter(nextOp);

#if !RESHAPE
      nextOp = localAllocOp;
#endif
      tailAllocOp->getResult(0).replaceAllUsesExcept(localAllocOp, nextOp);

      localAllocOp->getParentOp()->dump();

#if LOCAL_GATHER
      auto localGather = builder.create<triton::gpu::LocalGatherOp>(
          loadOp.getLoc(), localGatherShape,
          localAllocOp.getResult(), offset);
      nextOp = localGather;
#else
      llvm::SmallVector<Value, 2> offsetsVal = {
          builder.create<arith::ConstantIntOp>(loadOp.getLoc(), 0, 32), offset};
      auto memDescSubview = builder.create<triton::gpu::MemDescSubviewOp>(
          loadOp.getLoc(), memDescType, localAllocOp, offsetsVal);

      auto localLoadOp = builder.create<triton::gpu::LocalLoadOp>(
          loadOp.getLoc(),
          RankedTensorType::get(memDescType.getShape(),
                                memDescType.getElementType(),
                                sharedBlockedLayout),
          memDescSubview);
      nextOp = localLoadOp;
#endif

      auto reshapeOpOut = builder.create<triton::ReshapeOp>(
          loadOp.getLoc(),
          RankedTensorType::get(oldShape, elementType, srcBlocked),
          nextOp->getResults()[0], false /* allowReorder */);
      nextOp = reshapeOpOut;

      op->getResult(0).replaceAllUsesWith(nextOp->getResults()[0]);
      // Erase the original operation
      eraser.push(addOp);
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
