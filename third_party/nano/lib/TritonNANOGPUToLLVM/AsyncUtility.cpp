#include "AsyncUtility.h"

#include "TargetInfo.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::NANO {
namespace {
constexpr const char *syncedViaAsyncWaitAttrName =
    "ttg.amdg.syncedViaAsyncWait";
// Traverses the def-chain including control flow of the token and returns true
// if all defining operations are an AsyncWait
bool comesFromAsyncWait(Value token) {
  if (auto defOp = token.getDefiningOp()) {
    return isa<triton::gpu::AsyncWaitOp>(defOp);
  }

  auto blockArg = dyn_cast<BlockArgument>(token);
  // If the token has no defining op and is not an BlockArgument bail out
  if (!blockArg) {
    return false;
  }

  auto block = blockArg.getOwner();
  auto argId = blockArg.getArgNumber();

  auto destOperandFromAsyncWait = [argId](auto &&operands) {
    assert(argId < operands.size());
    return comesFromAsyncWait(operands[argId]);
  };

  // Check all predecessor block's terminator and follow the passed value at
  // argId to see if they are immediately an AsyncWait.
  for (auto *pred : block->getPredecessors()) {
    auto terminator = pred->getTerminator();
    if (auto br = dyn_cast<BranchOpInterface>(terminator)) {
      for (auto successor : llvm::enumerate(br->getSuccessors())) {
        if (block != successor.value())
          continue;
        auto operands = br.getSuccessorOperands(successor.index());
        if (!destOperandFromAsyncWait(operands))
          return false;
      }
    } else {
      return false;
    }
  }
  return true;
}
} // namespace

void annotateLocalLoadsSyncedViaAsyncWait(ModuleOp mod) {
  auto *ctx = mod->getContext();

  mod->walk([&](Operation *op) {
    TypeSwitch<Operation *, void>(op)
        .Case<triton::gpu::LocalLoadOp>([&](auto loadOp) {
          if (loadOp->hasAttr(syncedViaAsyncWaitAttrName))
            return;
          Value token = loadOp.getToken();
          bool isSyncedViaAsyncWait = token && comesFromAsyncWait(token);
          loadOp->setAttr(syncedViaAsyncWaitAttrName,
                          BoolAttr::get(ctx, isSyncedViaAsyncWait));
        });
  });
}

bool isSyncedViaAsyncWait(Operation *op) {
  assert(op);

  auto attr = op->getAttr(syncedViaAsyncWaitAttrName);
  if (!attr) {
    op->emitRemark("has no async sync information attached to it which "
                   "might negatively affect performance. Run "
                   "annotateLocalLoadSyncedViaAsyncWait first");
    return false;
  }
  return cast<BoolAttr>(attr).getValue();
}

namespace {
LLVM::AliasScopeDomainAttr getLoadScopeDomain(MLIRContext *ctx) {
  Builder b(ctx);
  return b.getAttr<LLVM::AliasScopeDomainAttr>(
      b.getStringAttr("amdg.AsyncOps"),
      b.getStringAttr(
          "Domain to hold alias scopes to specify aliasing information between "
          "AsyncCopyGlobalToLocal and LocalLoad ops"));
}

LLVM::AliasScopeAttr getAsyncCopyScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("amdg.AsyncCopies");
  auto desc = b.getStringAttr(
      "Scope containing all AsyncCopyGlobalToLocal ops");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getLoadScopeDomain(ctx), desc);
}

LLVM::AliasScopeAttr getLoadCopyScope(MLIRContext *ctx) {
  Builder b(ctx);
  auto name = b.getStringAttr("amdg.LocalLoads");
  auto desc = b.getStringAttr("Scope containing all LocalLoad ops");
  return b.getAttr<LLVM::AliasScopeAttr>(name, getLoadScopeDomain(ctx), desc);
}
} // namespace

void addAsyncCopyAliasScope(LLVM::AliasAnalysisOpInterface directToLdsOp) {
  auto ctx = directToLdsOp->getContext();
  Builder b(ctx);
  directToLdsOp.setAliasScopes(b.getArrayAttr(getAsyncCopyScope(ctx)));
}

// addLocalLoadNoAliasScope removed - not needed for minimal nano backend

unsigned
fitToValidDirectToLdsVecSize(unsigned maxVecSize, unsigned elemBitwidth,
                             const triton::NANO::TargetInfo &targetInfo) {
  while (maxVecSize > 0 && !targetInfo.supportsDirectToLdsLoadBitWidth(
                               maxVecSize * elemBitwidth)) {
    maxVecSize /= 2;
  }
  return maxVecSize;
}

} // namespace mlir::triton::NANO
