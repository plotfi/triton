#include "AsyncUtility.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNANOGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <tuple>

using namespace mlir;
using namespace mlir::triton::gpu;

// TritonNANOGPU dialect MaskedLoadOp and MaskedStoreOp removed - not needed for minimal backend

namespace mlir::triton::NANO {

void populateMaskedOpsToLLVMPatterns(RewritePatternSet &patterns,
                                     const TargetInfo &targetInfo) {
  // MaskedLoadOp and MaskedStoreOp patterns removed - TritonNANOGPU dialect not available
}
} // namespace mlir::triton::NANO
