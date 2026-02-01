// ConvertLayoutOpToLLVM.cpp - Minimal implementation for vector add backend
// ConvertLayoutOpPermlaneSwap removed - not needed for simple kernels

#include "PatternTritonGPUOpToLLVM.h"

void mlir::triton::NANO::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // No NANO-specific convert layout patterns needed for minimal backend
}
