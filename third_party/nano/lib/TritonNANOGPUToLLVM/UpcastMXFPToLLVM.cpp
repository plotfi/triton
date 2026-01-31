#include "PatternTritonGPUOpToLLVM.h"

// TritonNANOGPU dialect removed - not needed for minimal nano backend

using namespace mlir;

void mlir::triton::NANO::populateUpcastMXFPToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  // UpcastMXFPOp pattern removed - TritonNANOGPU dialect not available
}
