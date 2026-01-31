#include "third_party/nano/include/TritonNANOGPUToLLVM/PatternTritonNANOGPUToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::NANO {
void populateTritonNANOGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const NANO::TargetInfo &targetInfo,
                                        PatternBenefit benefit) {
  // All NANO dialect ops have been removed for minimal vector add backend.
  // No patterns to add.
}
} // namespace mlir::triton::NANO
