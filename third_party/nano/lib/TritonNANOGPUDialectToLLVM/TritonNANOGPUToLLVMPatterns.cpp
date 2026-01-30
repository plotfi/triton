#include "third_party/nano/include/TritonNANOGPUToLLVM/PatternTritonNANOGPUToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::NANO {
void populateTritonNANOGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const NANO::TargetInfo &targetInfo,
                                        PatternBenefit benefit) {
  populateExtractSliceOpToLLVMPatterns(typeConverter, patterns, benefit);
  populateInThreadTransposeOpToTTGPatterns(patterns, benefit);
  populateConcatOpToLLVMPatterns(typeConverter, patterns, benefit);
  populateScaledUpcastOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                       benefit);
}
} // namespace mlir::triton::NANO
