#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;

void mlir::triton::NANO::populateTensorPtrOpsToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  // TDM tensor descriptor operations removed - not needed for basic kernels
  return;
}
