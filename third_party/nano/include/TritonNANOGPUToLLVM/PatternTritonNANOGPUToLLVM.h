#ifndef TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_PATTERNTRITONNANOGPUTOLLVM_H_
#define TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_PATTERNTRITONNANOGPUTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "third_party/nano/lib/TritonNANOGPUToLLVM/TargetInfo.h"

namespace mlir::triton::NANO {

void populateExtractSliceOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit);

} // namespace mlir::triton::NANO

#endif // TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_PATTERNTRITONNANOGPUTOLLVM_H_
