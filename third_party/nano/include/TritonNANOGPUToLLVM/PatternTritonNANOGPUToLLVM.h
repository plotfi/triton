#ifndef TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_PATTERNTRITONNANOGPUTOLLVM_H_
#define TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_PATTERNTRITONNANOGPUTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "third_party/nano/lib/TritonNANOGPUToLLVM/TargetInfo.h"

namespace mlir::triton::NANO {

// All NANO dialect ops have been removed for minimal vector add backend.
// This header is kept for API compatibility.

} // namespace mlir::triton::NANO

#endif // TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_PATTERNTRITONNANOGPUTOLLVM_H_
