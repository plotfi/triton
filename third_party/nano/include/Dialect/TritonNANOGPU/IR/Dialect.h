#ifndef TRITON_THIRD_PARTY_NANO_INCLUDE_DIALECT_TRITONNANOGPU_IR_DIALECT_H_
#define TRITON_THIRD_PARTY_NANO_INCLUDE_DIALECT_TRITONNANOGPU_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// clang-format off
#include "nano/include/Dialect/TritonNANOGPU/IR/Dialect.h.inc"
#include "nano/include/Dialect/TritonNANOGPU/IR/TritonNANOGPUEnums.h.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "nano/include/Dialect/TritonNANOGPU/IR/TritonNANOGPUAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "nano/include/Dialect/TritonNANOGPU/IR/Ops.h.inc"

#endif // TRITON_THIRD_PARTY_NANO_INCLUDE_DIALECT_TRITONNANOGPU_IR_DIALECT_H_
