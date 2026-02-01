// Minimal TritonNANOGPU Dialect - All operators removed for simple vector add backend

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// clang-format off
#include "Dialect/TritonNANOGPU/IR/Dialect.h"
#include "Dialect/TritonNANOGPU/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::nanogpu;

void mlir::triton::nanogpu::TritonNANOGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/TritonNANOGPU/IR/TritonNANOGPUAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/TritonNANOGPU/IR/Ops.cpp.inc"
      >();

}

#include "Dialect/TritonNANOGPU/IR/TritonNANOGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonNANOGPU/IR/TritonNANOGPUAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/TritonNANOGPU/IR/Ops.cpp.inc"

// All operator implementations have been removed for minimal vector add backend.
// The dialect infrastructure (attributes, types, initialization) is kept for compatibility.
