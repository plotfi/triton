// Minimal TritonNANOGPU Ops - All operators removed for simple vector add backend
// This file is kept for dialect infrastructure compatibility

#ifndef TRITON_NANOGPU_OPS
#define TRITON_NANOGPU_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUOpInterfaces.td"

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "TritonNANOGPUDialect.td"
include "TritonNANOGPUAttrDefs.td"


class TT_AMDGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonNANOGPU_Dialect, mnemonic, !listconcat(traits, [])>;

// All operators have been removed for minimal vector add backend.
// The dialect infrastructure (attributes, types) is kept for compatibility.

#endif
