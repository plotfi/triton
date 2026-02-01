#ifndef TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUTOLLVM_UTILITY_H_

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::LLVM::NANO {

enum class MemoryOp { Load, Store };

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               ProgramIDDim axis);

// Loads from shared or global memory with predication.
// `otherElems` is used to mask out the elements that are not loaded
// forceNoAliasAsyncLoads parameter kept for API compatibility but is unused
Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, Value multicastMask,
             triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false);

// Stores to shared or global memory with predication.
// forceNoAliasAsyncLoads parameter kept for API compatibility but is unused
void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false);

// Return a tensor of pointers with the same type of `basePtr` and the same
// shape of `offset`
Type getPointerTypeWithShape(Value basePtr, Value offset);

// Get contiguity for a tensor pointer `ptr`
unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);

// Get contiguity for a scalar pointer `ptr` and a tensor `offset`
unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass);

// Determine the vector size of a tensor of pointers
unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);

// Given a scalar pointer and a tensor of offsets, determine the vector size
unsigned getVectorSize(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass);

} // namespace mlir::LLVM::NANO

#endif // TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUTOLLVM_UTILITY_H_
