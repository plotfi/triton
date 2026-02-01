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

Value llLoad(RewriterBase &rewriter, Location loc, Value ptr, Type elemTy,
             Value pred, Value falseVal, Value multicastMask,
             triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false);

void llStore(RewriterBase &rewriter, Location loc, Value ptr, Value val,
             Value pred, triton::CacheModifier cm = triton::CacheModifier::NONE,
             bool forceNoAliasAsyncLoads = false);

Type getPointerTypeWithShape(Value basePtr, Value offset);

unsigned getContiguity(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);
unsigned getContiguity(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass);

unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisAnalysisPass);
unsigned getVectorSize(Value ptr, Value offset,
                       ModuleAxisInfoAnalysis &axisAnalysisPass);

} // namespace mlir::LLVM::NANO

#endif // TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUTOLLVM_UTILITY_H_
