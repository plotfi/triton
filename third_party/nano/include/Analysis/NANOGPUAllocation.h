#ifndef TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
#define TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::NANO {

unsigned getConvertLayoutScratchInBytes(RankedTensorType srcTy,
                                        RankedTensorType dstTy);

unsigned AMDAllocationAnalysisScratchSizeFn(Operation *op);

std::pair</*inVec*/ unsigned, /*outVec*/ unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy);

} // namespace mlir::triton::NANO

#endif // TRITONAMD_ANALYSIS_AMDGPU_ALLOCATION_H
