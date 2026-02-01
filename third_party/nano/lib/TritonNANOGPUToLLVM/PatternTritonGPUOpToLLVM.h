#ifndef TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
#define TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::NANO {
void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfo &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit);

void adjustModeRegister(ModuleOp mod, const TargetInfo &targetInfo);

void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

void populateTritonNANOGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const NANO::TargetInfo &,
                                        PatternBenefit benefit);

void populateWarpIdOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   const TargetInfo &targetInfo,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);
void populateFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit);

} // namespace mlir::triton::NANO

#endif // TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
