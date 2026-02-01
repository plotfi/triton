// Minimal ElementwiseOpToLLVM.cpp - Only what's needed for f32 vector add
#include "TargetInfo.h"
#include "TritonNANOGPUToLLVM/TargetUtils.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using mlir::triton::gpu::appendOrGetExternFuncOp;
using mlir::triton::gpu::ElementwiseOpConversion;
using mlir::triton::gpu::ElementwiseOpConversionBase;
using mlir::triton::gpu::getElementType;
using mlir::triton::gpu::getFunctionType;
using mlir::triton::gpu::MultipleOperandsRange;

namespace {

//===----------------------------------------------------------------------===//
// Basic type conversion utility functions
//===----------------------------------------------------------------------===//

static Value checkIsNan(TritonLLVMOpBuilder &builder, Value v) {
  Location loc = builder.loc;
  OpBuilder &rewriter = *builder.builder;
  // bits 0 and 1 indicate signaling Nan and quiet Nan, respectively
  IntegerAttr controlBits = rewriter.getIntegerAttr(i32_ty, 0b11);
  return LLVM::IsFPClass::create(rewriter, loc, i1_ty, v, controlBits);
}

static Value convertFp32ToBf16(Location loc,
                               ConversionPatternRewriter &rewriter,
                               const Value &v, const RoundingMode rounding) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int32 = b.bitcast(v, i32_ty);
  if (rounding == RoundingMode::RTZ) {
    auto shifted = b.lshr(i32_ty, as_int32, b.i32_val(16));
    auto truncated = b.trunc(i16_ty, shifted);
    return b.bitcast(truncated, bf16_ty);
  }

  // RTNE rounding - faster version from CK
  Value isNan = checkIsNan(b, v);
  Value v16 = b.i32_val(16);
  Value tmp = b.and_(i32_ty, b.lshr(i32_ty, as_int32, v16), b.i32_val(1));

  Value v7FFF = b.i32_val(0x7FFF);
  Value s1 = b.add(as_int32, tmp);
  Value s2 = b.add(s1, v7FFF);

  Value vNan = b.i32_val(0x7FFF0000);
  Value res = b.select(isNan, vNan, s2);

  Value shifted = b.lshr(i32_ty, res, v16);
  Value truncated = b.trunc(i16_ty, shifted);
  return b.bitcast(truncated, bf16_ty);
}

static Value convertBf16ToFp32(Location loc,
                               ConversionPatternRewriter &rewriter,
                               const Value &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto as_int16 = b.bitcast(v, i16_ty);
  auto as_int32 = b.zext(i32_ty, as_int16);
  auto shifted = b.shl(i32_ty, as_int32, b.i32_val(16));
  return b.bitcast(shifted, f32_ty);
}

//===----------------------------------------------------------------------===//
// BF16 elementwise helper
//===----------------------------------------------------------------------===//

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto v0 = convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 = convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = OP::create(rewriter, loc, f32_ty, v0, v1);
  return convertFp32ToBf16(loc, rewriter, result, RoundingMode::RTNE);
}

struct FAddOpConversion
    : ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
    }
    return {LLVM::FAddOp::create(rewriter, loc, elemTy, operands[0][0],
                                 operands[0][1])};
  }
};
} // namespace

namespace mlir::triton::NANO {

void adjustModeRegister(ModuleOp mod, const TargetInfo &targetInfo) {
  // Not needed for minimal backend
}

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit) {

  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  // Register base Triton elementwise patterns (handles arith ops like AddFOp)
  triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}

} // namespace mlir::triton::NANO
