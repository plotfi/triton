// Simplified ElementwiseOpToLLVM.cpp - F8/MX format conversion code removed
#include "TargetInfo.h"
#include "TritonNANOGPUToLLVM/TargetUtils.h"
#include "Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
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
bool isCDNA4(NANO::ISAFamily family) { return family == NANO::ISAFamily::CDNA4; }
bool isCDNA4OrHigher(NANO::ISAFamily family) {
  return family == NANO::ISAFamily::CDNA4 || family == NANO::ISAFamily::GFX1250;
}

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

// Fp16 -> Fp32
static Value cvtFp16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                           const Value &v) {
  TritonLLVMOpBuilder b(loc, rewriter);
  return b.fpext(f32_ty, v);
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

static SmallVector<Value>
convertFp32ToFp16RTZ(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  assert(v.size() == 2);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type v2f16Ty = vec_ty(f16_ty, 2);
  Value result = ROCDL::CvtPkRtz::create(rewriter, loc, v2f16Ty, v[0], v[1]);
  SmallVector<Value> ret(2);
  ret[0] = b.extract_element(f16_ty, result, b.i32_val(0));
  ret[1] = b.extract_element(f16_ty, result, b.i32_val(1));
  return ret;
}

// Fp32->Fp16/Bf16 (RTNE)
static SmallVector<Value>
convertFp32ToFp16RTNE(Location loc, ConversionPatternRewriter &rewriter,
                      ArrayRef<Value> v, Type outElemTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  if (v.size() == 1)
    return {b.fptrunc(outElemTy, v.front())};

  assert(v.size() == 2);
  auto inVecTy = vec_ty(f32_ty, 2);
  auto retVecTy = vec_ty(outElemTy, 2);
  Value inVec = b.undef(inVecTy);
  auto idx0 = b.i32_val(0);
  auto idx1 = b.i32_val(1);
  inVec = b.insert_element(inVecTy, inVec, v[0], idx0);
  inVec = b.insert_element(inVecTy, inVec, v[1], idx1);
  Value retVec = b.fptrunc(retVecTy, inVec);
  SmallVector<Value> ret(2);
  ret[0] = b.extract_element(outElemTy, retVec, idx0);
  ret[1] = b.extract_element(outElemTy, retVec, idx1);
  return ret;
}

// Fp32_to_F16/Bf16 RTNE
static SmallVector<Value> Fp32_to_F16_RTNE(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           Type inElemTy, Type outElemTy,
                                           MultipleOperandsRange operands,
                                           NANO::ISAFamily isaFamily) {
  if (isCDNA4(isaFamily)) {
    SmallVector<Value> inVals;
    size_t numElem = std::min(size_t(2), operands.size());
    inVals.reserve(numElem);
    for (unsigned i = 0; i < numElem; i++) {
      inVals.push_back(operands[i][0]);
    }
    return convertFp32ToFp16RTNE(loc, rewriter, inVals, outElemTy);
  }

  if (outElemTy.isBF16()) {
    assert(inElemTy.isF32() && "unsupported conversion");
    return {
        convertFp32ToBf16(loc, rewriter, operands[0][0], RoundingMode::RTNE)};
  }
  return {LLVM::FPTruncOp::create(rewriter, loc, outElemTy, operands[0][0])};
}

//===----------------------------------------------------------------------===//
// FpToFpOpConversion - Simplified (F8/MX conversions removed)
//===----------------------------------------------------------------------===//

struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<triton::FpToFpOp, FpToFpOpConversion> {
  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              NANO::ISAFamily isaFamily,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return cvtFp16ToFp32(loc, rewriter, v);
  }

  SmallVector<Value> createDestOps(triton::FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    // F32 -> F16/BF16 with RTNE
    if (srcElementType.isF32() &&
        (dstElementType.isF16() || dstElementType.isBF16())) {
      if (roundingMode.has_value() && roundingMode.value() == RoundingMode::RTNE) {
        return Fp32_to_F16_RTNE(loc, rewriter, srcElementType, dstElementType,
                                operands, isaFamily);
      }
    }

    // F32 -> BF16 with RTZ
    if (srcElementType.isF32() && dstElementType.isBF16()) {
      return {convertFp32ToBf16(loc, rewriter, operands[0][0], RoundingMode::RTZ)};
    }

    // F32 -> F16 with RTZ
    if (srcElementType.isF32() && dstElementType.isF16() &&
        roundingMode.has_value() && roundingMode.value() == RoundingMode::RTZ) {
      if (operands.size() >= 2) {
        SmallVector<Value> inVals = {operands[0][0], operands[1][0]};
        auto outVals = convertFp32ToFp16RTZ(loc, rewriter, inVals);
        return {outVals[0]};
      }
      return {LLVM::FPTruncOp::create(rewriter, loc, dstElementType, operands[0][0])};
    }

    // BF16 -> F32
    if (srcElementType.isBF16() && dstElementType.isF32()) {
      return {convertBf16ToFp32(loc, rewriter, operands[0][0])};
    }

    // F16 -> F32
    if (srcElementType.isF16() && dstElementType.isF32()) {
      return {convertFp16ToFp32(loc, rewriter, operands[0][0])};
    }

    // Default: use LLVM conversion ops
    if (srcElementType.getIntOrFloatBitWidth() < dstElementType.getIntOrFloatBitWidth()) {
      return {LLVM::FPExtOp::create(rewriter, loc, dstElementType, operands[0][0])};
    } else {
      return {LLVM::FPTruncOp::create(rewriter, loc, dstElementType, operands[0][0])};
    }
  }

private:
  NANO::ISAFamily isaFamily;
};

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

//===----------------------------------------------------------------------===//
// Arithmetic op conversions
//===----------------------------------------------------------------------===//

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    return {LLVM::FDivOp::create(rewriter, loc, elemTy, operands[0][0],
                                 operands[0][1])};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion> {

  explicit FMulOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisAnalysisPass,
                            NANO::ISAFamily isaFamily,
                            PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  SmallVector<Value> createDestOps(arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      if (isRDNA(isaFamily)) {
        auto b = TritonLLVMOpBuilder(loc, rewriter);
        Value aVal = packLLVector(
            loc, ValueRange{operands[0][0], b.bf16_val(0.0)}, rewriter);
        Value bVal = packLLVector(
            loc, ValueRange{operands[0][1], b.bf16_val(0.0)}, rewriter);
        return {LLVM::createLLVMIntrinsicCallOp(
                    rewriter, loc, "llvm.amdgcn.fdot2.bf16.bf16", bf16_ty,
                    ValueRange{aVal, bVal, b.bf16_val(0.0)})
                    ->getResult(0)};
      } else {
        return {EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
      }
    }
    return {LLVM::FMulOp::create(rewriter, loc, elemTy, operands[0][0],
                                 operands[0][1])};
  }

private:
  NANO::ISAFamily isaFamily;
};

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

struct FSubOpConversion
    : ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    if (lhsElemTy.isBF16() && rhsElemTy.isBF16()) {
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
    }
    return {LLVM::FSubOp::create(rewriter, loc, elemTy, operands[0][0],
                                 operands[0][1])};
  }
};

//===----------------------------------------------------------------------===//
// Integer/Float conversion ops
//===----------------------------------------------------------------------===//

struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto value = LLVM::SIToFPOp::create(rewriter, loc, f32_ty, operands[0][0]);
      return {convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
    }
    return {LLVM::SIToFPOp::create(rewriter, loc, elemTy, operands[0][0])};
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value = convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, value)};
    }
    return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, operands[0][0])};
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {convertBf16ToFp32(loc, rewriter, operands[0][0])};
    }
    return {LLVM::FPExtOp::create(rewriter, loc, elemTy, operands[0][0])};
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  explicit TruncFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              NANO::ISAFamily isaFamily,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        isaFamily(isaFamily) {}

  SmallVector<Value> createDestOps(arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isF32() && (outElemTy.isBF16() || outElemTy.isF16())) {
      return Fp32_to_F16_RTNE(loc, rewriter, inElemTy, outElemTy, operands,
                              isaFamily);
    }
    return {LLVM::FPTruncOp::create(rewriter, loc, elemTy, operands[0][0])};
  }

private:
  NANO::ISAFamily isaFamily;
};

//===----------------------------------------------------------------------===//
// Math op conversions
//===----------------------------------------------------------------------===//

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using ElementwiseOpConversionBase::ElementwiseOpConversionBase;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));
    return {LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.exp2.f32",
                                            f32_ty, prod)
                ->getResult(0)};
  }
};

struct Exp2OpConversion
    : ElementwiseOpConversionBase<math::Exp2Op, Exp2OpConversion> {
  explicit Exp2OpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::Exp2Op op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};
    StringRef funcName = ftz ? "llvm.amdgcn.exp2.f32" : "llvm.exp2.f32";
    return {LLVM::createLLVMIntrinsicCallOp(rewriter, loc, funcName, f32_ty,
                                            operands[0])
                ->getResult(0)};
  }

private:
  bool ftz;
};

static inline std::pair<Value, Value>
scaleUpIfDenorm(ConversionPatternRewriter &rewriter, Location loc,
                const Value &src, float scaleThreshold, float scaleFactor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value needScale = b.fcmp_ogt(b.f32_val(scaleThreshold), src);
  Value scaledSrc = b.fmul(f32_ty, src, b.f32_val(scaleFactor));
  Value selectedSrc = b.select(needScale, scaledSrc, src);
  return {needScale, selectedSrc};
}

static inline Value scaleDownIfDenorm(ConversionPatternRewriter &rewriter,
                                      Location loc, const Value &src,
                                      Value needScale, float scaleFactor) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value scaledSrc = b.fmul(f32_ty, src, b.f32_val(scaleFactor));
  return b.select(needScale, scaledSrc, src);
}

struct RsqrtOpConversion
    : ElementwiseOpConversionBase<math::RsqrtOp, RsqrtOpConversion> {
  explicit RsqrtOpConversion(LLVMTypeConverter &typeConverter,
                             ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                             PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::RsqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value needScale = b.false_val();
    Value scaledSrc = operands[0][0];
    if (!ftz) {
      std::tie(needScale, scaledSrc) = scaleUpIfDenorm(
          rewriter, loc, operands[0][0], 0x1.0p-96f, 0x1.0p+32f);
    }

    Value intrinsicsOutput =
        ROCDL::ROCDLRsq::create(rewriter, loc, elemTy, scaledSrc);

    if (!ftz) {
      return {scaleDownIfDenorm(rewriter, loc, intrinsicsOutput, needScale,
                                0x1.0p+16f)};
    }
    return {intrinsicsOutput};
  }

private:
  bool ftz;
};

struct SqrtOpConversion
    : ElementwiseOpConversionBase<math::SqrtOp, SqrtOpConversion> {
  explicit SqrtOpConversion(LLVMTypeConverter &typeConverter,
                            ModuleAxisInfoAnalysis &axisInfoAnalysis, bool ftz,
                            PatternBenefit benefit)
      : ElementwiseOpConversionBase(typeConverter, axisInfoAnalysis, benefit),
        ftz(ftz) {}

  SmallVector<Value> createDestOps(math::SqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    Value needScale = b.false_val();
    Value scaledSrc = operands[0][0];
    if (!ftz) {
      std::tie(needScale, scaledSrc) = scaleUpIfDenorm(
          rewriter, loc, operands[0][0], 0x1.0p-96f, 0x1.0p+32f);
    }

    Value intrinsicsOutput =
        ROCDL::ROCDLSqrt::create(rewriter, loc, elemTy, operands[0]);

    if (!ftz) {
      return {scaleDownIfDenorm(rewriter, loc, intrinsicsOutput, needScale,
                                0x1.0p-16f)};
    }
    return {intrinsicsOutput};
  }

private:
  bool ftz;
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<triton::ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<triton::ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(triton::ClampFOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (!(elemTy.isF16() || elemTy.isF32()))
      return {};

    Value x = operands[0][0];
    Value lo = operands[0][1];
    Value hi = operands[0][2];

    Value med = ROCDL::FMed3Op::create(rewriter, loc, elemTy, x, lo, hi);

    if (op.getPropagateNan() == PropagateNan::ALL) {
      Value isNan =
          LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::une, x, x);
      Value res = LLVM::SelectOp::create(rewriter, loc, isNan, x, med);
      return {res};
    }

    return {med};
  }
};

} // namespace

namespace mlir::triton::NANO {

void adjustModeRegister(ModuleOp mod, const TargetInfo &targetInfo) {
  // F8 clamping mode register adjustment removed - not needed for minimal backend
}

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit) {

  // fmin (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MinimumFOp, LLVM::MinimumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  // fmax (return NaN if either op is NaN)
  patterns.add<ElementwiseOpConversion<arith::MaximumFOp, LLVM::MaximumOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<triton::PreciseDivFOp, LLVM::FDivOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<triton::PreciseSqrtOp, LLVM::SqrtOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis,
                                 targetInfo.getISAFamily(), benefit);

  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis,
                                   targetInfo.getISAFamily(), benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   targetInfo.getISAFamily(), benefit);

  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<Exp2OpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  patterns.add<RsqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  patterns.add<SqrtOpConversion>(typeConverter, axisInfoAnalysis, ftz, benefit);
  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis,
                                   benefit.getBenefit() + 1);

  triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  bool hwNanPropagationSupported = targetInfo.supportMaximumMinimum();
  triton::populateMinMaxFOpToLLVMPattern(typeConverter, patterns,
                                         axisInfoAnalysis,
                                         hwNanPropagationSupported, benefit);
  triton::populateClampFOpToLLVMPattern(typeConverter, patterns,
                                        axisInfoAnalysis, targetInfo, benefit);
}

} // namespace mlir::triton::NANO
