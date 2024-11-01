/// __FACEBOOK__ (facebook) begin T203329359
class TritonLocalCopyOpPattern
    : public OpConversionPattern<triton::LocalCopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LocalCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    triton::gpu::LocalAllocOp newOp =
        rewriter.replaceOpWithNewOp<triton::gpu::LocalAllocOp>(
            op, op.getType(), adaptor.getOperands());
    return success();
  }
};

class TritonGatherOpPattern : public OpConversionPattern<triton::GatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();

    RankedTensorType oldIndexType =
        cast<RankedTensorType>(adaptor.getIndices().getType());
    RankedTensorType newType = RankedTensorType::get(
        oldIndexType.getShape(), op.getType().getElementType(),
        oldIndexType.getEncoding());

    triton::gpu::LocalGatherOp newOp =
        rewriter.replaceOpWithNewOp<triton::gpu::LocalGatherOp>(
            op, newType, adaptor.getSrc(), adaptor.getIndices(),
            adaptor.getMask(), nullptr);
    return success();
  }
};
/// __FACEBOOK__ (facebook) end T203329359
