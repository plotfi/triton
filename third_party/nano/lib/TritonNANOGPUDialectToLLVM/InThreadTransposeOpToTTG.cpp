#include "Dialect/TritonNANOGPU/IR/Dialect.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

struct InThreadTransposeOpConversion
    : public OpConversionPattern<triton::nanogpu::InThreadTransposeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::nanogpu::InThreadTransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, op.getType(),
                                                      op.getSrc());
    return success();
  }
};

} // namespace

namespace mlir::triton::NANO {

void populateInThreadTransposeOpToTTGPatterns(RewritePatternSet &patterns,
                                              PatternBenefit benefit) {
  patterns.add<InThreadTransposeOpConversion>(patterns.getContext(), benefit);
}

} // namespace mlir::triton::NANO
