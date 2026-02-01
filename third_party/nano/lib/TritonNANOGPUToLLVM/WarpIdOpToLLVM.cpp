// Minimal WarpIdOpToLLVM - simplified for vector add
#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

class WarpIdOpPattern : public ConvertOpToLLVMPattern<WarpIdOp> {
public:
  WarpIdOpPattern(LLVMTypeConverter &typeConverter,
                  const NANO::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<WarpIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Simple warp ID: thread ID / warp size
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Value tid = getThreadId(rewriter, loc);
    Value warpId = b.udiv(tid, b.i32_val(threadsPerWarp));

    rewriter.replaceOp(op, warpId);
    return success();
  }

private:
  const NANO::TargetInfo &targetInfo;
};
} // namespace

void mlir::triton::NANO::populateWarpIdOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<WarpIdOpPattern>(typeConverter, targetInfo, benefit);
}
