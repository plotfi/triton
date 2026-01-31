#include "third_party/nano/include/Analysis/AxisInfoExt.h"

namespace mlir::triton::NANO {

void AxisInfoExt::addVisitors(mlir::triton::AxisInfoVisitorList &visitors) {
  // TritonNANOGPU dialect operations removed - no custom visitors needed
  return;
}
} // namespace mlir::triton::NANO
