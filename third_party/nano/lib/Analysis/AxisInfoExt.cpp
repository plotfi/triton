#include "third_party/nano/include/Analysis/AxisInfoExt.h"
#include "third_party/nano/include/Dialect/TritonNANOGPU/IR/Dialect.h"

namespace mlir::triton::NANO {

namespace {
template <typename OpTy> class CastOpAxisInfoVisitor : public AxisInfoVisitor {
public:
  using AxisInfoVisitor::AxisInfoVisitor;

  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) final {
    return operands[0]->getValue();
  }

  virtual bool match(Operation *op) final { return isa<OpTy>(op); }
};
} // namespace

void AxisInfoExt::addVisitors(mlir::triton::AxisInfoVisitorList &visitors) {
  visitors.append<CastOpAxisInfoVisitor<nanogpu::ExtractSliceOp>>();
  return;
}
} // namespace mlir::triton::NANO
