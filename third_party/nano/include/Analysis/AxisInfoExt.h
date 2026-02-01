#ifndef TRITONNANO_ANALYSIS_AXIS_INFO_EXT_H
#define TRITONNANO_ANALYSIS_AXIS_INFO_EXT_H

#include "include/triton/Analysis/AxisInfo.h"

namespace mlir::triton::NANO {

struct AxisInfoExt {
  static void addVisitors(mlir::triton::AxisInfoVisitorList &visitors);
};

class ModuleAxisInfoAnalysis : public mlir::triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : mlir::triton::ModuleAxisInfoAnalysis(moduleOp,
                                             AxisInfoExt::addVisitors) {}
};
} // namespace mlir::triton::NANO

#endif // TRITONNANO_ANALYSIS_AXIS_INFO_EXT_H
