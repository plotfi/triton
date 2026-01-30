#ifndef TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUDIALECTTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUDIALECTTOLLVM_UTILITY_H_

#include "triton/Tools/LinearLayout.h"

namespace tt = mlir::triton;

namespace mlir::LLVM::NANO {
using ElemLocationKey = SmallVector<std::pair<StringAttr, int32_t>>;

ElemLocationKey getElemCoordinatesFromRegisters(tt::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx);

std::optional<int> getRegFromCoordinates(tt::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx);

} // namespace mlir::LLVM::NANO
#endif // TRITON_THIRD_PARTY_NANO_LIB_TRITONNANOGPUDIALECTTOLLVM_UTILITY_H_
