#ifndef TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_TARGETUTILS_H_
#define TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_TARGETUTILS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::NANO {

// A list of ISA families we care about.
enum class ISAFamily {
  Unknown,
  CDNA1,
  CDNA2,
  CDNA3,
  CDNA4,
  RDNA1,
  RDNA2,
  RDNA3,
  RDNA4,
  GFX1250,
};

// Deduces the corresponding ISA family for the given target gfx |arch|.
ISAFamily deduceISAFamily(llvm::StringRef arch);

} // namespace mlir::triton::NANO

#endif // TRITON_THIRD_PARTY_NANO_INCLUDE_TRITONNANOGPUTOLLVM_TARGETUTILS_H_
