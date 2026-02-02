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
inline ISAFamily deduceISAFamily(llvm::StringRef arch) {
  llvm::AMDGPU::GPUKind kind = llvm::AMDGPU::parseArchAMDGCN(arch);

  if (kind == llvm::AMDGPU::GK_GFX1250)
    return ISAFamily::GFX1250;

  // CDNA ISA cases
  switch (kind) {
  case llvm::AMDGPU::GK_GFX950:
    return ISAFamily::CDNA4;
  case llvm::AMDGPU::GK_GFX942:
    return ISAFamily::CDNA3;
  case llvm::AMDGPU::GK_GFX90A:
    return ISAFamily::CDNA2;
  case llvm::AMDGPU::GK_GFX908:
    return ISAFamily::CDNA1;
  default:
    break;
  }

  // RDNA ISA cases
  if (kind >= llvm::AMDGPU::GK_GFX1200 && kind <= llvm::AMDGPU::GK_GFX1201)
    return ISAFamily::RDNA4;
  if (kind >= llvm::AMDGPU::GK_GFX1100 && kind <= llvm::AMDGPU::GK_GFX1153)
    return ISAFamily::RDNA3;
  if (kind >= llvm::AMDGPU::GK_GFX1030 && kind <= llvm::AMDGPU::GK_GFX1036)
    return ISAFamily::RDNA2;
  if (kind >= llvm::AMDGPU::GK_GFX1010 && kind <= llvm::AMDGPU::GK_GFX1013)
    return ISAFamily::RDNA1;

  return ISAFamily::Unknown;
}
} // namespace mlir::triton::NANO

#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"

#define populateISASpecificConversionPatterns() \
  do { \
    auto maybeChipset = mlir::amdgpu::Chipset::parse(this->arch); \
    if (failed(maybeChipset)) { \
      emitError(UnknownLoc::get(&getContext()), \
                "Invalid chipset name: " + this->arch); \
      return signalPassFailure(); \
    } \
    mlir::populateGpuToROCDLConversionPatterns( \
        typeConverter, patterns, mlir::gpu::amd::HIP, *maybeChipset); \
  } while (false)

