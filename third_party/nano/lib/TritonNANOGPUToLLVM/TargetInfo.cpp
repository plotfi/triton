// Minimal TargetInfo.cpp - Only what's needed for vector add
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::NANO {

llvm::AMDGPU::IsaVersion TargetInfo::getIsaVersion() const {
  return llvm::AMDGPU::getIsaVersion(arch);
}

llvm::AMDGPU::GPUKind TargetInfo::getGPUKind() const {
  return llvm::AMDGPU::parseArchAMDGCN(arch);
}

int TargetInfo::getWarpSize() const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return 64;
  case ISAFamily::GFX1250:
    return 32;
  default:
    break;
  }
  return 32;
}

int TargetInfo::getSharedMemorySize() const {
  switch (getISAFamily()) {
  case ISAFamily::GFX1250:
    return 320 * 1024;
  case ISAFamily::CDNA4:
    return 160 * 1024;
  default:
    return 64 * 1024;
  }
}

bool TargetInfo::supportMaximumMinimum() const {
  return getISAFamily() == ISAFamily::CDNA4;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  return ROCDL::BallotOp::create(rewriter, loc, type, cmp);
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier(targets);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  LLVM::createLLVMIntrinsicCallOp(rewriter, loc, "llvm.amdgcn.wave.barrier", {},
                                  {});
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  mlir::LLVM::NANO::llStore(rewriter, loc, ptr, val, pred);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  Value falseVal = LLVM::ConstantOp::create(rewriter, loc, elemTy,
                                            rewriter.getZeroAttr(elemTy));
  return mlir::LLVM::NANO::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal,
                                  {}, triton::CacheModifier::NONE, false);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("shuffleXor not supported in minimal NANO backend");
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  llvm_unreachable("shuffleUp not supported in minimal NANO backend");
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  llvm_unreachable("shuffleIdx not supported in minimal NANO backend");
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  llvm_unreachable("shuffleIdx not supported in minimal NANO backend");
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  llvm_unreachable("permute not supported in minimal NANO backend");
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::NANO::llGetPid(loc, rewriter, moduleOp, axis);
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  // Warp reduction not supported in minimal backend
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  return resultElementTy.isInteger(32) ? "llvm.amdgcn.mul.hi.u32"
                                       : "llvm.amdgcn.mul.hi.u64";
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  // Printf not supported in minimal backend
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  // Printf not supported in minimal backend
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  // Just trap without printing
  LLVM::Trap::create(rewriter, loc);
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    return 3;
  }
  llvm::report_fatal_error("Only support SharedMemorySpace for now");
}

bool TargetInfo::supportVectorizedAtomics() const { return true; }

bool TargetInfo::supportsDirectToLDSScattering() const {
  return getISAFamily() == ISAFamily::GFX1250;
}

bool TargetInfo::requiresAliasInfoForAsyncOps() const {
  return getISAFamily() == ISAFamily::CDNA3 ||
         getISAFamily() == ISAFamily::CDNA4;
}

bool TargetInfo::supportsDirectToLdsLoadBitWidth(int bitWidth) const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA3:
    return bitWidth == 32;
  case ISAFamily::CDNA4:
    return bitWidth == 128 || bitWidth == 32;
  case ISAFamily::GFX1250:
    return bitWidth == 128 || bitWidth == 64 || bitWidth == 32;
  default:
    return false;
  }
}

bool TargetInfo::supportsMultiCTALaunch() const {
  return getISAFamily() == ISAFamily::GFX1250;
}

bool TargetInfo::supportsTDM() const { return false; }

bool TargetInfo::supportsClusterLoadBitWidth(int bitWidth) const {
  if (getISAFamily() == ISAFamily::GFX1250) {
    return bitWidth == 32 || bitWidth == 64 || bitWidth == 128;
  }
  return false;
}

bool TargetInfo::supportsDirectFromLdsStoreBitWidth(int bitWidth) const {
  if (getISAFamily() == ISAFamily::GFX1250) {
    return bitWidth == 128 || bitWidth == 64 || bitWidth == 32 || bitWidth == 8;
  }
  return false;
}

void TargetInfo::localLoadOpAnnotation(triton::gpu::LocalLoadOp localLoadOp,
                                       Operation *llLoadOp) const {
  // Not needed for minimal backend
}

} // namespace mlir::triton::NANO
