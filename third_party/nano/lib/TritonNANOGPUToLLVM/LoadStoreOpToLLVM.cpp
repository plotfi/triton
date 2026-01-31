#include "AtomicRMWOpsEmitter.h"
// TritonNANOGPU dialect removed - not needed for minimal nano backend
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::NANO::getVectorSize;
using ::mlir::LLVM::NANO::llLoad;
using ::mlir::LLVM::NANO::llStore;
using ::mlir::triton::NANO::ISAFamily;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

std::optional<const char *> getAMDGPUMemScopeStr(MemSyncScope scope) {
  switch (scope) {
  case MemSyncScope::GPU:
    return "agent";
  case MemSyncScope::CTA:
    return "workgroup";
  // The default AMDHSA LLVM Sync Scope is "system", so no string is
  // provided here
  case MemSyncScope::SYSTEM:
  default:
    return "";
  }
}

std::pair<bool, bool> getOrderingFlags(MemSemantic memOrdering) {
  bool emitReleaseFence = false;
  bool emitAcquireFence = false;
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    // In this case, no memory fences are needed
    break;
  case MemSemantic::RELEASE:
    emitReleaseFence = true;
    break;
  case MemSemantic::ACQUIRE:
    emitAcquireFence = true;
    break;
  case MemSemantic::ACQUIRE_RELEASE:
    emitAcquireFence = true;
    emitReleaseFence = true;
  default:
    // default == acq_rel, so we emit the same barriers
    emitAcquireFence = true;
    emitReleaseFence = true;
  }
  return {emitAcquireFence, emitReleaseFence};
}

LogicalResult emitFence(Operation *op, ConversionPatternRewriter &rewriter,
                        Location loc, MemSemantic memOrdering,
                        MemSyncScope memScope, bool preAtomic) {
  // This function emits an LLVM::FenceOp which will get lowered by the
  // LLVM backend to the right scope and ordering instructions, as
  // described in the "atomicrmw" entries for "global" address-space,
  // in the "AMDHSA Memory Model Code Sequences GFX942"
  // table in https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942
  //
  // Triton supports three scopes for atomic access
  // 1. System
  // 2. GPU (default) ('Agent' for AMDGPU)
  // 3. CTA ('Workgroup' for AMDGPU)
  //
  // and 4 orderings
  // 1. Relaxed
  // 2. Acquire
  // 3. Release
  // 4. AcquireRelease
  //
  // The following table shows the scope and ordering instructions that
  // are emitted by this function for each combination of scope and ordering
  // for buffer-atomic instructions.
  //
  // Note: In the following comments, "[buffer-atomic_0.. buffer-atomic_n]"
  // represents a sequence of buffer-atomic instructions that are lowered from
  // a single tl.atomic_*
  //
  // Unordered(Relaxed):
  //   agent/workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //                    No scope/ordering instrs are required.
  //   system: //TODO:
  // Acquire:
  //   workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //              All waves in the workgroup use same L1 and L2.
  //              No scope/ordering instrs are required.
  //   agent: Instr seq: [buffer-atomic_0.. buffer-atomic_n],
  //                     s_waitcnt vmcnt(0), buffer_inv sc1=1
  //          Waves across an agent may use different L1 and L2.
  //          Atomic ops bypass L1 and operate on L2.
  //          s_waitcnt vmcnt(0) ensures that the atomicrmw has completed
  //          before invalidating the cache. buffer_inv sc1=1 will a) L1:
  //          invalidate cache b) L2: Invalidate non-coherently modified lines
  //          if multiple L2s are configured, NOP otherwise. This buffer_inv
  //          ensures that following loads do not see stale global values.
  //   system: //TODO:
  //
  // Release:
  //   workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //              All waves in the workgroup use same L1 and L2 so all
  //              previous global writes of a waver are visible to all other
  //              waves in the workgroup. LDS operations for all waves are
  //              executed in a total global ordering and are observed by all
  //              waves in the workgroup. So LDS stores issued before the
  //              release will be visible to LDS loads after the read of the
  //              released buffer-atomic. So, swait_cnt lgkmcnt is not
  //              required.
  //   agent: Instr seq: buffer_wbl2 sc1=1, s_waitcnt vmcnt(0),
  //                     [buffer-atomic_0.. buffer-atomic_n]
  //          buffer_wbl2 sc1=1 ensures that dirtly L2 lines are visible to
  //          CUs that don't use the same L2.
  //          From SIMemoryLegalizer.cpp SIGfx940CacheControl::insertRelease:
  //            "Inserting a "S_WAITCNT vmcnt(0)" before is not required
  //             because the hardware does not reorder memory operations by
  //             the same wave with respect to a following "BUFFER_WBL2".
  //             The "BUFFER_WBL2" is guaranteed to initiate writeback of
  //             any dirty cache lines of earlier writes by the same wave.
  //             A "S_WAITCNT vmcnt(0)" is needed after to ensure the writeback
  //             has completed.""
  //   system: //TODO:
  //
  // AcquireRelease:
  //   Instr seq: Release scope/order insts,
  //              [buffer-atomic_0..buffer-atomic_n],
  //              Acquire scope/order instrs.
  //
  // LLVM::FenceOp lowering will emit the required cache ops and s_waitcnt
  // vmcnt(0) instrs

  auto [emitReleaseFence, emitAcquireFence] = getOrderingFlags(memOrdering);
  if (MemSyncScope::SYSTEM == memScope)
    return rewriter.notifyMatchFailure(
        op, "System memory scope is not supported for Buffer Atomic Ops");
  auto scopeStr = getAMDGPUMemScopeStr(memScope);
  if (!scopeStr)
    return rewriter.notifyMatchFailure(
        op, "Unsupported memory scope for Buffer Atomic Ops");

  StringAttr scope = mlir::StringAttr::get(loc.getContext(), *scopeStr);

  if (emitReleaseFence && preAtomic) {
    LLVM::FenceOp::create(rewriter, loc, TypeRange{},
                          LLVM::AtomicOrdering::release, scope);
  }

  if (emitAcquireFence && !preAtomic) {
    LLVM::FenceOp::create(rewriter, loc, TypeRange{},
                          LLVM::AtomicOrdering::acquire, scope);
  }
  return success();
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const NANO::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred = b.true_val();
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = b.and_(pred, dimPred);
    }
  }
  return pred;
}

std::pair<Block *, Block *> emitBranch(RewriterBase &rewriter, Location loc,
                                       Value cond) {
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *after =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *body = rewriter.createBlock(after);
  rewriter.setInsertionPointToEnd(currentBlock);
  LLVM::CondBrOp::create(rewriter, loc, cond, body, after);
  rewriter.setInsertionPointToStart(body);
  LLVM::BrOp::create(rewriter, loc, after);
  rewriter.setInsertionPointToStart(body);
  return {body, after};
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const NANO::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  // Create a LLVM vector of type `vecTy` containing all zeros
  Value createZeroVector(OpBuilder &builder, Location loc,
                         VectorType vecTy) const {
    mlir::Attribute zeroAttr = builder.getZeroAttr(vecTy.getElementType());
    auto denseValue =
        DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
    Value zeroVal = LLVM::ConstantOp::create(builder, loc, vecTy, denseValue);
    return zeroVal;
  }

  // Given a vector of values `elems` and a starting point `start`, create a
  // LLVM vector of length `vec` whose elements are `elems[start, ...,
  // elems+vec-1]`
  Value packElementRangeIntoVector(RewriterBase &rewriter,
                                   const LLVMTypeConverter *typeConverter,
                                   Location loc, VectorType vecTy,
                                   ArrayRef<Value> elems, int64_t start) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int64_t vec = vecTy.getNumElements();
    // If we need to mask the loaded value with other elements
    Value v = b.undef(vecTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = elems[start + s];
      Value indexVal =
          LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = b.insert_element(vecTy, v, otherElem, indexVal);
    }
    return v;
  }

  // Return a tensor of pointers with the same type of `basePtr` and the same
  // shape of `offset`
  Type getPointerTypeWithShape(Value basePtr, Value offset) const {
    Type basePtrType = basePtr.getType();
    auto offsetType = cast<RankedTensorType>(offset.getType());
    return offsetType.cloneWith(std::nullopt, basePtrType);
  }

  // Unpack the elements contained in a `llvmStruct` into a `SmallVector` of
  // `Value`s. While you do that, check also the alignment of the mask and
  // update the vector length `vec` accordingly
  SmallVector<Value>
  getMaskElemsAndUpdateVeclen(ConversionPatternRewriter &rewriter, Location loc,
                              Value llMask, Value mask, unsigned &vec) const {
    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      maskElems = unpackLLElements(loc, llMask, rewriter);
    }
    return maskElems;
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const NANO::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

// Contains some helper functions for direct to lds loads.
struct DirectToLdsLoadConversionBase : public LoadStoreConversionBase {
  explicit DirectToLdsLoadConversionBase(
      const NANO::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass)
      : LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  // For each load emit the computation to get the lane id offset which holds
  // the source pointers/offsets we need to store to shared memory
  SmallVector<Value>
  emitSwizzledLaneOffsets(RewriterBase &rewriter, Operation *op,
                          RankedTensorType srcTy, MemDescType swizzledTy,
                          MemDescType flatTy, Value llDst, Type resElemTy,
                          unsigned vec) const {
    auto loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    // Create regToShared layout for the swizzled and flat encoding
    auto regLayout = triton::gpu::toLinearLayout(srcTy);

    auto sharedSwizz = triton::gpu::toLinearLayout(swizzledTy);
    auto sharedFlat = triton::gpu::toLinearLayout(flatTy);

    auto regToSharedSwizzled = regLayout.invertAndCompose(sharedSwizz);
    auto regToSharedFlat = regLayout.invertAndCompose(sharedFlat);

    MLIRContext *ctx = rewriter.getContext();
    StringAttr kBlock = str_attr("block");
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    Value blockId = b.i32_val(0);

    int numberOfLoads = regToSharedSwizzled.getInDimSize(kRegister) / vec;

    // For each load compute the difference between the flat and the swizzled
    // linear offsets into shared memory
    // TODO (alex): this is only correct as long as the lds view is a contiguous
    // block. So this can break if we slice along the 2 minor dimensions
    SmallVector<Value> swizzledOffsets;
    swizzledOffsets.reserve(numberOfLoads);
    auto vecVal = b.i32_val(vec);
    for (int i = 0; i < numberOfLoads; i++) {
      auto regId = b.i32_val(i * vec);

      std::array<std::pair<StringAttr, Value>, 4> indices{{
          {kRegister, regId},
          {kLane, laneId},
          {kWarp, warpId},
          {kBlock, blockId},
      }};

      Value swizzledOffset =
          applyLinearLayout(loc, rewriter, regToSharedSwizzled, indices)[0]
              .second;
      Value flatOffset =
          applyLinearLayout(loc, rewriter, regToSharedFlat, indices)[0].second;

      // Normalize the offset by vecTy to obtain the offset in lanes
      auto laneOffet = b.sdiv(b.sub(swizzledOffset, flatOffset), vecVal);
      swizzledOffsets.push_back(laneOffet);
    }
    return swizzledOffsets;
  }

  // Swizzle the mask (1bit) based on selectLane via ballot
  Value shuffleMask(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                    Location loc, const TargetInfoBase &targetInfo,
                    Value selectLane, Value mask) const {
    auto warpMask =
        targetInfo.ballot(rewriter, loc, rewriter.getI64Type(), mask);
    // Extract the selectLane bit
    auto bitMask = b.lshr(warpMask, b.zext(rewriter.getI64Type(), selectLane));
    return b.trunc(i1_ty, bitMask);
  }

  SmallVector<Value>
  zipAsyncCopyValues(RewriterBase &rewriter, Location loc, unsigned vec,
                     ArrayRef<Value> srcElems, Type srcTy,
                     ArrayRef<Value> maskElems, ArrayRef<Value> otherElems,
                     Type otherTy, ArrayRef<Value> swizzledLaneOffsets) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> loadVals;
    auto structTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), ArrayRef<Type>{srcTy, i1_ty, otherTy, i32_ty});
    for (int i = 0; i < srcElems.size(); i++) {
      Value packedArr = LLVM::UndefOp::create(rewriter, loc, structTy);
      // src
      packedArr = b.insert_val(packedArr, srcElems[i], 0);
      // mask
      auto maskElem = maskElems.empty() ? b.true_val() : maskElems[i];
      packedArr = b.insert_val(packedArr, maskElem, 1);
      // other
      if (!otherElems.empty())
        packedArr = b.insert_val(packedArr, otherElems[i], 2);
      // swizzleOffset are per vec so we need to duplicate values vec times
      auto swizzleOffset = swizzledLaneOffsets.empty()
                               ? b.i32_val(0)
                               : swizzledLaneOffsets[i / vec];
      packedArr = b.insert_val(packedArr, swizzleOffset, 3);

      loadVals.push_back(packedArr);
    }
    return loadVals;
  }

  auto unzipAsyncCopyValues(RewriterBase &rewriter, Location loc, int startIdx,
                            ArrayRef<Value> values, Type srcTy, Type otherTy,
                            bool hasOther, unsigned vec) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto structElem = values[startIdx];
    Value offsetElem = b.extract_val(srcTy, structElem, 0);
    Value maskElem = b.extract_val(i1_ty, structElem, 1);
    // Gather other elements
    SmallVector<Value> otherElems;
    if (hasOther) {
      for (int i = 0; i < vec; i++) {
        otherElems.push_back(b.extract_val(otherTy, values[startIdx + i], 2));
      }
    }

    Value swizzleLaneOffset = b.extract_val(i32_ty, structElem, 3);

    return std::make_tuple(offsetElem, maskElem, std::move(otherElems),
                           swizzleLaneOffset);
  }

  void applySwizzling(RewriterBase &rewriter, Location loc, Value &srcOrOffset,
                      Value &mask, Value laneId,
                      Value swizzleLaneOffset) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    // laneId + swizzleOffset will always stay inside the warp [0,
    // threadsPerWarp) because we only swizzle inside a warp
    Value swizzledLaneId = b.add(laneId, swizzleLaneOffset);
    // Shuffle based on swizzleLaneId to apply the swizzling
    srcOrOffset =
        targetInfo.shuffleIdx(rewriter, loc, srcOrOffset, swizzledLaneId);

    if (mask) {
      mask = shuffleMask(rewriter, b, loc, targetInfo, swizzledLaneId, mask);
    }
  }

  // Unified helper for async copy between global and shared memory.
  // Works for both load (global→shared) and store (shared→global).
  // Parameters:
  //   globalTy: The global memory tensor type (src for load, dst for store)
  //   sharedTy: The shared memory descriptor type (dst for load, src for store)
  //   vals: Values to process (packed pointers/masks)
  //   llShared: LLVM value for shared memory struct
  //   isLoad: true for global→shared, false for shared→global
  //   isaFamily: ISA family (only used for load multicast)
  //   lowerInst: Callback to emit the actual load/store instruction
  LogicalResult lowerDirectLDSAsyncCopy(
      RewriterBase &rewriter, Location loc, RankedTensorType globalTy,
      MemDescType sharedTy, SmallVector<Value> vals, Value llShared,
      Type resElemTy, unsigned vec, bool isLoad,
      std::function<SmallVector<Value>(RewriterBase &, Location,
                                       ArrayRef<Value>, Value, int, VectorType,
                                       Value)>
          lowerInst) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto *ctx = rewriter.getContext();

    // Build global to shared layout and remove broadcasted registers
    auto globalLayout = triton::gpu::toLinearLayout(globalTy);
    auto removeBroadcast = actionRemoveBroadcastedRegs(globalLayout);
    globalLayout = removeBroadcast.apply(globalLayout);
    vals = removeBroadcast.apply(vals);

    LinearLayout sharedLayout;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            sharedTy.getEncoding())) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(sharedTy);
    }
    auto cvt = globalLayout.invertAndCompose(sharedLayout);
    if (!cvt.isTrivialOver({str_attr("block")})) {
      return emitError(loc, isLoad ? "direct to lds loads do not support "
                                     "non-trivial block dimension"
                                   : "direct from lds stores do not support "
                                     "non-trivial block dimension");
    }
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});

    // Multicast is only supported for loads
    Value ctaMulticastMask;
    if (isLoad && targetInfo.supportsMultiCTALaunch()) {
      ctaMulticastMask = LLVM::NANO::emitCtaMulticastMask(
          rewriter, loc, targetInfo.getClusterCTAId(rewriter, loc),
          globalLayout);
    }

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, llShared,
                                                         resElemTy, rewriter);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, sharedTy);
    auto maskSpanAffineOffset =
        SharedMemoryObject::getMaskSpanOffsets(sharedTy);

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);

    auto paddingShifts = getPaddedSharedShifts(
        sharedTy.getEncoding(), sharedTy.getElementTypeBitWidth(),
        /*offsetInBytes=*/true);

    auto lowerInstForwardMulticastMask =
        [&](RewriterBase &rewriter, Location loc, ArrayRef<Value> vals,
            Value shmemAddr, int idx, VectorType vecTy) {
          return lowerInst(rewriter, loc, vals, shmemAddr, idx, vecTy,
                           ctaMulticastMask);
        };

    // For loads on GFX9 (no scattering support), the address should be the
    // start address (scalar) of the warp
    if (isLoad && !targetInfo.supportsDirectToLDSScattering()) {
      laneId = b.i32_val(0);
    }

    lowerLdSt(loc, ctx, cvt, vals, resElemTy, smemObj.getBase(), paddingShifts,
              affineOffset, maskSpanAffineOffset, laneId, warpId, rewriter,
              targetInfo, vec, lowerInstForwardMulticastMask);
    return success();
  }

  void emitOtherStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter, VectorType vecTy,
                      Value mask, ArrayRef<Value> otherElems, Value shmemAddr,
                      Value laneId, bool requiresSrcPtrSwizzling,
                      Value swizzleLaneOffset) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    Value storeVal = packElementRangeIntoVector(rewriter, typeConverter, loc,
                                                vecTy, otherElems, 0);
    Type ptrTy = shmemAddr.getType();
    Value ldsAddr = shmemAddr;
    // When scattering is unsupported, shmemAddr is the warp base address.
    // Use shmemAddr + lane_id [+ swizzleOffset] to compute each lane's address.
    if (!targetInfo.supportsDirectToLDSScattering()) {
      ldsAddr = b.gep(ptrTy, vecTy, shmemAddr, laneId);
      if (requiresSrcPtrSwizzling)
        ldsAddr = b.gep(ptrTy, vecTy, ldsAddr, swizzleLaneOffset);
    }
    llStore(rewriter, loc, ldsAddr, storeVal, b.icmp_ne(mask, b.true_val()),
            CacheModifier::NONE, targetInfo.requiresAliasInfoForAsyncOps());
  }
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const NANO::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    SmallVector<Value> otherElems;
    if (other)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    Value multicastMask;
    if (targetInfo.supportsMultiCTALaunch()) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType())) {
        Value clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
        auto regLayout = triton::gpu::toLinearLayout(tensorTy);
        multicastMask = LLVM::NANO::emitCtaMulticastMask(
            rewriter, loc, clusterCTAId, regLayout);
      }
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    const int numVecs = numElems / vec;

    auto cacheMod = op.getCache();
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value ptr = ptrElems[vecStart];

      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);

      Value loadVal = llLoad(rewriter, loc, ptr, vecTy, pred, falseVal,
                             multicastMask, cacheMod);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};



struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public DirectToLdsLoadConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const NANO::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        DirectToLdsLoadConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto srcTy = op.getSrc().getType();

    auto dstTy = op.getResult().getType();
    auto dstEnc = dstTy.getEncoding();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value llDst = adaptor.getResult();

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned vec = getVectorSize(op.getSrc(), axisAnalysisPass);
    auto maskElements = getMaskElemsAndUpdateVeclen(
        rewriter, loc, adaptor.getMask(), op.getMask(), vec);

    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> otherElems;
    if (op.getOther())
      otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);

    // If the op has a contiguity hint use it to increase the vector size.
    vec = std::max(vec, op.getContiguity());

    if (!LLVM::NANO::canLoadDirectToLDS(targetInfo, srcTy, dstEnc,
                                       dstTy.getAllocShape(), vec)) {
      return failure();
    }

    // For swizzled layouts we need to use the non swizzled layout to compute
    // the LDS addresses since we gather into LDS
    auto flatDstTy = dstTy;
    SmallVector<Value> swizzledLaneOffsets;
    auto maybeSwizzledEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstEnc);
    bool requiresSrcPtrSwizzling =
        !targetInfo.supportsDirectToLDSScattering() && maybeSwizzledEnc &&
        maybeSwizzledEnc.getMaxPhase() != 1;
    if (requiresSrcPtrSwizzling) {
      auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), maybeSwizzledEnc.getVec(), 1, 1,
          maybeSwizzledEnc.getOrder(), maybeSwizzledEnc.getCGALayout());
      flatDstTy = MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                                   flatSharedEnc, dstTy.getMemorySpace());
      swizzledLaneOffsets = emitSwizzledLaneOffsets(
          rewriter, op, srcTy, dstTy, flatDstTy, llDst, resElemTy, vec);
    }

    Type srcPtrTy = srcElems[0].getType();
    bool hasOther = !otherElems.empty();
    Type otherTy = hasOther ? otherElems[0].getType() : i1_ty;
    // Zip buffer_offset, mask, other, swizzleOffsets for lowerLdSt
    SmallVector<Value> loadVals =
        zipAsyncCopyValues(rewriter, loc, vec, srcElems, srcPtrTy, maskElements,
                           otherElems, otherTy, swizzledLaneOffsets);

    auto freeVarMasks = getFreeVariableMasks(srcTy);
    // We load redundant data on different CTAs so each CTA has a copy in its
    // shared memory; the multicast mask will be used by the hardware to
    // efficiently broadcast to different CTAs.
    freeVarMasks[rewriter.getStringAttr("block")] = 0;
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto emitGlobalLoadLds =
        [this, &op, &b, laneId = laneId, threadPred, srcPtrTy, otherTy,
         hasOther, requiresSrcPtrSwizzling](
            RewriterBase &rewriter, Location loc, ArrayRef<Value> loadValues,
            Value shmemAddr, int startIdx, VectorType vecTy,
            Value multicastMask) -> SmallVector<Value> {
      auto [srcElem, maskElem, otherElems, swizzleLaneOffset] =
          unzipAsyncCopyValues(rewriter, loc, startIdx, loadValues, srcPtrTy,
                               otherTy, hasOther, vecTy.getNumElements());
      int vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      assert(targetInfo.supportsDirectToLdsLoadBitWidth(vecBits));
      Value maybeSwizzledMaskElem = maskElem;

      if (requiresSrcPtrSwizzling)
        applySwizzling(rewriter, loc, srcElem, maybeSwizzledMaskElem, laneId,
                       swizzleLaneOffset);

      // Predicate load based on threadPred && swizzledMask
      auto cond = b.and_(threadPred, maybeSwizzledMaskElem);
      auto [loadBlock, afterLoadBlock] = emitBranch(rewriter, loc, cond);

      emitAsyncLoad(rewriter, loc, targetInfo, vecBits, srcElem, shmemAddr,
                    op.getCache(), multicastMask);

      rewriter.setInsertionPointToStart(afterLoadBlock);

      if (hasOther) {
        emitOtherStore(rewriter, loc, this->getTypeConverter(), vecTy, maskElem,
                       otherElems, shmemAddr, laneId, requiresSrcPtrSwizzling,
                       swizzleLaneOffset);
      }

      return {};
    };

    auto res = lowerDirectLDSAsyncCopy(rewriter, loc, srcTy, flatDstTy,
                                       loadVals, llDst, resElemTy, vec,
                                       /*isLoad=*/true, emitGlobalLoadLds);
    if (failed(res)) {
      return failure();
    }

    // Drop the result token.
    Value zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                          IntegerType::get(op.getContext(), 32),
                                          rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }

  void emitAsyncLoad(RewriterBase &rewriter, Location loc,
                     NANO::TargetInfo targetInfo, int vecBits, Value srcPtr,
                     Value shmemAddr, triton::CacheModifier cacheMod,
                     Value multicastMask) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int32_t cacheModifiers =
        mlir::LLVM::NANO::getCtrlBitsForCacheModifierOnTarget(
            cacheMod, /*isLoad=*/true, targetInfo);

    if (llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                           targetInfo.getISAFamily())) {
      auto globalLoadLdsOp = ROCDL::GlobalLoadLDSOp::create(
          rewriter, loc, srcPtr, shmemAddr, vecBits / 8,
          /*offset=*/0, cacheModifiers, nullptr, nullptr, nullptr);
      (void)globalLoadLdsOp;
    } else if (targetInfo.getISAFamily() == ISAFamily::GFX1250) {
      if (cacheMod != triton::CacheModifier::NONE) {
        emitRemark(loc) << "cache modifiers not yet implemented on gfx1250";
      }
      if (multicastMask) {
        std::string intrinsic =
            "llvm.amdgcn.cluster.load.async.to.lds.b" + std::to_string(vecBits);
        auto globalLoadLdsOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, {},
            {srcPtr, shmemAddr, b.i32_val(0), b.i32_val(cacheModifiers),
             multicastMask});
      } else {
        std::string intrinsic =
            "llvm.amdgcn.global.load.async.to.lds.b" + std::to_string(vecBits);
        auto globalLoadLdsOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, {},
            {srcPtr, shmemAddr, b.i32_val(0), b.i32_val(cacheModifiers)});
      }
    }
  }
};

// AsyncCopyLocalToGlobalOpConversion removed - TritonNANOGPU dialect not available

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const NANO::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    // Determine the vectorization size
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    const size_t valueElemNBits =
        std::max<int>(8, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;

    auto cacheMod = op.getCache();
    const int numVecs = elemsPerThread / vec;
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      auto vecTy = LLVM::getVectorType(valueElemTy, vec);

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      Value elem = valueElems[vecStart];
      Value ptr = ptrElems[vecStart];

      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      llStore(rewriter, loc, ptr, storeVal, pred, cacheMod);
    } // end vec
    rewriter.eraseOp(op);
    return success();
  }
};




struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const NANO::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // extract relevant info from Module
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    // prep data by unpacking to get data ready
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);
    if (!atomicMemOrdering)
      return rewriter.notifyMatchFailure(op, "Unknown AMDGPU memory ordering");
    auto scope = getAMDGPUMemScopeStr(op.getScope());
    if (!scope)
      return rewriter.notifyMatchFailure(op, "Unknown AMDGPU memory scope");

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    Type valueElemIntTy{};
    if (!valueElemTy.isSignlessInteger()) {
      valueElemIntTy = rewriter.getIntegerType(valueElemNBits);
    }
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    SmallVector<Value> resultVals(elemsPerThread);

    auto successOrdering = *atomicMemOrdering;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto scopeStr = StringRef(scope.value());

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += 1) {
      Value casVal = valElements[i];
      Value casCmp = cmpElements[i];
      Value casPtr = ptrElements[i];
      if (valueElemIntTy) {
        casVal = LLVM::BitcastOp::create(rewriter, loc, valueElemIntTy, casVal);
        casCmp = LLVM::BitcastOp::create(rewriter, loc, valueElemIntTy, casCmp);
      }
      // use op
      if (tensorTy) { // for tensor
        auto retType = valueElemTy;
        // TODO: USE ATOMIC CAS OP on Tensor

        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, scopeStr);

        // Extract the new_loaded value from the pair.
        Value ret;
        if (valueElemIntTy) {
          ret = b.extract_val(valueElemIntTy, cmpxchg, 0);
          ret = LLVM::BitcastOp::create(rewriter, loc, valueElemTy, ret);
        } else {
          ret = b.extract_val(valueElemTy, cmpxchg, 0);
        }
        resultVals[i] = ret;
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = getThreadId(rewriter, loc);
        Value pred = b.icmp_eq(tid, b.i32_val(i));
        LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, scopeStr);

        if (!op.getResult().use_empty()) {
          // Extract the new_loaded value from the pair.
          Value newLoaded;
          if (valueElemIntTy) {
            newLoaded = b.extract_val(valueElemIntTy, cmpxchg, 0);
            newLoaded =
                LLVM::BitcastOp::create(rewriter, loc, valueElemTy, newLoaded);
          } else {
            newLoaded = b.extract_val(valueElemTy, cmpxchg, 0);
          }
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(newLoaded, atomPtr);
        }

        LLVM::BrOp::create(rewriter, loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        b.barrier(triton::gpu::AddrSpace::Local);
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }

    // FIXME: threadPred = b.true_val() is buggy
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, b.true_val(), targetInfo,
                                getTypeConverter());
    return success();
  }
};

bool supportsGlobalAtomicF16PackedAndDpp(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return true;
  default:
    break;
  }
  return false;
}

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const NANO::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto binOp = matchAtomicOp(op.getAtomicRmwOp());
    if (!binOp)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW operation");

    auto memOrder = getMemoryOrdering(op.getSem());
    if (!memOrder)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW memory order");

    auto scopeStr = getAMDGPUMemScopeStr(op.getScope());
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW scope");

    auto emitter =
        LLVM::NANO::AtomicRMWEmitter(targetInfo, *binOp, *memOrder, *scopeStr);

    Value val = op.getVal();
    Value ptr = op.getPtr();
    Value opResult = op.getResult();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();

    int numElems = 1;
    // In the case of unpaired f16 elements utilize dpp instructions to
    // accelerate atomics. Here is an algorithm of lowering
    // tt::atomicRmwOp(%ptr, %val, %mask):
    // 0. Group thread by pairs. Master thread is (tid % 2 == 0);
    // 1. All the threads send %val to (tid - 1) thread via dppUpdateOp shl, so
    //    all the masters receive value from secondary threads;
    // 2. Take into account parity in the %mask value, build control flow
    //    structures according to it;
    // 3. Generate llvm::atomicRmwOp in the threads enabled by %mask value;
    // 4. All the threads send result of generated operation to (tid + 1) thread
    //    via dppUpdateOp shl, so all secondary thread also receive their
    //    result.
    //
    // This approach enables us to use half the active threads committing atomic
    // requests to avoid generating of code providing unified access to f16
    // element and reduce contention.
    bool applyPackingF16 = false;
    auto vec = getVectorSize(ptr, axisAnalysisPass);
    if (llMask) {
      vec = std::min<unsigned>(vec, getMaskAlignment(op.getMask()));
    }

    // CDNA3/CDNA4 arch allows to accelerate its atomics with LDS reduction
    // algorithm, which is only applicable for atomics with no return. Otherwise
    // we have to deal with an additional overhead.
    bool enableIntraWaveReduce =
        llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                           targetInfo.getISAFamily()) &&
        tensorTy && opResult.use_empty();

    // TODO: support data types less than 32 bits
    enableIntraWaveReduce &= valueElemTy.getIntOrFloatBitWidth() >= 32;

    if (tensorTy) {
      bool isF16Ty = valueElemTy.isF16() || valueElemTy.isBF16();
      unsigned availableVecSize = isF16Ty ? 2 : 1;
      vec = std::min<unsigned>(vec, availableVecSize);
      // Force F16 packing in the case it's not coming in as packed, but the
      // ISA can support packed atomic instructions.
      applyPackingF16 =
          supportsGlobalAtomicF16PackedAndDpp(targetInfo.getISAFamily()) &&
          vec == 1 && isF16Ty && atomicRmwAttr == RMWOp::FADD &&
          !enableIntraWaveReduce;
      numElems = tensorTy.getNumElements();

      auto threadOrder = getThreadOrder(tensorTy);
      unsigned contigWithinLanes =
          axisAnalysisPass.getAxisInfo(ptr)->getContiguity(threadOrder.front());
      enableIntraWaveReduce &= contigWithinLanes == 1;
    }

    auto vecTy = vec_ty(valueElemTy, vec);
    auto elemsPerThread = getTotalElemsPerThread(val.getType());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    auto tid = getThreadId(rewriter, loc);

    bool needLdsStaging = !tensorTy && !opResult.use_empty();
    std::optional<Value> atomicSharedMemBase =
        op->hasAttr("allocation.offset") && needLdsStaging
            ? std::optional<Value>(getSharedMemoryBase(
                  loc, rewriter, targetInfo, op.getOperation()))
            : std::nullopt;

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? b.and_(threadPred, maskElements[i]) : threadPred;
      if (applyPackingF16) {
        resultVals[i] = emitter.emitPairedAtomicForEvenTID(
            rewriter, ptrElements[i], valElements[i], rmwMask);
      } else {
        Value valElement;
        if (vec == 1) {
          valElement = valElements[i];
        } else {
          Value vecVal = b.undef(vecTy);
          for (size_t ii = 0; ii < vec; ++ii)
            vecVal = b.insert_element(vecTy, vecVal, valElements[i + ii],
                                      b.i32_val(ii));
          valElement = vecVal;
        }

        // If we have a single tl.atomic_rmw that is lowered into multiple
        // llvm.atomic_rmw, and we set the ordering for each to aql_rel (the
        // default if no sem value is explicitly set in the DSL level
        // tl.atomic_add. The llvm backend will insert extra buffer invalidates
        // and L2 write backs causing a perforance degration. To avoid this we
        // set the ordering to release for the first, acquire for the last, and
        // relaxed for anything in between so that only a single set of
        // buffer_inv and buffer_wbl2 instructions are inserted by the backend
        // for any "cluster" of atomic ops.
        if ((vec > 1 || elemsPerThread > 1) &&
            op.getSem() == MemSemantic::ACQUIRE_RELEASE) {
          if (i == 0) {
            // First
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::release);
          } else if (i == elemsPerThread - vec) {
            // Last
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::acquire);
          } else {
            // Middle
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::monotonic);
          }
        }

        Value retVal =
            emitter.emitAtomicRMW(rewriter, ptrElements[i], valElement, rmwMask,
                                  atomicSharedMemBase, enableIntraWaveReduce);

        if (tensorTy) {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1
                    ? retVal
                    : b.extract_element(valueElemTy, retVal, b.i32_val(ii));
          }
        } else {
          if (!atomicSharedMemBase.has_value()) {
            rewriter.eraseOp(op);
            return success();
          }
          Value atomPtr = *atomicSharedMemBase;
          b.barrier(triton::gpu::AddrSpace::Local);
          Value ret = b.load(valueElemTy, atomPtr);

          rewriter.replaceOp(op, {ret});
          return success();
        }
      }
    }
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

// AsyncWaitOpConversion removed - TritonNANOGPU dialect not available

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Drop the result AsyncToken
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

// AsyncCopyMbarrierArriveOpConversion removed - TritonNANOGPU dialect not available

} // namespace

namespace mlir::triton::NANO {
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns.add<
      AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
      StoreOpConversion, AsyncCopyGlobalToLocalOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
  // AsyncWaitOpConversion, AsyncCopyLocalToGlobalOpConversion, AsyncCopyMbarrierArriveOpConversion
  // removed - TritonNANOGPU dialect not available
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::NANO
