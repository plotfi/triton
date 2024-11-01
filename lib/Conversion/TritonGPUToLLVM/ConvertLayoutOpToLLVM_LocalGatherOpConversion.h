/// __FACEBOOK__ (facebook) begin T203329359
/// third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertLayoutOpToLLVM_LocalGatherOpConversion.h
/// and
/// lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM_LocalGatherOpConversion.h
/// are nearly identical, the Nvidia version does use inline ptx to generate
/// shared memory loads that use predication to handle Triton masking.
#define ENABLE_INLINE_PTX_LDS 0

#if ENABLE_INLINE_PTX_LDS
inline Value
loadShared_NoPredicate(ConversionPatternRewriter &rewriter, Location loc,
                       const TypeConverter *converter, Value ptr,
                       Type elemTy, Value pred) {
  MLIRContext *ctx = rewriter.getContext();
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrTy.getAddressSpace() == 3 && "Invalid addr space for loadShared");
  unsigned bitwidth = std::max(8u, elemTy.getIntOrFloatBitWidth());

  const char *c = bitwidth == 64 ? "=l" : (bitwidth == 16 ? "=h" : "=r");

  PTXBuilder builder;
  auto *dOpr = builder.newOperand(c);
  auto *ptrOpr = builder.newAddrOperand(ptr, "r");
  auto &ld = builder.create<>("ld")->shared().b(bitwidth);
  // ld(dOpr, ptrOpr).predicate(pred, "b");
  return builder.launch(rewriter, loc, elemTy);
}

inline auto getSharedMemLoad(Location loc, const LLVMTypeConverter *typeConverter,
                      ConversionPatternRewriter &rewriter,
                      const TargetInfoBase &target, SmallVector<Value> &outVals,
                      unsigned i, uint64_t alignment, unsigned minVec,
                      Type elemLlvmTy, Type vecTy, Value basePtr, Value offset,
                      std::optional<Value> mask) {
  auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);
  Value smemAddr = gep(dstPtrTy, elemLlvmTy, basePtr, offset);
  smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));

  if (mask.has_value()) {
    auto valVec = target.loadShared(rewriter, loc,
                                    smemAddr, elemLlvmTy, mask.value());
    auto bitcastVal = bitcast(valVec, vecTy);
    for (unsigned v = 0; v < minVec; ++v) {
      Value currVal = extract_element(elemLlvmTy, bitcastVal, i32_val(v));
      outVals[i * minVec + v] = currVal;
    }

    return;
  }

  auto valVec = load(vecTy, smemAddr);
  valVec.setAlignment(alignment);

  for (unsigned v = 0; v < minVec; ++v) {
    Value currVal = extract_element(elemLlvmTy, valVec, i32_val(v));

    #if !ENABLE_INLINE_PTX_LDS
    if (mask.has_value()) {
      auto defaultVal = rewriter.create<LLVM::ConstantOp>(
          loc, elemLlvmTy, rewriter.getZeroAttr(elemLlvmTy));
      auto selectOp = select(mask.value(), currVal, defaultVal);
      outVals[i * minVec + v] = selectOp;
      continue;
    }
    #endif

    outVals[i * minVec + v] = currVal;
  }
}
#endif

inline SmallVector<Value> generateSharedLoads(
    Value dst, Value src, SharedMemoryObject smemObj, Type elemLlvmTy,
    Location loc,
    const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter,
    const TargetInfoBase &target,
    SmallVector<Value> localGatherIndices = {},
    SmallVector<Value> localGatherMask = {}) {
  auto dstTy = cast<RankedTensorType>(dst.getType());
  auto dstShape = dstTy.getShape();
  assert(dstShape.size() <= 2 && "Unexpected rank of loadSharedToDistributed");
  auto srcTy = cast<MemDescType>(src.getType());
  auto dstDistributedLayout = dstTy.getEncoding();
  auto srcSharedLayout =
      cast<triton::gpu::SharedEncodingAttr>(srcTy.getEncoding());
  auto srcElemTy = srcTy.getElementType();
  auto dstElemTy = dstTy.getElementType();
  auto inOrd = triton::gpu::getOrder(srcSharedLayout);
  auto outOrd = triton::gpu::getOrder(dstDistributedLayout);
  unsigned outVec = inOrd == outOrd
                        ? triton::gpu::getUniqueContigPerThread(
                              dstDistributedLayout, dstShape)[outOrd[0]]
                        : 1;

  unsigned inVec = srcSharedLayout.getMaxPhase() == 1
                       ? srcTy.getShape()[inOrd[0]]
                       : srcSharedLayout.getVec();
  unsigned minVec = std::min(outVec, inVec);
  unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
  SmallVector<Value> offsetVals = {smemObj.strides.size(), i32_val(0)};

  auto wordTy = vec_ty(elemLlvmTy, minVec);
  SmallVector<Value> outVals(outElems);
  auto width = elemLlvmTy.getIntOrFloatBitWidth();
  auto byteWidth = width / 8;
  for (unsigned i = 0; i < localGatherIndices.size(); ++i) {
    auto dstOffset = localGatherIndices[i];
#if !ENABLE_INLINE_PTX_LDS
    auto dstPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value smemAddr = gep(dstPtrTy, elemLlvmTy, smemObj.base, dstOffset);
    smemAddr = bitcast(smemAddr, ptr_ty(rewriter.getContext(), 3));
    auto valVec = load(wordTy, smemAddr);
    valVec.setAlignment(minVec * elemLlvmTy.getIntOrFloatBitWidth() / 8);

    if (localGatherMask.size()) {
      Value currVal = extract_element(elemLlvmTy, valVec, i32_val(0));
      auto defaultVal = rewriter.create<LLVM::ConstantOp>(
          loc, elemLlvmTy, rewriter.getZeroAttr(elemLlvmTy));

      auto selectOp = select(localGatherMask[i], currVal, defaultVal);
      outVals[i] = selectOp;
      continue;
    }

    for (unsigned v = 0; v < minVec; ++v) {
      Value currVal = extract_element(elemLlvmTy, valVec, i32_val(v));
      outVals[i * minVec + v] = currVal;
    }
#else

    uint64_t alignment = minVec * elemLlvmTy.getIntOrFloatBitWidth() / 8;
    std::optional<Value> mask;
    if (localGatherMask.size() > i) {
      mask = localGatherMask[i];
    }
    getSharedMemLoad(loc, typeConverter, rewriter, target,
                     outVals,
                     i,
                     alignment,
                     minVec,
                     elemLlvmTy, wordTy, smemObj.base, dstOffset, mask);
#endif
  }
  return outVals;
}

struct LocalGatherOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalGatherOp> {
public:
  LocalGatherOpConversion(LLVMTypeConverter &typeConverter,
                          const TargetInfoBase &targetInfo,
                          PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(triton::gpu::LocalGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemDescType srcTy = op.getSrc().getType();
    RankedTensorType dstTy = op.getType();
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    return lowerSharedToDistributed(op, adaptor, getTypeConverter(), rewriter);
  }

private:
  LogicalResult
  lowerSharedToDistributed(triton::gpu::LocalGatherOp op,
                           triton::gpu::LocalGatherOpAdaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() <= 2 &&
           "Unexpected rank of ConvertLayout(shared->blocked)");

    auto smemObj = getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(),
        typeConverter->convertType(srcTy.getElementType()), rewriter);
    auto elemLlvmTy = typeConverter->convertType(dstTy.getElementType());

    auto indices = unpackLLElements(loc, adaptor.getIndices(), rewriter);
    SmallVector<Value> maskElems;
    if (auto mask = adaptor.getMask())
      maskElems = unpackLLElements(loc, mask, rewriter);

#if 1
    auto elemTy = elemLlvmTy;
    SmallVector<Value> outVals =
        generateSharedLoads(op.getResult(), op.getSrc(),
                            smemObj, elemTy, loc,
                            typeConverter,
                            rewriter, targetInfo, indices, maskElems);
#else
    // USE AFTER PIN UPDATE:
    // After PIN Update, linear layout changes may cause a lot of rewrite for
    // generateSharedLoads
#endif

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, dstTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};
/// __FACEBOOK__ (facebook) end T203329359
