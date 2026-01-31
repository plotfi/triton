add_triton_library(TritonNANOGPUToLLVM
    AsyncUtility.cpp
    AtomicRMWOpsEmitter.cpp
    AllocateSharedMemory.cpp
    TensorPtrOpsToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    MaskedOpsToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    SPMDOpToLLVM.cpp
    UpcastMXFPToLLVM.cpp
    Fp4ToFpOpToLLVM.cpp
    MembarUtility.cpp
    ScalarizePackedFOps.cpp
    BarrierOpToLLVM.cpp
    WarpIdOpToLLVM.cpp

    DEPENDS
    TritonNANOGPUConversionPassIncGen
    LLVMIRIncGen

    LINK_LIBS PUBLIC
    MLIRReconcileUnrealizedCasts
    TritonGPUToLLVM
    TritonNANOGPUIR
    LLVMCore
    LLVMPasses
    LLVMSupport
)
