add_triton_library(TritonNANOGPUToLLVM
    AsyncUtility.cpp
    AtomicRMWOpsEmitter.cpp
    AllocateSharedMemory.cpp
    TensorPtrOpsToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    UpcastMXFPToLLVM.cpp
    MembarUtility.cpp
    ScalarizePackedFOps.cpp
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
