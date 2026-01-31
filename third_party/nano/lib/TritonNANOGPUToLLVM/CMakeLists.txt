add_triton_library(TritonNANOGPUToLLVM
    ConvertLayoutOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    TargetUtils.cpp
    MembarUtility.cpp
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
