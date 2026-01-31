add_triton_library(TritonNANOGPUDialectToLLVM
    TritonNANOGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    ScaledUpcastToLLVM.cpp

    DEPENDS
    TritonNANOGPUIR
)
