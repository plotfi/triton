add_triton_library(TritonNANOGPUDialectToLLVM
    TritonNANOGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    InThreadTransposeOpToTTG.cpp
    ConcatOpToLLVM.cpp
    ScaledUpcastToLLVM.cpp

    DEPENDS
    TritonNANOGPUIR
)
