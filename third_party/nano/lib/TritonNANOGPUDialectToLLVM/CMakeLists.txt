add_triton_library(TritonNANOGPUDialectToLLVM
    TritonNANOGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp

    DEPENDS
    TritonNANOGPUIR
)
