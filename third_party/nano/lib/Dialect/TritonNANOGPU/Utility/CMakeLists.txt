add_triton_library(TritonNANOUtils
  CommonUtils.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
