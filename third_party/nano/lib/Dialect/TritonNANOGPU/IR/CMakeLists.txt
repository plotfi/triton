add_triton_library(TritonNANOGPUIR
  Dialect.cpp

  DEPENDS
  TritonNANOGPUTableGen
  TritonNANOGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
