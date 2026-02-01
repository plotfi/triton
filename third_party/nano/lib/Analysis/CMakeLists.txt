add_triton_library(TritonNANOAnalysis
  AxisInfoExt.cpp

  DEPENDS
  TritonTableGen
  TritonNANOGPUTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
