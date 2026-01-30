add_triton_library(TritonNANOAnalysis
  RangeAnalysis.cpp
  AxisInfoExt.cpp
  NANOGPUAllocation.cpp

  DEPENDS
  TritonTableGen
  TritonNANOGPUTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
