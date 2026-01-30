add_library(TritonNANOGPUTestAnalysis
  TestNANORangeAnalysis.cpp
  TestNANOGPUMembar.cpp
  TestAxisInfo.cpp
)
add_dependencies(TritonNANOGPUTestAnalysis
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
)
target_link_libraries(TritonNANOGPUTestAnalysis MLIRPass)
target_compile_options(TritonNANOGPUTestAnalysis PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
