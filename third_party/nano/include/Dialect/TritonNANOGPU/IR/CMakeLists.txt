set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonNANOGPUOps.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=nanog)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=nanog)
mlir_tablegen(OpsConversions.inc -gen-llvmir-conversions)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(TritonNANOGPUDialect TritonNANOGPUDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TritonNANOGPUOps TritonNANOGPUOps dialects/ -gen-op-doc)
add_public_tablegen_target(TritonNANOGPUTableGen)

set(LLVM_TARGET_DEFINITIONS TritonNANOGPUAttrDefs.td)
mlir_tablegen(TritonNANOGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(TritonNANOGPUEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TritonNANOGPUAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonNANOGPUAttrDefs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(TritonNANOGPUAttrDefsIncGen)
