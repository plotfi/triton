#ifndef TRITON_NANOGPU_DIALECT
#define TRITON_NANOGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonNANOGPU_Dialect : Dialect {
  let name = "amdg";
  let cppNamespace = "::mlir::triton::nanogpu";

  let description = [{
    TritonNANOGPU Dialect hosts AMD specific ops at TritonGPU abstraction level.
  }];

  let dependentDialects = ["triton::TritonDialect"];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
