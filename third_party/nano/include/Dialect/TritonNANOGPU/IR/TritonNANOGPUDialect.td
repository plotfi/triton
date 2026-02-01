#ifndef TRITON_NANOGPU_DIALECT
#define TRITON_NANOGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonNANOGPU_Dialect : Dialect {
  let name = "nanog";
  let cppNamespace = "::mlir::triton::nanogpu";

  let description = [{
    TritonNANOGPU Dialect is a placeholder for NANO specific ops.
  }];

  let dependentDialects = ["triton::TritonDialect"];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
