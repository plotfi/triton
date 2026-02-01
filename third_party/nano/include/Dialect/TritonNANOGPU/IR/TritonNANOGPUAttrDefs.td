#ifndef TRITON_NANOGPU_ATTRDEFS
#define TRITON_NANOGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "TritonNANOGPUDialect.td"
include "mlir/IR/EnumAttr.td"

// These are required because useDefaultAttributePrinterParser
// is set on the dialect
class TritonNANOGPU_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::triton::nanogpu";
}
class TritonNANOGPU_I32EnumAttr<string mnemonic, TritonNANOGPU_I32Enum enumInfo> :
    EnumAttr<TritonNANOGPU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::triton::nanogpu";
}
def SchedHintCaseNone : I32EnumAttrCase<"none", 0>;
def SchedHintCaseAttention : I32EnumAttrCase<"attention", 2>;
def TritonNANOGPU_SchedHintsEnum : TritonNANOGPU_I32Enum<
  "SchedHint", "Instruction Scheduling Hints for LLVM Backend", [
    SchedHintCaseNone,
    SchedHintCaseAttention,
  ]>;
def TritonNANOGPU_SchedHintVariantAttr :
  TritonNANOGPU_I32EnumAttr<"SchedHintVariant", TritonNANOGPU_SchedHintsEnum>;

#endif
