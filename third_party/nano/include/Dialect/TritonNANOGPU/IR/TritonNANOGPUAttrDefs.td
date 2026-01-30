/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_NANOGPU_ATTRDEFS
#define TRITON_NANOGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "TritonNANOGPUDialect.td"
include "mlir/IR/EnumAttr.td"

class TritonNANOGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TritonNANOGPU_Dialect, name, traits, baseCppClass> {
}

def SetFP8Clamping : TritonNANOGPU_Attr<"SetFP8Clamping"> {
  let mnemonic = "amdgcn.set.fp8.clamping";
}

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
  "SchedHint", "Instruction Scheduling Hints for AMD GPUs", [
    SchedHintCaseNone,
    SchedHintCaseAttention,
  ]>;

def TritonNANOGPU_SchedHintVariantAttr :
  TritonNANOGPU_I32EnumAttr<"SchedHintVariant", TritonNANOGPU_SchedHintsEnum>;

#endif
