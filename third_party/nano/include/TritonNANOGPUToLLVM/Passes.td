#ifndef TRITONNANOGPU_CONVERSION_PASSES
#define TRITONNANOGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateNANOGPUSharedMemory : Pass<"allocate-nanogpu-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation";

  let description = [{
    This pass uses the `ModuleAllocation` analysis to:
      - Annotate modules with an attribute with the amount of shared/local
        memory used.
      - Annotate operations with an offset into the total shared/local memory.
  }];
}

def ConvertTritonNANOGPUToLLVM : Pass<"convert-triton-amdgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let constructor = "mlir::triton::createConvertTritonNANOGPUToLLVMPass(\"\", /*ftz=*/true)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::ROCDL::ROCDLDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def TritonNANOGPUInsertInstructionSchedHints : Pass<"triton-amdgpu-insert-instruction-sched-hints", "mlir::ModuleOp"> {
    let summary = "Insert instruction scheduling hints after the dot ops in the main loop";
    let constructor = "mlir::triton::createTritonNANOGPUInsertInstructionSchedHintsPass(/*variant=*/\"\")";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::triton::nanogpu::TritonNANOGPUDialect"];

    let options = [
        Option<"variant", "variant", "std::string", /*default*/"\"none\"",
               "instruction scheduling variant">,
    ];
}

def TritonNANOGPULowerInstructionSchedHints : Pass<"triton-amdgpu-lower-insert-instruction-sched-hints", "mlir::ModuleOp"> {
    let summary = "Lower instruction scheduling hints to LLVM intrinsics";
    let constructor = "mlir::triton::createTritonNANOGPULowerInstructionSchedHintsPass(/*arch=*/\"\",/*numStages=*/2)";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::triton::nanogpu::TritonNANOGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"numStages", "num_stages", "int32_t", /*default*/"2",
                "number of pipeline stages">,
    ];
}

def ConvertWarpPipeline : Pass<"convert-warp-pipeline", "mlir::ModuleOp"> {
    let summary = "Emit conditional barrier and inlines scf.execute_region for warp-pipeline";
    let constructor = "mlir::triton::NANO::createConvertWarpPipelinePass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::triton::nanogpu::TritonNANOGPUDialect"];
}

#endif
