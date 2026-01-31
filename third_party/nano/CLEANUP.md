# Nano Backend Cleanup

## Already Done

1. Created `hip_minimal.h` (~300 lines) replacing the full HIP headers (~10k+ lines)
2. Updated `driver.c` to use `hip_minimal.h` instead of `hip_runtime.h` and `hip_runtime_api.h`
3. Removed texture/surface, math library, and profiling includes from various headers
4. Removed TDM (Tensor Data Mover) support from `driver.c`, `driver.py`, and `compiler.py`
5. Removed ockl.bc dependency (printf stubbed out, mulhi uses LLVM intrinsics)
6. Removed TDM support from LLVM lowering:
   - Removed TDMUtility.cpp from CMakeLists.txt
   - Removed TDMUtility.h includes from all files
   - Removed TDM conversion patterns from LoadStoreOpToLLVM.cpp
   - Stubbed TensorPtrOpsToLLVM.cpp (was all TDM code)
   - Made supportsTDM() always return false in TargetInfo.cpp
   - Made supports_tdm Python binding always return false in triton_nano.cc
   - Removed AsyncTDMWait from ConvertWarpPipeline.cpp
   - Stubbed TDM operation verifiers in Dialect.cpp to emit errors

7. **Removed TritonNANOGPU dialect**:
   - Removed `include/Dialect/TritonNANOGPU/` from CMakeLists.txt (add_subdirectory commented out)
   - Removed `lib/Dialect/TritonNANOGPU/` from CMakeLists.txt (add_subdirectory commented out)
   - Removed `lib/TritonNANOGPUDialectToLLVM/` from CMakeLists.txt
   - Moved CommonUtils.h/cpp to `include/Utils/` and `lib/Analysis/`
   - Updated python/triton_nano.cc to not register nanogpu dialect
   - Updated all TritonNANOGPUToLLVM files to remove dialect includes:
     - TritonGPUToLLVM.cpp - removed dialect registration and pattern calls
     - MembarUtility.cpp - stubbed filterLDSMemoryBarriersDependencies
     - AsyncUtility.cpp - removed nanogpu::AsyncWaitOp and LocalLoadPackedTransposedOp
     - SPMDOpToLLVM.cpp - removed CondBarrierOpConversion
     - BarrierOpToLLVM.cpp - completely stubbed
     - MaskedOpsToLLVM.cpp - completely stubbed
     - Utility.cpp - rewrote llLoad/llStore to use LLVM ops directly
     - SchedInstructions.cpp - passes are now no-ops
     - ConvertWarpPipeline.cpp - removed CondBarrierOp usage
     - LoadStoreOpToLLVM.cpp - removed AsyncWaitOp, AsyncCopyLocalToGlobalOp, AsyncCopyMbarrierArriveOp
     - UpcastMXFPToLLVM.cpp - completely stubbed (UpcastMXFPOp pattern removed)
     - MemoryOpToLLVM.cpp - removed LocalLoadPackedTransposedOp, MemoryCounterWaitOp; updated BarrierOpConversion
     - ElementwiseOpToLLVM.cpp - replaced SetFP8ClampingAttr with simple UnitAttr
   - Analysis files updated:
     - AxisInfoExt.cpp - stubbed addVisitors
     - RangeAnalysis.cpp - removed ExtractSliceOp handling
     - NANOGPUAllocation.cpp - updated include path for CommonUtils.h

## Files to Remove (Not Needed for Vector Add)

Run this single command from `third_party/nano/` to remove all unnecessary files:

```bash
# Remove all unused HIP headers and libraries
rm -rf backend/include/roctracer/ backend/include/hipblas-common/ && \
rm -rf backend/include/hsa/ && \
rm -f backend/include/hip/hip_runtime.h \
      backend/include/hip/hip_runtime_api.h \
      backend/include/hip/hip_common.h \
      backend/include/hip/hip_version.h \
      backend/include/hip/hip_vector_types.h \
      backend/include/hip/hip_deprecated.h \
      backend/include/hip/driver_types.h \
      backend/include/hip/library_types.h \
      backend/include/hip/linker_types.h \
      backend/include/hip/amd_detail/amd_hip_runtime.h \
      backend/include/hip/amd_detail/amd_hip_common.h \
      backend/include/hip/amd_detail/amd_device_functions.h \
      backend/include/hip/amd_detail/amd_hip_vector_types.h \
      backend/include/hip/amd_detail/amd_warp_functions.h \
      backend/include/hip/amd_detail/amd_warp_sync_functions.h \
      backend/include/hip/amd_detail/amd_hip_atomic.h \
      backend/include/hip/amd_detail/amd_hip_unsafe_atomics.h \
      backend/include/hip/amd_detail/amd_hip_runtime_pt_api.h \
      backend/include/hip/amd_detail/host_defines.h \
      backend/include/hip/amd_detail/hip_assert.h \
      backend/include/hip/amd_detail/hip_prof_str.h \
      backend/include/hip/amd_detail/hip_runtime_prof.h \
      backend/include/hip/amd_detail/amd_math_functions.h \
      backend/include/hip/amd_detail/math_fwd.h \
      backend/include/hip/amd_detail/hip_fp16_math_fwd.h \
      backend/include/hip/amd_detail/device_library_decls.h \
      backend/include/hip/amd_detail/amd_channel_descriptor.h \
      backend/include/hip/amd_detail/amd_surface_functions.h \
      backend/include/hip/amd_detail/texture_fetch_functions.h \
      backend/include/hip/amd_detail/texture_indirect_functions.h \
      backend/include/hip/amd_detail/ockl_image.h \
      backend/include/hip/amd_detail/hip_ldg.h \
      backend/include/hip/amd_detail/amd_hip_gl_interop.h \
      backend/include/hip/channel_descriptor.h \
      backend/include/hip/texture_types.h \
      backend/include/hip/surface_types.h \
      backend/include/hip/hip_texture_types.h \
      backend/lib/ocml.bc \
      backend/lib/ockl.bc \
      backend/lib/asanrtl.bc \
      backend/include/TDMCommon.h \
      lib/TritonNANOGPUToLLVM/TDMUtility.h \
      lib/TritonNANOGPUToLLVM/TDMUtility.cpp \
      include/hipblas_types.h \
      include/hipblas_instance.h

# Clean up empty directories
rmdir backend/include/hip/amd_detail/ 2>/dev/null || true
```

## TritonNANOGPU Dialect - Removed

The TritonNANOGPU dialect has been completely removed from the build system. The directories still exist but are not built. All LLVM lowering patterns for dialect operations have been stubbed out or removed.

**Directories to delete (optional):**
- `include/Dialect/TritonNANOGPU/`
- `lib/Dialect/TritonNANOGPU/`
- `lib/TritonNANOGPUDialectToLLVM/`

**Removed dialect operations:**
- CondBarrierOp
- MemoryCounterWaitOp
- AsyncWaitOp
- AsyncCopyLocalToGlobalOp
- AsyncCopyMbarrierArriveOp
- LocalLoadPackedTransposedOp
- UpcastMXFPOp
- MaskedLoadOp/MaskedStoreOp
- InstructionSchedHint
- SetFP8ClampingAttr (replaced with simple UnitAttr)

## TDM Code - Status

TDM code has been stubbed out but the operations are still defined in TableGen (required for dialect to compile). Using TDM operations will emit errors at verification time.

Remaining TDM references (benign - stubbed or disabled):
- `include/Dialect/TritonNANOGPU/IR/TritonNANOGPUOps.td` - TDM operation definitions (kept for compilation, verifiers emit errors)
- `lib/TritonNANOGPUToLLVM/TargetInfo.h` - supportsTDM() declaration (returns false)

Files that can be deleted:
- `lib/TritonNANOGPUToLLVM/TDMUtility.h`
- `lib/TritonNANOGPUToLLVM/TDMUtility.cpp`
- `backend/include/TDMCommon.h`

## What Remains (Minimal for Vector Add)

```
backend/
├── __init__.py
├── compiler.py          # Compilation pipeline
├── driver.py            # Kernel launching (Python)
├── driver.c             # C extension for HIP calls
└── include/
    └── hip/
        └── hip_minimal.h  # Minimal HIP type definitions (~300 lines)
```

## HIP APIs Used (Dynamically Loaded)

| API | Purpose |
|-----|---------|
| `hipGetDeviceProperties` | Query GPU properties |
| `hipModuleLoadDataEx` | Load kernel binary |
| `hipModuleGetFunction` | Get kernel handle |
| `hipModuleLaunchKernel` | Launch kernel |
| `hipModuleLaunchCooperativeKernel` | Launch cooperative kernel |
| `hipDrvLaunchKernelEx` | Extended kernel launch |
| `hipGetLastError` | Error checking |
| `hipGetErrorString` | Error messages |
| `hipFuncGetAttribute` | Query kernel attributes |
| `hipPointerGetAttribute` | Query pointer properties |
| `hipDriverGetVersion` | Check HIP version |
| `hipGetProcAddress` | Get function pointers |

All APIs are dynamically loaded from `libamdhip64.so` at runtime via `dlopen`/`dlsym`.
Memory allocation, streams, events, etc. are handled by PyTorch.
