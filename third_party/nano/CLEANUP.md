# Nano Backend Cleanup

## Already Done

1. Created `hip_minimal.h` (~300 lines) replacing the full HIP headers (~10k+ lines)
2. Updated `driver.c` to use `hip_minimal.h` instead of `hip_runtime.h` and `hip_runtime_api.h`
3. Removed texture/surface, math library, and profiling includes from various headers
4. Removed TDM (Tensor Data Mover) support from `driver.c`, `driver.py`, and `compiler.py`

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
      backend/lib/asanrtl.bc \
      backend/include/TDMCommon.h \
      lib/TritonNANOGPUToLLVM/TDMUtility.h \
      lib/TritonNANOGPUToLLVM/TDMUtility.cpp \
      include/hipblas_types.h \
      include/hipblas_instance.h

# Clean up empty directories
rmdir backend/include/hip/amd_detail/ 2>/dev/null || true
```

## TDM Code Still To Remove

The following files still contain TDM references that need manual cleanup:

- `lib/Dialect/TritonNANOGPU/IR/Dialect.cpp` - TDM dialect operations
- `include/Dialect/TritonNANOGPU/IR/TritonNANOGPUOps.td` - TDM operation definitions
- `lib/TritonNANOGPUToLLVM/CMakeLists.txt` - Remove TDMUtility.cpp from sources
- `lib/TritonNANOGPUToLLVM/LoadStoreOpToLLVM.cpp` - TDM load/store lowering
- `lib/TritonNANOGPUToLLVM/TensorPtrOpsToLLVM.cpp` - TDM tensor pointer ops
- `lib/TritonNANOGPUToLLVM/TargetInfo.h` - TDM target info
- `lib/TritonNANOGPUToLLVM/TargetInfo.cpp` - TDM target info implementation
- `lib/TritonNANOGPUToLLVM/ConvertWarpPipeline.cpp` - TDM warp pipeline
- `python/triton_nano.cc` - TDM Python bindings

## What Remains (Minimal for Vector Add)

```
backend/
├── __init__.py
├── compiler.py          # Compilation pipeline
├── driver.py            # Kernel launching (Python)
├── driver.c             # C extension for HIP calls
├── include/
│   └── hip/
│       └── hip_minimal.h  # Minimal HIP type definitions (~300 lines)
└── lib/
    └── ockl.bc          # Device intrinsics library
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
