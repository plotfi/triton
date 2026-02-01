/*
 * Minimal HIP type definitions for Triton Nano backend.
 * Contains only the types needed for kernel launching via driver.c
 */

#ifndef HIP_MINIMAL_H
#define HIP_MINIMAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Basic pointer types */
typedef void* hipDeviceptr_t;
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;

/* Error codes */
typedef enum hipError_t {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorNotInitialized = 3,
  hipErrorDeinitialized = 4,
  hipErrorInvalidDevice = 101,
  hipErrorInvalidImage = 200,
  hipErrorInvalidContext = 201,
  hipErrorInvalidKernelFile = 218,
  hipErrorInvalidGraphicsContext = 219,
  hipErrorInvalidSource = 300,
  hipErrorFileNotFound = 301,
  hipErrorNoBinaryForGpu = 209,
  hipErrorNotFound = 500,
  hipErrorUnknown = 999
} hipError_t;

#define HIP_SUCCESS hipSuccess

/* UUID structure */
typedef struct hipUUID_t {
  char bytes[16];
} hipUUID;

/* Device architecture flags (simplified) */
typedef struct {
  unsigned hasGlobalInt32Atomics : 1;
  unsigned hasGlobalFloatAtomicExch : 1;
  unsigned hasSharedInt32Atomics : 1;
  unsigned hasSharedFloatAtomicExch : 1;
  unsigned hasFloatAtomicAdd : 1;
  unsigned hasGlobalInt64Atomics : 1;
  unsigned hasSharedInt64Atomics : 1;
  unsigned hasDoubles : 1;
  unsigned hasWarpVote : 1;
  unsigned hasWarpBallot : 1;
  unsigned hasWarpShuffle : 1;
  unsigned hasFunnelShift : 1;
  unsigned hasThreadFenceSystem : 1;
  unsigned hasSyncThreadsExt : 1;
  unsigned hasSurfaceFuncs : 1;
  unsigned has3dGrid : 1;
  unsigned hasDynamicParallelism : 1;
} hipDeviceArch_t;

/* Device properties - must match HIP ABI exactly */
typedef struct hipDeviceProp_t {
  char name[256];
  hipUUID uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  size_t sharedMemPerBlockOptin;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int maxBlocksPerMultiProcessor;
  int accessPolicyMaxWindowSize;
  size_t reservedSharedMemPerBlock;
  int hostRegisterSupported;
  int sparseHipArraySupported;
  int hostRegisterReadOnlySupported;
  int timelineSemaphoreInteropSupported;
  int memoryPoolsSupported;
  int gpuDirectRDMASupported;
  unsigned int gpuDirectRDMAFlushWritesOptions;
  int gpuDirectRDMAWritesOrdering;
  unsigned int memoryPoolSupportedHandleTypes;
  int deferredMappingHipArraySupported;
  int ipcEventSupported;
  int clusterLaunch;
  int unifiedFunctionPointers;
  int reserved[63];
  int hipReserved[32];
  /* HIP Only */
  char gcnArchName[256];
  size_t maxSharedMemoryPerMultiProcessor;
  int clockInstructionRate;
  hipDeviceArch_t arch;
  unsigned int* hdpMemFlushCntl;
  unsigned int* hdpRegFlushCntl;
  int cooperativeMultiDeviceUnmatchedFunc;
  int cooperativeMultiDeviceUnmatchedGridDim;
  int cooperativeMultiDeviceUnmatchedBlockDim;
  int cooperativeMultiDeviceUnmatchedSharedMem;
  int isLargeBar;
  int asicRevision;
} hipDeviceProp_t;

/* JIT options */
typedef enum hipJitOption {
  hipJitOptionMaxRegisters = 0,
  hipJitOptionThreadsPerBlock,
  hipJitOptionWallTime,
  hipJitOptionInfoLogBuffer,
  hipJitOptionInfoLogBufferSizeBytes,
  hipJitOptionErrorLogBuffer,
  hipJitOptionErrorLogBufferSizeBytes,
  hipJitOptionOptimizationLevel,
  hipJitOptionTargetFromContext,
  hipJitOptionTarget,
  hipJitOptionFallbackStrategy,
  hipJitOptionGenerateDebugInfo,
  hipJitOptionLogVerbose,
  hipJitOptionGenerateLineInfo,
  hipJitOptionCacheMode,
  hipJitOptionSm3xOpt,
  hipJitOptionFastCompile,
  hipJitOptionGlobalSymbolNames,
  hipJitOptionGlobalSymbolAddresses,
  hipJitOptionGlobalSymbolCount,
  hipJitOptionLto,
  hipJitOptionFtz,
  hipJitOptionPrecDiv,
  hipJitOptionPrecSqrt,
  hipJitOptionFma,
  hipJitOptionPositionIndependentCode,
  hipJitOptionMinCTAPerSM,
  hipJitOptionMaxThreadsPerBlock,
  hipJitOptionOverrideDirectiveValues,
  hipJitOptionNumOptions,
  hipJitOptionIRtoISAOptExt = 10000,
  hipJitOptionIRtoISAOptCountExt
} hipJitOption;

/* Function attributes */
typedef enum hipFunction_attribute {
  HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
  HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
  HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
  HIP_FUNC_ATTRIBUTE_NUM_REGS,
  HIP_FUNC_ATTRIBUTE_PTX_VERSION,
  HIP_FUNC_ATTRIBUTE_BINARY_VERSION,
  HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA,
  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
  HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
  HIP_FUNC_ATTRIBUTE_MAX
} hipFunction_attribute;

/* Pointer attributes */
typedef enum hipPointer_attribute {
  HIP_POINTER_ATTRIBUTE_CONTEXT = 1,
  HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
  HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
  HIP_POINTER_ATTRIBUTE_HOST_POINTER,
  HIP_POINTER_ATTRIBUTE_P2P_TOKENS,
  HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
  HIP_POINTER_ATTRIBUTE_BUFFER_ID,
  HIP_POINTER_ATTRIBUTE_IS_MANAGED,
  HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
  HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE,
  HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
  HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
  HIP_POINTER_ATTRIBUTE_MAPPED,
  HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
  HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
  HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,
  HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
} hipPointer_attribute;

/* Driver proc address query result */
typedef enum hipDriverProcAddressQueryResult {
  HIP_GET_PROC_ADDRESS_SUCCESS = 0,
  HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
  HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2
} hipDriverProcAddressQueryResult;

#ifdef __cplusplus
}
#endif

#endif /* HIP_MINIMAL_H */
