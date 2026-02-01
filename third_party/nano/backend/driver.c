// Minimal HIP driver for vector add
#define __HIP_PLATFORM_AMD__
#include <hip/hip_minimal.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// The list of paths to search for the HIP runtime library.
static const char *hipLibSearchPaths[] = {"/*py_libhip_search_path*/"};

#define HIP_SYMBOL_LIST(FOR_EACH_ERR_FN, FOR_EACH_STR_FN)                      \
  FOR_EACH_STR_FN(hipGetLastError)                                             \
  FOR_EACH_STR_FN(hipGetErrorString, hipError_t hipError)                      \
  FOR_EACH_ERR_FN(hipGetDeviceProperties, hipDeviceProp_t *prop, int deviceId) \
  FOR_EACH_ERR_FN(hipModuleLoadDataEx, hipModule_t *module, const void *image, \
                  unsigned int numOptions, hipJitOption *options,              \
                  void **optionValues)                                         \
  FOR_EACH_ERR_FN(hipModuleGetFunction, hipFunction_t *function,               \
                  hipModule_t module, const char *kname)                       \
  FOR_EACH_ERR_FN(hipFuncGetAttribute, int *, hipFunction_attribute attr,      \
                  hipFunction_t function)                                      \
  FOR_EACH_ERR_FN(hipModuleLaunchKernel, hipFunction_t f,                      \
                  unsigned int gridDimX, unsigned int gridDimY,                \
                  unsigned int gridDimZ, unsigned int blockDimX,               \
                  unsigned int blockDimY, unsigned int blockDimZ,              \
                  unsigned int sharedMemBytes, hipStream_t stream,             \
                  void **kernelParams, void **extra)                           \
  FOR_EACH_ERR_FN(hipPointerGetAttribute, void *data,                          \
                  hipPointer_attribute attribute, hipDeviceptr_t ptr)

#define TRITON_HIP_MSG_BUFF_SIZE (1024U)

struct HIPSymbolTable {
#define DEFINE_EACH_ERR_FIELD(hipSymbolName, ...)                              \
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#define DEFINE_EACH_STR_FIELD(hipSymbolName, ...)                              \
  const char *(*hipSymbolName)(__VA_ARGS__);
  HIP_SYMBOL_LIST(DEFINE_EACH_ERR_FIELD, DEFINE_EACH_STR_FIELD)
};

static struct HIPSymbolTable hipSymbolTable;

bool initSymbolTable() {
  void *lib = NULL;

  int n = sizeof(hipLibSearchPaths) / sizeof(hipLibSearchPaths[0]);
  for (int i = 0; i < n; ++i) {
    void *handle = dlopen(hipLibSearchPaths[i], RTLD_LAZY | RTLD_LOCAL);
    if (handle) {
      lib = handle;
    }
  }

  if (!lib) {
    PyErr_SetString(PyExc_RuntimeError, "cannot open libamdhip64.so");
    return false;
  }

  // Get HIP version
  int hipVersion = -1;
  typedef hipError_t (*hipDriverGetVersion_fn)(int *driverVersion);
  hipDriverGetVersion_fn hipDriverGetVersion =
      (hipDriverGetVersion_fn)dlsym(lib, "hipDriverGetVersion");
  if (!hipDriverGetVersion) {
    PyErr_SetString(PyExc_RuntimeError, "cannot query hipDriverGetVersion");
    dlclose(lib);
    return false;
  }
  hipDriverGetVersion(&hipVersion);

  // Get hipGetProcAddress
  typedef hipError_t (*hipGetProcAddress_fn)(
      const char *symbol, void **pfn, int hipVersion, uint64_t hipFlags,
      hipDriverProcAddressQueryResult *symbolStatus);
  hipGetProcAddress_fn hipGetProcAddress =
      (hipGetProcAddress_fn)dlsym(lib, "hipGetProcAddress");
  if (!hipGetProcAddress) {
    PyErr_SetString(PyExc_RuntimeError, "cannot query hipGetProcAddress");
    dlclose(lib);
    return false;
  }

  // Resolve symbols
  uint64_t hipFlags = 0;
  hipDriverProcAddressQueryResult symbolStatus;
#define QUERY_EACH_FN(hipSymbolName, ...)                                      \
  if (hipGetProcAddress(#hipSymbolName,                                        \
                        (void **)&hipSymbolTable.hipSymbolName, hipVersion,    \
                        hipFlags, &symbolStatus) != hipSuccess) {              \
    PyErr_SetString(PyExc_RuntimeError, "cannot get " #hipSymbolName);         \
    dlclose(lib);                                                              \
    return false;                                                              \
  }
  HIP_SYMBOL_LIST(QUERY_EACH_FN, QUERY_EACH_FN)

  return true;
}

static inline void gpuAssert(hipError_t code, const char *file, int line) {
  if (code != HIP_SUCCESS) {
    const char *str = hipSymbolTable.hipGetErrorString(code);
    char err[TRITON_HIP_MSG_BUFF_SIZE] = {0};
    snprintf(err, sizeof(err), "Triton Error [HIP]: Code: %d, Message: %s",
             code, str);
    PyGILState_STATE gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    PyGILState_Release(gil_state);
  }
}

#define HIP_CHECK_AND_RETURN_NULL(ans)                                         \
  do {                                                                         \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
    if (PyErr_Occurred())                                                      \
      return NULL;                                                             \
  } while (0)

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  hipDeviceProp_t props;
  HIP_CHECK_AND_RETURN_NULL(
      hipSymbolTable.hipGetDeviceProperties(&props, device_id));

  return Py_BuildValue(
      "{s:i, s:i, s:i, s:i, s:i, s:i, s:s, s:i, s:i, s:i}", "max_shared_mem",
      props.sharedMemPerBlock, "max_num_regs", props.regsPerBlock,
      "multiprocessor_count", props.multiProcessorCount, "sm_clock_rate",
      props.clockRate, "mem_clock_rate", props.memoryClockRate, "mem_bus_width",
      props.memoryBusWidth, "arch", props.gcnArchName, "warpSize",
      props.warpSize, "max_threads_per_sm", props.maxThreadsPerMultiProcessor,
      "cooperativeLaunch", props.cooperativeLaunch);
}

static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name;
  const char *data;
  Py_ssize_t data_size;
  int shared;
  int device;
  if (!PyArg_ParseTuple(args, "ss#ii", &name, &data, &data_size, &shared,
                        &device)) {
    return NULL;
  }

  hipJitOption opt[] = {hipJitOptionErrorLogBufferSizeBytes,
                        hipJitOptionErrorLogBuffer,
                        hipJitOptionInfoLogBufferSizeBytes,
                        hipJitOptionInfoLogBuffer, hipJitOptionLogVerbose};
  const unsigned int errbufsize = 8192;
  const unsigned int logbufsize = 8192;
  char _err[errbufsize];
  char _log[logbufsize];
  void *optval[] = {(void *)(uintptr_t)errbufsize, (void *)_err,
                    (void *)(uintptr_t)logbufsize, (void *)_log, (void *)1};

  hipModule_t mod;
  hipFunction_t fun;
  HIP_CHECK_AND_RETURN_NULL(
      hipSymbolTable.hipModuleLoadDataEx(&mod, data, 5, opt, optval));
  HIP_CHECK_AND_RETURN_NULL(
      hipSymbolTable.hipModuleGetFunction(&fun, mod, name));

  int n_regs = 0;
  int n_spills = 0;
  int32_t n_max_threads = 0;
  hipSymbolTable.hipFuncGetAttribute(&n_regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, fun);
  hipSymbolTable.hipFuncGetAttribute(&n_spills,
                                     HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fun);
  hipSymbolTable.hipFuncGetAttribute(
      &n_max_threads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fun);
  n_spills /= 4;

  if (PyErr_Occurred())
    return NULL;

  return Py_BuildValue("(KKiii)", (uint64_t)mod, (uint64_t)fun, n_regs,
                       n_spills, n_max_threads);
}

// Argument extraction functions
static PyObject *data_ptr_str = NULL;

bool extractPointer(void *ptr, PyObject *obj) {
  hipDeviceptr_t *dev_ptr = ptr;
  if (obj == Py_None) {
    *dev_ptr = (hipDeviceptr_t)0;
    return true;
  }
  if (PyLong_Check(obj)) {
    *dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(obj);
    return true;
  }
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {
    PyErr_SetString(PyExc_TypeError,
                    "Pointer argument must be uint64 or have data_ptr method");
    return false;
  }
  *dev_ptr = (hipDeviceptr_t)PyLong_AsUnsignedLongLong(ret);
  Py_DECREF(ret);
  if (*dev_ptr == 0)
    return true;

  hipError_t status = hipSymbolTable.hipPointerGetAttribute(
      dev_ptr, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER, *dev_ptr);
  if (status == hipErrorInvalidValue) {
    PyErr_SetString(PyExc_ValueError, "Pointer cannot be accessed from Triton");
    (void)hipSymbolTable.hipGetLastError();
    return false;
  }
  return true;
}

bool extractI32(void *ptr, PyObject *obj) {
  *((int32_t *)ptr) = PyLong_AsLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractU32(void *ptr, PyObject *obj) {
  *((uint32_t *)ptr) = PyLong_AsUnsignedLong(obj);
  return PyErr_Occurred() == NULL;
}

bool extractFP32(void *ptr, PyObject *obj) {
  double temp = PyFloat_AsDouble(obj);
  float f32 = (float)temp;
  *((uint32_t *)ptr) = *(uint32_t *)&f32;
  return PyErr_Occurred() == NULL;
}

typedef bool (*ExtractorFunc)(void *ptr, PyObject *obj);

typedef struct {
  ExtractorFunc extract;
  size_t size;
  const char *name[2];
} Extractor;

typedef enum {
  EXTRACTOR_UNKNOWN = 0,
  EXTRACTOR_POINTER = 1,
  EXTRACTOR_INT32 = 2,
  EXTRACTOR_UINT32 = 3,
  EXTRACTOR_FP32 = 4,
  EXTRACTOR_TYPE_COUNT
} ExtractorTypeIndex;

static Extractor extraction_map[EXTRACTOR_TYPE_COUNT] = {
    [EXTRACTOR_UNKNOWN] = {NULL, 0, {NULL}},
    [EXTRACTOR_POINTER] = {extractPointer, sizeof(hipDeviceptr_t), {NULL}},
    [EXTRACTOR_INT32] = {extractI32, sizeof(int32_t), {"i1", "i32"}},
    [EXTRACTOR_UINT32] = {extractU32, sizeof(uint32_t), {"u1", "u32"}},
    [EXTRACTOR_FP32] = {extractFP32, sizeof(uint32_t), {"fp32", "f32"}},
};

ExtractorTypeIndex getExtractorIndex(PyObject *type) {
  Py_ssize_t len = 0;
  const char *s = PyUnicode_AsUTF8AndSize(type, &len);
  if (!s || len < 2) {
    PyErr_Format(PyExc_RuntimeError, "Unexpected data type: %R", type);
    return EXTRACTOR_UNKNOWN;
  }
  if (s[0] == '*')
    return EXTRACTOR_POINTER;
  for (ExtractorTypeIndex i = EXTRACTOR_INT32; i < EXTRACTOR_TYPE_COUNT; i++) {
    for (int j = 0; j < 2 && extraction_map[i].name[j]; j++) {
      if (strcmp(s, extraction_map[i].name[j]) == 0)
        return i;
    }
  }
  PyErr_Format(PyExc_RuntimeError, "Unknown data type: %R", type);
  return EXTRACTOR_UNKNOWN;
}

static PyObject *buildSignatureMetadata(PyObject *self, PyObject *args) {
  PyObject *signature = NULL;
  if (!PyArg_ParseTuple(args, "O", &signature))
    return NULL;

  PyObject *fast = PySequence_Fast(signature, "Expected sequence");
  if (!fast)
    return NULL;

  Py_ssize_t size = PySequence_Fast_GET_SIZE(fast);
  PyObject **items = PySequence_Fast_ITEMS(fast);

  PyObject *ret = PyBytes_FromStringAndSize(NULL, size);
  if (!ret) {
    Py_DECREF(fast);
    return NULL;
  }

  char *buf = PyBytes_AS_STRING(ret);
  for (Py_ssize_t i = 0; i < size; ++i) {
    ExtractorTypeIndex idx = getExtractorIndex(items[i]);
    if (idx == EXTRACTOR_UNKNOWN) {
      Py_DECREF(fast);
      Py_DECREF(ret);
      return NULL;
    }
    buf[i] = (uint8_t)idx;
  }

  Py_DECREF(fast);
  return ret;
}

static PyObject *launchKernel(PyObject *self, PyObject *args) {
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int num_warps, shared_memory, warp_size;
  Py_buffer signature;
  PyObject *kernel_args = NULL;

  // Simplified argument parsing - removed hooks, profile scratch, cooperative
  if (!PyArg_ParseTuple(args, "iiiKKiiiy*O", &gridX, &gridY, &gridZ, &_stream,
                        &_function, &num_warps, &shared_memory, &warp_size,
                        &signature, &kernel_args)) {
    return NULL;
  }

  if (gridX * gridY * gridZ == 0) {
    PyBuffer_Release(&signature);
    Py_RETURN_NONE;
  }

  uint8_t *extractor_data = (uint8_t *)signature.buf;
  Py_ssize_t num_args = signature.len;

  PyObject *fast_args = PySequence_Fast(kernel_args, "Expected sequence");
  if (!fast_args) {
    PyBuffer_Release(&signature);
    return NULL;
  }
  PyObject **args_data = PySequence_Fast_ITEMS(fast_args);

  void **params = (void **)alloca(num_args * sizeof(void *));
  for (Py_ssize_t i = 0; i < num_args; ++i) {
    Extractor ext = extraction_map[extractor_data[i]];
    if (!ext.extract) {
      Py_DECREF(fast_args);
      PyBuffer_Release(&signature);
      return NULL;
    }
    params[i] = alloca(ext.size);
    if (!ext.extract(params[i], args_data[i])) {
      Py_DECREF(fast_args);
      PyBuffer_Release(&signature);
      return NULL;
    }
  }

  hipError_t err = hipSymbolTable.hipModuleLaunchKernel(
      (hipFunction_t)_function, gridX, gridY, gridZ, warp_size * num_warps, 1,
      1, shared_memory, (hipStream_t)_stream, params, 0);

  Py_DECREF(fast_args);
  PyBuffer_Release(&signature);

  if (err != hipSuccess) {
    gpuAssert(err, __FILE__, __LINE__);
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS, "Load HSACO binary"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get device properties"},
    {"build_signature_metadata", buildSignatureMetadata, METH_VARARGS,
     "Build signature metadata for launch"},
    {"launch", launchKernel, METH_VARARGS, "Launch kernel"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "hip_utils", NULL,
                                       -1, ModuleMethods};

PyMODINIT_FUNC PyInit_hip_utils(void) {
  if (!initSymbolTable())
    return NULL;

  PyObject *m = PyModule_Create(&ModuleDef);
  if (!m)
    return NULL;

  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if (!data_ptr_str)
    return NULL;

  return m;
}
