/*
 * Triton Nano Backend - Python module for HIP kernel launching
 * All HIP types and implementations are in ISASupport.h
 */

#include <stddef.h>
#include <stdint.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Path to ISA runtime library (replaced by driver.py at compile time) */
const char *isaLibSearchPaths[] = {"/*py_libisa_search_path*/"};

#include "ISASupport.h"

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS, "Load NANO GPU binary"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get device properties"},
    {"build_signature_metadata", buildSignatureMetadata, METH_VARARGS,
     "Build signature metadata for launch"},
    {"launch", launchKernel, METH_VARARGS, "Launch kernel"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "nano_utils", NULL,
                                       -1, ModuleMethods};

PyMODINIT_FUNC PyInit_nano_utils(void) {
  if (!initSymbolTable())
    return NULL;

  PyObject *m = PyModule_Create(&ModuleDef);
  if (!m)
    return NULL;

  if (!initDataPtrStr())
    return NULL;

  return m;
}
