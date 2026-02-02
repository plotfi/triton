# Triton Nano Backend for AMDGPU

This is a minimal backend writen by plotfi, mostly using claude code, to produce
an ultra minimal reuslt for experimentation and for educational purposes.

If you wish to repurpose this backend for an ISA different than AMD, start by
altering the driver code, backend setup, and ISA support files at:

```
backend/__init__.py
backend/driver.py
backend/compiler.py
backend/driver.c
include/TritonNANOGPUToLLVM/ISASupport.h
triton_nano.cc
```

