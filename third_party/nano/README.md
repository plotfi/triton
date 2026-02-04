# Triton Nano Backend

This is a minimal backend writen by plotfi, mostly using claude code, to
produce an ultra minimal result for experimentation and educational purposes.

Currently it is a minimal distillation of the Triton AMD backend but if you
wish to repurpose this backend for an ISA different than AMD, start by altering
the driver code, backend setup, and ISASupport.* ISA support and triton_nano.cc
files at:

```
backend/ISASupport.py
backend/include/ISASupport.h
include/TritonNANOGPUToLLVM/ISASupport.h
triton_nano.cc
```

