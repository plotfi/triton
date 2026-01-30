import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel_add(a, b, c):
    idx = tl.arange(0, 32)
    tl.store(c + idx, tl.load(a + idx) + tl.load(b + idx))

k = triton.compile(
    triton.compiler.ASTSource(fn=kernel_add, signature={"a": "*fp32", "b": "*fp32", "c": "*fp32"}, constexprs={}),
    target=GPUTarget("nano", "gfx942", 64))
ptx = k.asm["ttir"]


