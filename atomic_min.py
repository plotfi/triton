import torch
import triton
import triton.language as tl

@triton.jit
def triton_(src0, src1, dst):
    offset = tl.load(src0, None)
    val = tl.load(src1, None)
    tl.atomic_add(dst + offset, val)

acc = torch.zeros(256, device="cuda")
idx = torch.randint(0, 256, (16 << 20,), device="cuda")
val = torch.ones(16 << 20, device="cuda")
triton_[(triton.cdiv(idx.numel(), 1024),)](idx, val, acc)
