#!/usr/bin/env python3
"""
Triton Nano Backend - Vector Addition Example

This is a simple test script to verify the nano backend works with a basic
vector addition kernel, similar to the tutorial at python/tutorials/01-vector-add.py.

To use the nano backend, you need to set the TRITON_BACKEND environment variable:
    export TRITON_DEFAULT_BACKEND=nano; python ./third_party/nano/test.py

Or use the nano backend programmatically.
"""

import torch
import triton
import triton.language as tl

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel_add(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, 32)
    tl.store(c + idx, tl.load(a + idx) + tl.load(b + idx))

def add(x: torch.Tensor, y: torch.Tensor, k) -> torch.Tensor:
    # Add two vectors using Triton.
    # Allocate output tensor
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()

    # Define grid size (number of blocks) - BLOCK_SIZE=1024 is baked into compiled kernel
    grid = (triton.cdiv(n_elements, 1024), 1, 1)

    # Launch kernel (BLOCK_SIZE is already compiled in as constexpr)
    k[grid](x, y, output, n_elements)

    return output

def test_vector_add():
    k = triton.compile(
        triton.compiler.ASTSource(
            fn=kernel_add,
            signature={"a": "*fp32", "b": "*fp32", "c": "*fp32", "n_elements": "i32"},
            constexprs={(4,): 1024}  # BLOCK_SIZE = 1024
        ),
        target=GPUTarget("nano", "gfx942", 64))
    ttir = k.asm["ttir"]
    ttgir = k.asm["ttgir"]
    llir = k.asm["llir"]

    print(f'TTIR:\n\n{ttir}\n\n')
    print(f'TTGIR:\n\n{ttgir}\n\n')
    print(f'LLIR:\n\n{llir}\n\n')

    # Test the vector addition kernel.
    print("=" * 60)
    print("Triton Nano Backend - Vector Addition Test")
    print("=" * 60)

    # Get device info
    device = triton.runtime.driver.active.get_active_torch_device()
    print(f"Device: {device}")

    # Create test tensors
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)

    print(f"Vector size: {size}")
    print(f"Input tensors allocated on: {x.device}")

    # Compute result using Triton
    output_triton = add(x, y, k)

    print(f"\nResults:")
    print(f"  Triton output (first 5):  {output_triton[:5].tolist()}")

if __name__ == "__main__":
    success = test_vector_add()
    exit(0 if success else 1)

