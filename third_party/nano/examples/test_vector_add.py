#!/usr/bin/env python3
"""
Triton Nano Backend - Vector Addition Example

This is a simple test script to verify the nano backend works with a basic
vector addition kernel, similar to the tutorial at python/tutorials/01-vector-add.py.

To use the nano backend, you need to set the TRITON_BACKEND environment variable:
    TRITON_BACKEND=nano python test_vector_add.py

Or use the nano backend programmatically.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    """Simple vector addition kernel."""
    # Identify which program (block) we are
    pid = tl.program_id(axis=0)

    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements

    # Load x and y from DRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the addition
    output = x + y

    # Write the result back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Add two vectors using Triton."""
    # Allocate output tensor
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    n_elements = output.numel()

    # Define grid size (number of blocks)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


def test_vector_add():
    """Test the vector addition kernel."""
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

    # Compute reference result using PyTorch
    output_torch = x + y

    # Compute result using Triton
    output_triton = add(x, y)

    # Verify correctness
    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    print(f"\nResults:")
    print(f"  PyTorch output (first 5): {output_torch[:5].tolist()}")
    print(f"  Triton output (first 5):  {output_triton[:5].tolist()}")
    print(f"  Maximum difference: {max_diff}")

    if max_diff < 1e-6:
        print("\n[PASS] Vector addition test passed!")
        return True
    else:
        print("\n[FAIL] Vector addition test failed!")
        return False


if __name__ == "__main__":
    success = test_vector_add()
    exit(0 if success else 1)
