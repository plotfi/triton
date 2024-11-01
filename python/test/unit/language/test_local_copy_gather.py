import os
import shutil

import pytest

import torch
import triton
import re

import triton.language as tl

from torch._dynamo.testing import rand_strided
empty_strided_cuda = torch.empty_strided
reinterpret_tensor = torch.as_strided

@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 64,
                "RBLOCK": 64,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_global_gather(base_ptr, vec_ptr, ts_0_ptrs, ts_1_ptrs, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    rbase = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    ts_0 = tl.load(base_ptr + x0)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        r1 = rindex
        tmp7 = tl.load(vec_ptr + (r1), None, eviction_policy='evict_last').to(tl.float32)
        ts_1 = tl.load(ts_1_ptrs + rindex)
        ts = ts_0 - ts_1
        ts = tl.where(ts > 0, ts, 0)
        ts = tl.where(ts < xnumel, ts, xnumel-1)

        tmp4 = tl.load(base_ptr + ts)

        tmp5 = tmp4.to(tl.float32)
        tmp12 = _tmp11 + tmp5
        _tmp11 = tmp12
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tmp11.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp13, None)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 64,
                "RBLOCK": 64,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
@triton.jit
def triton_local_gather(base_ptr, vec_ptr, ts_0_ptrs, ts_1_ptrs, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    rbase = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    ts_0 = tl.load(base_ptr + x0)


    base_range = tl.arange(0, 4096)
    base_tensor = tl.load(base_ptr + base_range)
    # base_tensor = tl.reshape(base_tensor, (1, 4096))
    base_ptr_smem = tl.local_copy(base_tensor)

    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        r1 = rindex
        tmp7 = tl.load(vec_ptr + (r1), None, eviction_policy='evict_last').to(tl.float32)
        ts_1 = tl.load(ts_1_ptrs + rindex)
        ts = ts_0 - ts_1
        ts = tl.where(ts > 0, ts, 0)
        ts = tl.where(ts < xnumel, ts, xnumel-1)

        tmp4 = tl.gather(base_ptr_smem, ts)
        # tmp4 = tl.load(base_ptr + ts)

        tmp5 = tmp4.to(tl.float32)
        tmp12 = _tmp11 + tmp5
        _tmp11 = tmp12
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tmp11.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp13, None)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_local_copy_gather():
    S = (2048)
    base = rand_strided((2*S, ), (1, ), device='cuda:0', dtype=torch.int8)
    vec = rand_strided((S, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    ts_0 = rand_strided((2*S, ), (1, ), device='cuda:0', dtype=torch.int32)
    ts_1 = rand_strided((S, ), (1, ), device='cuda:0', dtype=torch.int32)
    S, = vec.shape
    xnumel = 2*S
    rnumel = S
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((2*S, ), (1, ), dtype=torch.bfloat16, device='cuda:0')
        buf2 = empty_strided_cuda((2*S, ), (1, ), dtype=torch.bfloat16, device='cuda:0')
        grid = lambda META: (
            triton.cdiv(2*S, META["XBLOCK"]),
        )
        triton_local_gather[grid](base, vec, ts_0, ts_1, buf1, xnumel, rnumel)
        triton_global_gather[grid](base, vec, ts_0, ts_1, buf2, xnumel, rnumel)
        result1 = (reinterpret_tensor(buf1, (2, S), (S, 1), 0), )
        result2 = (reinterpret_tensor(buf2, (2, S), (S, 1), 0), )
        torch.testing.assert_close(result1, result2, check_dtype=False)

