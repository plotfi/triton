import contextlib
import functools
import pathlib
import re

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor

NUM_SMS: tl.constexpr = torch.cuda.get_device_properties("cuda").multi_processor_count


@triton.jit
def epilogue_fn(x: tl.tensor, EPILOGUE: tl.constexpr) -> tl.tensor:
    """Helper to share epilogues across multiple kernel impls."""
    M: tl.constexpr = x.shape[0]
    N: tl.constexpr = x.shape[1]

    if EPILOGUE == "noop_reshape":
        x = x.reshape(M, N // 2, 2)
        x = x.reshape(M, N)
        x = x.reshape(M, N // 2, 2)
        x = x.reshape(M, N)
        x = x.reshape(M, N // 2, 2)
        x = x.reshape(M, N)

    elif EPILOGUE == "faux_rotate":
        # In reality this is `x0 * cos - x1 * sin` and `x1 * cos + x0 * sin`
        x0, x1 = x.reshape(M, N // 2, 2).split()
        o0 = x0 - x1
        o1 = x1 + x0
        x = tl.interleave(o0, o1)

    elif EPILOGUE == "downcast_faux_rotate":
        x = x.to(tl.bfloat16)
        x0, x1 = x.reshape(M, N // 2, 2).split()
        o0 = x0 - x1
        o1 = x1 + x0
        x = tl.interleave(o0, o1)

    elif EPILOGUE == "downcast_upcast_faux_rotate":
        x = x.to(tl.bfloat16)
        x = x.to(tl.float32)
        x0, x1 = x.reshape(M, N // 2, 2).split()
        o0 = x0 - x1
        o1 = x1 + x0
        x = tl.interleave(o0, o1)

    elif EPILOGUE == "leaky_relu":
        x = tl.where(x >= 0, x, 0.01 * x)

    else:
        tl.static_assert(EPILOGUE == "")

    return x


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = epilogue_fn(accumulator, EPILOGUE)
    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(
                tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M
            )
            offs_bn = tl.max_contiguous(
                tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N
            )
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(
            a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
        )
        b = tl.load(
            b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
        )
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            accumulator = epilogue_fn(accumulator, EPILOGUE)
            c = accumulator.to(c_ptr.type.element_ty)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


@triton.jit
def matmul_kernel_tma_persistent(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPILOGUE: tl.constexpr,
):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            accumulator = epilogue_fn(accumulator, EPILOGUE)
            c = accumulator.to(dtype)

            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

def matmul(a, b, config, epilogue=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    result = matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        EPILOGUE=epilogue,
        **config,
    )
    return c, result


def matmul_persistent(a, b, config, epilogue=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    result = matmul_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        EPILOGUE=epilogue,
        **config,
    )
    return c, result


def _make_descriptor(x, block_size_i, block_size_j):
    return triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        x.data_ptr(),
        x.shape[0],
        x.shape[1],
        block_size_i,
        block_size_j,
        x.element_size(),
    )


def matmul_tma_persistent(a, b, config, epilogue=""):
    b = b.t()  # For consistency with `matmul` and `matmul_persistent`
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, K = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    result = matmul_kernel_tma_persistent[grid](
        _make_descriptor(a, config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]),
        _make_descriptor(b, config["BLOCK_SIZE_N"], config["BLOCK_SIZE_K"]),
        _make_descriptor(c, config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]),
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
        EPILOGUE=epilogue,
        **config,
    )
    return c, result


def get_cuda_autotune_config():
    return (
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        },
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        },
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        },
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        },
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        },
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        },
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 2,
            "num_stages": 5,
        },
        {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 2,
            "num_stages": 5,
        },
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 4,
        },
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 4,
        },
    )


def benchmark_fn(make_fn, config, cache):
    key = tuple(sorted(config.items()))
    if key not in cache:
        with contextlib.suppress(triton.runtime.errors.OutOfResources, RuntimeError):
            cache[key] = triton.testing.do_bench(make_fn(config))
    return cache.setdefault(key, None)


def coord_desc(factory, best_t, best_config, iters, threshold, cache):
    double = lambda x: 2 * x
    half = lambda x: x // 2
    add_one = lambda x: x + 1
    sub_one = lambda x: x - 1
    step_transforms = {
        "BLOCK_SIZE_M": (double, half),
        "BLOCK_SIZE_N": (double, half),
        "BLOCK_SIZE_K": (double, half),
        "GROUP_SIZE_M": (double, half),
        "num_warps": (double, half),
        "num_stages": (add_one, sub_one),
        ("BLOCK_SIZE_M", "BLOCK_SIZE_N"): (
            (double, half),
            (half, double),
            (double, double),
        ),
        ("BLOCK_SIZE_M", "BLOCK_SIZE_K"): (
            (double, half),
            (half, double),
            (double, double),
        ),
        ("BLOCK_SIZE_N", "BLOCK_SIZE_K"): (
            (double, half),
            (half, double),
            (double, double),
        ),
        ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "num_stages"): (
            (double, double, sub_one),
            (half, half, add_one),
        ),
        ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"): (
            (double, double, half),
            (half, half, double),
        ),
        ("BLOCK_SIZE_K", "num_stages"): ((double, sub_one), (half, add_one)),
    }

    for _ in range(iters):
        baseline_kwargs = best_config.copy()

        step_result = []
        for k, v in step_transforms.items():
            if isinstance(k, str):
                k = (k,)
                v = ((vi,) for vi in v)

            for transforms in v:
                test_kwargs = baseline_kwargs.copy()
                for ki, fn in zip(k, transforms, strict=True):
                    test_kwargs[ki] = fn(test_kwargs[ki])
                if not 1 <= test_kwargs["num_warps"] <= 8:
                    continue

                if (t := benchmark_fn(factory, test_kwargs, cache)) is not None:
                    step_result.append((t, test_kwargs))

        step_t, step_config = min(step_result, key=lambda x: x[0])
        if step_t / best_t >= 1 - threshold:
            break  # Converged

        best_t, best_config = step_t, step_config
    return best_t, best_config


def manual_autotune(factory, refine: bool = False):
    results = []
    cache = {}
    for config in get_cuda_autotune_config():

        if (t := benchmark_fn(factory, config, cache)) is not None:
            results.append((t, config))

    best_t, best_config = min(results, key=lambda x: x[0])
    if refine:
        best_t, best_config = coord_desc(
            factory, best_t, best_config, iters=10, threshold=0.01, cache=cache
        )

    return best_t, best_config


class BenchmarkMatmul:
    _base_fn = matmul

    def __init__(self, a, b, config, epilogue=""):
        self._a = a
        self._b = b
        self._config = config.copy()
        self._epilogue = epilogue

    def __call__(self):
        return self.__class__._base_fn(self._a, self._b, config=self._config, epilogue=self._epilogue)


class BenchmarkPersistentMatmul(BenchmarkMatmul):
    _base_fn = matmul_persistent


class BenchmarkPersistentTMAMatmul(BenchmarkMatmul):
    _base_fn = matmul_tma_persistent


class Hacked_BenchmarkPersistentMatmul(BenchmarkPersistentMatmul):

    def __init__(self, x, w, config, epilogue=""):
        super().__init__(x, w, config, epilogue)
        _, result = self.__class__._base_fn(x, w, config, epilogue)
        ttgir = self._rewrite(result.asm["ttgir"])

        # We use this path rather than tempfile to make it easier to inspect.
        scratch_path = pathlib.Path(__file__).parent / "scratch.ttgir"
        with open(scratch_path, "wt") as f:
            f.write(ttgir)

        self._kernel = triton.compile(str(scratch_path))

    def __call__(self):
        M, K = self._a.shape
        K, N = self._b.shape
        c = torch.empty((M, N), device=self._a.device, dtype=self._a.dtype)
        grid = (min(NUM_SMS, triton.cdiv(M, self._config["BLOCK_SIZE_M"]) * triton.cdiv(N, self._config["BLOCK_SIZE_N"])), 1, 1)
        result = self._kernel[grid](
            self._a, self._b, c,  #
            M, N, K,  #
            # Note: the other strides are one and were specialized out.
            self._a.stride(0),  #
            self._b.stride(1),  #
            c.stride(0),  #
        )
        return c, result

    @staticmethod
    def _rewrite(ttgir: str):
        """Awful terrible hacky brittle rewrite pass to re-establish dot pipeline.
        
        Rewrites:
        ```
        ... = triton_nvidia_gpu.warp_group_dot ...
        ... = triton_nvidia_gpu.warp_group_dot_wait ... {pendings = 0 : i32}
        ... = triton_gpu.convert_layout ...
        ```

        to

        ```
        ... = triton_nvidia_gpu.warp_group_dot ...
        ... = triton_nvidia_gpu.warp_group_dot_wait ... {pendings = 1 : i32}
        ...
        scf.if ... {
            // Epilogue
            ... = triton_nvidia_gpu.warp_group_dot_wait ... {pendings = 0 : i32}
            ... = triton_gpu.convert_layout ...
        ```
        """
        lines = ttgir.splitlines(False)
        dot_waits = [(idx, l) for idx, l in enumerate(lines) if "warp_group_dot_wait" in l and "pendings = 0" in l]
        if len(dot_waits) != 1:
            return ttgir

        ((idx, dot_wait),) = dot_waits
        assert "triton_gpu.convert_layout" in lines[idx + 1]
        convert_layout = lines[idx + 1]

        lines[idx] = dot_wait.replace("pendings = 0", "pendings = 1")
        lines[idx +1] = ""

        idy, *_ = [idx + idy for idy, l in enumerate(lines[idx:]) if "scf.if" in l]

        var_name = re.match(r"\s*%([0-9]+):", dot_wait).groups()[0]
        dot_wait = re.sub(rf"%{var_name}:", "%10000:", dot_wait)
        convert_layout = re.sub(rf"%{var_name}#", "%10000#", convert_layout)
        lines[idy] = f"{lines[idy]}\n  {dot_wait}\n  {convert_layout}"

        return "\n".join(lines)


class Hacked_BenchmarkPersistentTMAMatmul(Hacked_BenchmarkPersistentMatmul):
    _base_fn = matmul_tma_persistent

    def __call__(self):
        a = self._a
        b = self._b.t()  # For consistency with `matmul` and `matmul_persistent`
        assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
        assert a.dtype == b.dtype, "Incompatible dtypes"

        M, K = a.shape
        N, K = b.shape
        dtype = a.dtype

        c = torch.empty((M, N), device=a.device, dtype=dtype)
            

        grid = (min(NUM_SMS, triton.cdiv(M, self._config["BLOCK_SIZE_M"]) * triton.cdiv(N, self._config["BLOCK_SIZE_N"])), 1, 1)
        result = self._kernel[grid](
            _make_descriptor(a, self._config["BLOCK_SIZE_M"], self._config["BLOCK_SIZE_K"]),
            _make_descriptor(b, self._config["BLOCK_SIZE_N"], self._config["BLOCK_SIZE_K"]),
            _make_descriptor(c, self._config["BLOCK_SIZE_M"], self._config["BLOCK_SIZE_N"]),
            M, N, K,  #
        )
        return c, result


M: int = 768_000
N: int = 9 * 128
K: int = 3 * 128


if __name__ == "__main__":
    X: torch.Tensor = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    W: torch.Tensor = torch.randn((N, K), dtype=X.dtype, device=X.device).t()

    # print(f"Baseline: {triton.testing.do_bench(lambda: X @ W):>5.3f} ms")
    # for epilogue in ("", "leaky_relu", "noop_reshape", "faux_rotate", "downcast_faux_rotate", "downcast_upcast_faux_rotate"):
    #     print(f"\n{epilogue or 'No epilogue'}")
    #     for factory in [
    #         BenchmarkMatmul,
    #         BenchmarkPersistentMatmul,
    #         Hacked_BenchmarkPersistentMatmul,
    #         BenchmarkPersistentTMAMatmul,
    #         Hacked_BenchmarkPersistentTMAMatmul,
    #     ]:
    #         t, config = manual_autotune(functools.partial(factory, X, W, epilogue=epilogue), refine=True)
    #         name = "  (Hacked)" if isinstance(factory, Hacked_BenchmarkPersistentMatmul) else factory.__name__
    #         print(f"  {factory.__name__:<38} {t:>5.3f} ms  {config}")

    _, result = matmul_persistent(X, W, {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 4}, epilogue="noop_reshape")

    with open("scratch.ttgir", "wt") as f:
        f.write(result.asm["ttgir"])
