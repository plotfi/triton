// RUN: triton-opt %s --convert-triton-gpu-to-llvm  2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 4], order = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = false}>
module attributes
{
  "triton_gpu.num-ctas" = 1 : i32,
  "triton_gpu.num-warps" = 8 : i32,
  triton_gpu.shared = 64 : i32,
  triton_gpu.target = "cuda:90",
  "triton_gpu.threads-per-warp" = 32 : i32
}
{
  tt.func public @triton_global_gather(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                                       %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                                       %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                       %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                       %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
                                       %arg5: i32 {tt.divisibility = 16 : i32},
                                       %arg6: i32 {tt.divisibility = 16 : i32})
  attributes {noinline = false} {

    // Constants:
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32

    // Program TID for Store Op:
    %c64_i64 = arith.constant 64 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c64_i64 : i64
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %5 = arith.extsi %4 : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
    %6 = tt.splat %2 : i64 -> tensor<64x1xi64, #blocked>
    %7 = arith.addi %6, %5 : tensor<64x1xi64, #blocked>

    // Index Tensor Base:
    %14 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<1x64x!tt.ptr<i32>, #blocked>

    // PRELOAD GLOBAL TO LOCAL:
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %21 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x64x!tt.ptr<bf16>, #blocked>
    %22 = arith.extsi %20 : tensor<1x64xi32, #blocked> to tensor<1x64xi64, #blocked>
    %23 = tt.addptr %21, %22 : tensor<1x64x!tt.ptr<bf16>, #blocked>, tensor<1x64xi64, #blocked>
    %24 = tt.load %23 : tensor<1x64x!tt.ptr<bf16>, #blocked>

    // load from gmem
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16
    // CHECK: ld.global.b16 { $0 }, [ $1 + 0 ];", "=c,l,b" %{{.*}}, %{{.*}} : (!llvm.ptr<1>, i1) -> i16

    // 16 stores to shared memory to copy 64 tensor
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = {{.*}} : i64}  : vector<1xbf16>, !llvm.ptr<3>
    %25 = triton_gpu.local_alloc %24 {allocation.offset = 0 : i32} : (tensor<1x64xbf16, #blocked>) -> !tt.memdesc<1x64xbf16, #shared, #triton_gpu.shared_memory>

    cf.br ^bb1(%c0_i32, %cst : i32, tensor<64x64xbf16, #blocked>)

  ^bb1(%26: i32, %27: tensor<64x64xbf16, #blocked>):
    %28 = arith.cmpi slt, %26, %arg6 : i32
    cf.cond_br %28, ^bb2, ^bb3

  ^bb2:
    %29 = arith.extsi %26 : i32 to i64
    %30 = tt.splat %29 : i64 -> tensor<1x64xi64, #blocked>

    // Load the index tensor here:
    %32 = tt.addptr %14, %30 : tensor<1x64x!tt.ptr<i32>, #blocked>, tensor<1x64xi64, #blocked>
    %33 = tt.load %32 : tensor<1x64x!tt.ptr<i32>, #blocked>
    %34 = tt.broadcast %33 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>

    // GATHER FROM LOCAL:
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    %40 = triton_gpu.local_gather %25[%34] : (<1x64xbf16, #shared, #triton_gpu.shared_memory>, tensor<64x64xi32, #blocked>) -> tensor<64x64xbf16, #blocked>

    // Accumulate contents of gather:
    %42 = arith.addf %27, %40 : tensor<64x64xbf16, #blocked>

    // Increment Loop Index
    %43 = arith.addi %26, %c64_i32 : i32
    cf.br ^bb1(%43, %42 : i32, tensor<64x64xbf16, #blocked>)

  ^bb3:

    // Reduce Gathered Accumulation Result:
    %44 = "tt.reduce"(%27) <{axis = 1 : i32}> ({
    ^bb0(%arg7: bf16, %arg8: bf16):
      %49 = arith.addf %arg7, %arg8 : bf16
      tt.reduce.return %49 : bf16
    }) {allocation.offset = 0 : i32} : (tensor<64x64xbf16, #blocked>) -> tensor<64xbf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>>

    %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<64xbf16, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xbf16, #blocked>
    %46 = tt.splat %arg4 : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
    %47 = tt.addptr %46, %7 : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi64, #blocked>
    tt.store %47, %45 : tensor<64x1x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

