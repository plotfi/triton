// RUN: triton-opt %s --convert-triton-gpu-to-llvm  2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#loc = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0)
#loc53 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":502:11)
#loc96 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:22)
#loc97 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:70)
#loc98 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:88)
#loc99 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:106)
#loc100 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:124)
#loc101 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:142)
#loc102 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:161)
#loc103 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:176)
#loc140 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":569:11)
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 110848 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_ragged_hstu_attn_fwd(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg8: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg9: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg10: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg11: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg12: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg13: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg14: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg15: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg16: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg17: i32 loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg18: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg19: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg20: f32 loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg21: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg22: i32 loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg23: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg24: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg25: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg26: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg27: i32 {tt.divisibility = 16 : i32} loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg28: i32 loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg29: i32 loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg30: f32 loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0), %arg31: f32 loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":369:0)) attributes {noinline = false} {
    %c-1_i32 = arith.constant -1 : i32 loc(#loc1)
    %cst = arith.constant dense<0> : tensor<64x1xi64, #blocked> loc(#loc1)
    %cst_0 = arith.constant dense<0> : tensor<1x64xi64, #blocked1> loc(#loc1)
    %cst_1 = arith.constant dense<9.99999997E-7> : tensor<64x64xf32, #blocked2> loc(#loc1)
    %cst_2 = arith.constant dense<0> : tensor<64x64xi32, #blocked2> loc(#loc1)
    %cst_3 = arith.constant dense<1> : tensor<64x64xi64, #blocked2> loc(#loc1)
    %cst_4 = arith.constant dense<0> : tensor<64x64xi64, #blocked2> loc(#loc1)
    %cst_5 = arith.constant dense<1> : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #blocked> loc(#loc1)
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #blocked1> loc(#loc1)
    %c63_i64 = arith.constant 63 : i64 loc(#loc1)
    %c64_i64 = arith.constant 64 : i64 loc(#loc1)
    %c2_i32 = arith.constant 2 : i32 loc(#loc1)
    %cst_8 = arith.constant 1.000000e+00 : f32 loc(#loc1)
    %c0_i64 = arith.constant 0 : i64 loc(#loc1)
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked2> loc(#loc1)
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #mma> loc(#loc1)
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #mma1> loc(#loc1)
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma1> loc(#loc1)
    %0 = tt.get_program_id y : i32 loc(#loc2)
    %1 = arith.divsi %0, %arg22 : i32 loc(#loc3)
    %2 = arith.remsi %0, %arg22 : i32 loc(#loc4)
    %3 = tt.addptr %arg3, %1 : !tt.ptr<i64>, i32 loc(#loc5)
    %4 = tt.load %3 : !tt.ptr<i64> loc(#loc6)
    %5 = tt.addptr %3, %c1_i32 : !tt.ptr<i64>, i32 loc(#loc7)
    %6 = tt.load %5 : !tt.ptr<i64> loc(#loc8)
    %7 = arith.subi %6, %4 : i64 loc(#loc9)
    %8 = arith.trunci %7 : i64 to i32 loc(#loc10)
    %9 = tt.get_program_id x : i32 loc(#loc11)
    %10 = arith.muli %9, %c64_i32 : i32 loc(#loc12)
    %11 = arith.cmpi sge, %10, %8 : i32 loc(#loc13)
    cf.cond_br %11, ^bb1, ^bb2 loc(#loc13)
  ^bb1:  // pred: ^bb0
    tt.return loc(#loc14)
  ^bb2:  // pred: ^bb0
    %12 = tt.addptr %arg7, %1 : !tt.ptr<i64>, i32 loc(#loc15)
    %13 = tt.load %12 : !tt.ptr<i64> loc(#loc16)
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc17)
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> loc(#loc17)
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc17)
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc17)
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc17)
    %19 = tt.splat %10 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> loc(#loc18)
    %20 = tt.splat %10 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc18)
    %21 = tt.splat %10 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc18)
    %22 = tt.splat %10 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc18)
    %23 = arith.addi %19, %15 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> loc(#loc18)
    %24 = arith.addi %20, %16 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc18)
    %25 = arith.addi %21, %14 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc18)
    %26 = arith.addi %22, %18 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc18)
    %27 = arith.muli %2, %arg10 : i32 loc(#loc19)
    %28 = tt.addptr %arg0, %27 : !tt.ptr<bf16>, i32 loc(#loc20)
    %29 = arith.extsi %arg9 : i32 to i64 loc(#loc21)
    %30 = arith.muli %4, %29 : i64 loc(#loc21)
    %31 = tt.addptr %28, %30 : !tt.ptr<bf16>, i64 loc(#loc22)
    %32 = arith.extsi %8 : i32 to i64 loc(#loc23)
    %33 = arith.extsi %10 : i32 to i64 loc(#loc23)
    %34 = arith.muli %2, %arg12 : i32 loc(#loc24)
    %35 = tt.addptr %arg1, %34 : !tt.ptr<bf16>, i32 loc(#loc25)
    %36 = arith.extsi %arg11 : i32 to i64 loc(#loc26)
    %37 = arith.muli %4, %36 : i64 loc(#loc26)
    %38 = tt.addptr %35, %37 : !tt.ptr<bf16>, i64 loc(#loc27)
    %39 = arith.muli %2, %arg14 : i32 loc(#loc28)
    %40 = tt.addptr %arg2, %39 : !tt.ptr<bf16>, i32 loc(#loc29)
    %41 = arith.extsi %arg13 : i32 to i64 loc(#loc30)
    %42 = arith.muli %4, %41 : i64 loc(#loc30)
    %43 = tt.addptr %40, %42 : !tt.ptr<bf16>, i64 loc(#loc31)
    %44 = tt.splat %8 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc32)
    %45 = tt.splat %8 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc32)
    %46 = arith.cmpi slt, %24, %44 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc32)
    %47 = arith.cmpi slt, %25, %45 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc32)
    %48 = arith.muli %1, %arg17 : i32 loc(#loc33)
    %49 = tt.addptr %arg4, %48 : !tt.ptr<i64>, i32 loc(#loc34)
    %50 = tt.splat %49 : !tt.ptr<i64> -> tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc35)
    %51 = tt.splat %49 : !tt.ptr<i64> -> tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc35)
    %52 = tt.addptr %50, %24 : tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>, tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc35)
    %53 = tt.addptr %51, %18 : tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc36)
    %54 = tt.addptr %52, %cst_5 : tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>, tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc37)
    %55 = tt.load %54, %46 : tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc38)

    // CHECK: llvm.bitcast %arg5 : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK: ld.global.v4.b32
    // CHECK: ld.global.v4.b32
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<8xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<8xbf16>, !llvm.ptr<3>
    %56 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> loc(#loc39)
    %57 = tt.expand_dims %56 {axis = 0 : i32} : tensor<2048xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2048xi32, #blocked3> loc(#loc40)
    %58 = tt.splat %arg5 : !tt.ptr<bf16> -> tensor<1x2048x!tt.ptr<bf16>, #blocked3> loc(#loc41)
    %59 = arith.extsi %57 : tensor<1x2048xi32, #blocked3> to tensor<1x2048xi64, #blocked3> loc(#loc42)
    %60 = tt.addptr %58, %59 : tensor<1x2048x!tt.ptr<bf16>, #blocked3>, tensor<1x2048xi64, #blocked3> loc(#loc43)
    %61 = tt.load %60 : tensor<1x2048x!tt.ptr<bf16>, #blocked3> loc(#loc44)
    %62 = triton_gpu.local_alloc %61 {allocation.offset = 0 : i32} : (tensor<1x2048xbf16, #blocked3>) -> !tt.memdesc<1x2048xbf16, #shared, #triton_gpu.shared_memory> loc(#loc45)

    %63 = tt.splat %31 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked> loc(#loc46)
    %64 = tt.splat %33 : i64 -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc46)
    %65 = arith.extsi %14 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc46)
    %66 = arith.extsi %17 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc46)
    %67 = arith.addi %64, %65 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc46)
    %68 = tt.expand_dims %67 {axis = 1 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked> loc(#loc46)
    %69 = tt.splat %29 : i64 -> tensor<64x1xi64, #blocked> loc(#loc46)
    %70 = arith.muli %68, %69 : tensor<64x1xi64, #blocked> loc(#loc46)
    %71 = tt.broadcast %70 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked> loc(#loc46)
    %72 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc46)
    %73 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc46)
    %74 = arith.extsi %72 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> loc(#loc46)
    %75 = arith.extsi %73 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> loc(#loc46)
    %76 = tt.expand_dims %74 {axis = 0 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi64, #blocked> loc(#loc46)
    %77 = tt.broadcast %76 : tensor<1x128xi64, #blocked> -> tensor<64x128xi64, #blocked> loc(#loc46)
    %78 = arith.addi %71, %77 : tensor<64x128xi64, #blocked> loc(#loc46)
    %79 = tt.addptr %63, %78 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi64, #blocked> loc(#loc46)
    %80 = arith.cmpi sge, %68, %cst : tensor<64x1xi64, #blocked> loc(#loc46)
    %81 = tt.splat %32 : i64 -> tensor<64x1xi64, #blocked> loc(#loc46)
    %82 = arith.cmpi slt, %68, %81 : tensor<64x1xi64, #blocked> loc(#loc46)
    %83 = arith.andi %80, %82 : tensor<64x1xi1, #blocked> loc(#loc46)
    %84 = tt.broadcast %83 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc46)
    %85 = tt.load %79, %84, %cst_6 : tensor<64x128x!tt.ptr<bf16>, #blocked> loc(#loc46)
    %86 = triton_gpu.local_alloc %85 {allocation.offset = 45056 : i32} : (tensor<64x128xbf16, #blocked>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> loc(#loc46)
    %87 = arith.subi %32, %13 : i64 loc(#loc47)
    %88 = arith.addi %87, %c63_i64 : i64 loc(#loc160)
    %89 = arith.divsi %88, %c64_i64 : i64 loc(#loc50)
    %90 = arith.muli %89, %c64_i64 : i64 loc(#loc51)
    %91 = arith.cmpi slt, %90, %33 : i64 loc(#loc52)
    cf.cond_br %91, ^bb3, ^bb4 loc(#loc53)
  ^bb3:  // pred: ^bb2
    %92 = arith.trunci %13 : i64 to i32 loc(#loc54)
    %93 = arith.subi %8, %92 : i32 loc(#loc55)
    cf.br ^bb5(%93 : i32) loc(#loc53)
  ^bb4:  // pred: ^bb2
    %94 = arith.addi %10, %c64_i32 : i32 loc(#loc56)
    cf.br ^bb5(%94 : i32) loc(#loc53)
  ^bb5(%95: i32 loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":502:11)):  // 2 preds: ^bb3, ^bb4
    cf.br ^bb6 loc(#loc53)
  ^bb6:  // pred: ^bb5
    %96 = tt.splat %38 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>, #blocked1> loc(#loc161)
    %97 = tt.expand_dims %75 {axis = 1 : i32} : tensor<128xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1> loc(#loc161)
    %98 = tt.broadcast %97 : tensor<128x1xi64, #blocked1> -> tensor<128x64xi64, #blocked1> loc(#loc161)
    %99 = tt.splat %36 : i64 -> tensor<1x64xi64, #blocked1> loc(#loc161)
    %100 = tt.splat %32 : i64 -> tensor<1x64xi64, #blocked1> loc(#loc161)
    %101 = tt.splat %arg20 : f32 -> tensor<64x64xf32, #mma1> loc(#loc162)
    %102 = tt.expand_dims %55 {axis = 1 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi64, #blocked2> loc(#loc163)
    %103 = tt.broadcast %102 : tensor<64x1xi64, #blocked2> -> tensor<64x64xi64, #blocked2> loc(#loc164)
    %104 = tt.splat %arg31 : f32 -> tensor<64x64xf32, #blocked2> loc(#loc165)
    %105 = arith.sitofp %arg29 : i32 to f32 loc(#loc166)
    %106 = arith.divf %cst_8, %105 : f32 loc(#loc166)
    %107 = tt.splat %106 : f32 -> tensor<64x64xf32, #blocked2> loc(#loc167)
    %108 = arith.divf %cst_8, %arg30 : f32 loc(#loc168)
    %109 = tt.splat %108 : f32 -> tensor<64x64xf32, #blocked2> loc(#loc169)
    %110 = tt.splat %arg27 : i32 -> tensor<64x64xi32, #blocked2> loc(#loc170)
    %111 = tt.expand_dims %46 {axis = 1 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi1, #blocked2> loc(#loc171)
    %112 = tt.broadcast %111 : tensor<64x1xi1, #blocked2> -> tensor<64x64xi1, #blocked2> loc(#loc172)
    %113 = tt.splat %arg5 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked2> loc(#loc173)
    %114 = tt.splat %87 : i64 -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc174)
    %115 = tt.splat %87 : i64 -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc174)
    %116 = arith.extsi %24 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc174)
    %117 = arith.cmpi slt, %116, %114 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc174)
    %118 = arith.select %117, %116, %114 : tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>, tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> loc(#loc175)
    %119 = tt.expand_dims %118 {axis = 1 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi64, #blocked2> loc(#loc176)
    %120 = tt.broadcast %119 : tensor<64x1xi64, #blocked2> -> tensor<64x64xi64, #blocked2> loc(#loc177)
    %121 = arith.extsi %arg28 : i32 to i64 loc(#loc178)
    %122 = tt.splat %121 : i64 -> tensor<64x64xi64, #blocked2> loc(#loc178)
    %123 = arith.muli %arg28, %c2_i32 : i32 loc(#loc179)
    %124 = arith.subi %123, %c2_i32 : i32 loc(#loc180)
    %125 = arith.extsi %124 : i32 to i64 loc(#loc181)
    %126 = tt.splat %125 : i64 -> tensor<64x64xi64, #blocked2> loc(#loc181)
    %127 = tt.splat %arg6 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked2> loc(#loc182)

    // CHECK: llvm.bitcast %arg6 : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK: ld.global.v4.b32
    // CHECK: ld.global.v4.b32
    // CHECK: ld.global.v4.b32
    // CHECK: ld.global.v4.b32
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<8xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<8xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<8xbf16>, !llvm.ptr<3>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<8xbf16>, !llvm.ptr<3>
    %128 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> loc(#loc80)
    %129 = tt.expand_dims %128 {axis = 0 : i32} : tensor<4096xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x4096xi32, #blocked3> loc(#loc81)
    %130 = tt.splat %arg6 : !tt.ptr<bf16> -> tensor<1x4096x!tt.ptr<bf16>, #blocked3> loc(#loc82)
    %131 = arith.extsi %129 : tensor<1x4096xi32, #blocked3> to tensor<1x4096xi64, #blocked3> loc(#loc83)
    %132 = tt.addptr %130, %131 : tensor<1x4096x!tt.ptr<bf16>, #blocked3>, tensor<1x4096xi64, #blocked3> loc(#loc84)
    %133 = tt.load %132 : tensor<1x4096x!tt.ptr<bf16>, #blocked3> loc(#loc85)
    %134 = triton_gpu.local_alloc %133 {allocation.offset = 4096 : i32} : (tensor<1x4096xbf16, #blocked3>) -> !tt.memdesc<1x4096xbf16, #shared, #triton_gpu.shared_memory> loc(#loc86)

    %135 = arith.sitofp %arg23 : i32 to f32 loc(#loc183)
    %136 = arith.divf %cst_8, %135 : f32 loc(#loc183)
    %137 = tt.splat %136 : f32 -> tensor<64x64xf32, #mma1> loc(#loc184)
    %138 = tt.expand_dims %23 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #mma1}>> -> tensor<64x1xi32, #mma1> loc(#loc185)
    %139 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> loc(#loc186)
    %140 = tt.expand_dims %139 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma1}>> -> tensor<1x64xi32, #mma1> loc(#loc186)
    %141 = tt.broadcast %138 : tensor<64x1xi32, #mma1> -> tensor<64x64xi32, #mma1> loc(#loc187)
    %142 = tt.splat %87 : i64 -> tensor<1x64xi64, #mma1> loc(#loc188)
    %143 = tt.splat %43 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked> loc(#loc189)
    %144 = tt.splat %41 : i64 -> tensor<64x1xi64, #blocked> loc(#loc189)
    %145 = triton_gpu.local_alloc  {allocation.offset = 12288 : i32} : () -> !tt.memdesc<2x128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc161)
    %146 = triton_gpu.local_alloc  {allocation.offset = 61440 : i32} : () -> !tt.memdesc<2x64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc189)
    %147 = arith.cmpi sgt, %95, %c0_i32 : i32 loc(#loc94)
    %148 = tt.expand_dims %66 {axis = 0 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi64, #blocked1> loc(#loc161)
    %149 = arith.muli %148, %99 : tensor<1x64xi64, #blocked1> loc(#loc161)
    %150 = tt.broadcast %149 : tensor<1x64xi64, #blocked1> -> tensor<128x64xi64, #blocked1> loc(#loc161)
    %151 = arith.addi %98, %150 : tensor<128x64xi64, #blocked1> loc(#loc161)
    %152 = tt.addptr %96, %151 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi64, #blocked1> loc(#loc161)
    %153 = arith.cmpi sge, %148, %cst_0 : tensor<1x64xi64, #blocked1> loc(#loc161)
    %154 = arith.cmpi slt, %148, %100 : tensor<1x64xi64, #blocked1> loc(#loc161)
    %155 = arith.andi %153, %154 : tensor<1x64xi1, #blocked1> loc(#loc161)
    %156 = tt.broadcast %155 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc161)
    %157 = triton_gpu.memdesc_subview %145[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc161)
    %158 = tt.splat %147 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc94)
    %159 = arith.andi %158, %156 : tensor<128x64xi1, #blocked1> loc(#loc94)
    %160 = triton_gpu.async_copy_global_to_local %152, %157 mask %159 other %cst_7 : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc161)
    %161 = triton_gpu.async_commit_group %160 loc(#loc161)
    %162 = tt.expand_dims %65 {axis = 1 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked> loc(#loc189)
    %163 = arith.muli %162, %144 : tensor<64x1xi64, #blocked> loc(#loc189)
    %164 = tt.broadcast %163 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked> loc(#loc189)
    %165 = arith.addi %164, %77 : tensor<64x128xi64, #blocked> loc(#loc189)
    %166 = tt.addptr %143, %165 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi64, #blocked> loc(#loc189)
    %167 = arith.cmpi sge, %162, %cst : tensor<64x1xi64, #blocked> loc(#loc189)
    %168 = arith.cmpi slt, %162, %81 : tensor<64x1xi64, #blocked> loc(#loc189)
    %169 = arith.andi %167, %168 : tensor<64x1xi1, #blocked> loc(#loc189)
    %170 = tt.broadcast %169 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc189)
    %171 = triton_gpu.memdesc_subview %146[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<2x64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc189)
    %172 = tt.splat %147 : i1 -> tensor<64x128xi1, #blocked> loc(#loc94)
    %173 = arith.andi %172, %170 : tensor<64x128xi1, #blocked> loc(#loc94)
    %174 = triton_gpu.async_copy_global_to_local %166, %171 mask %173 other %cst_6 : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc189)
    %175 = triton_gpu.async_commit_group %174 loc(#loc189)
    triton_nvidia_gpu.fence_async_shared {bCluster = false} loc(#loc190)
    cf.br ^bb7(%c0_i32, %cst_10, %c0_i64, %c0_i64, %c0_i32, %c-1_i32, %161, %175 : i32, tensor<64x128xf32, #mma>, i64, i64, i32, i32, !triton_gpu.async.token, !triton_gpu.async.token) loc(#loc94)
  ^bb7(%176: i32 loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:22), %177: tensor<64x128xf32, #mma> loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:70), %178: i64 loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:88), %179: i64 loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:106), %180: i32 loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:124), %181: i32 loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:142), %182: !triton_gpu.async.token loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:161), %183: !triton_gpu.async.token loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":219:176)):  // 2 preds: ^bb6, ^bb8
    %184 = arith.cmpi slt, %176, %95 : i32 loc(#loc94)
    cf.cond_br %184, ^bb8, ^bb9 loc(#loc94)
  ^bb8:  // pred: ^bb7
    %185 = arith.subi %95, %c64_i32 : i32 loc(#loc94)
    %186 = arith.cmpi slt, %176, %185 : i32 loc(#loc94)
    %187 = arith.addi %181, %c1_i32 : i32 loc(#loc94)
    %188 = arith.cmpi slt, %187, %c2_i32 : i32 loc(#loc94)
    %189 = arith.select %188, %187, %c0_i32 : i32 loc(#loc94)
    %190 = arith.subi %8, %176 : i32 loc(#loc191)
    %191 = tt.splat %190 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc192)
    %192 = arith.cmpi slt, %18, %191 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc192)
    %193 = triton_gpu.async_wait %182 {num = 1 : i32} loc(#loc161)
    %194 = triton_gpu.memdesc_subview %145[%189, %c0_i32, %c0_i32] : !tt.memdesc<2x128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc161)
    %195 = triton_nvidia_gpu.warp_group_dot %86, %194, %cst_12 {isAsync = true} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> -> tensor<64x64xf32, #mma1> loc(#loc190)
    %196:3 = triton_nvidia_gpu.warp_group_dot_wait %195, %86, %194 {pendings = 0 : i32} : tensor<64x64xf32, #mma1>, !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>, !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc190)
    %197 = arith.mulf %196#0, %101 : tensor<64x64xf32, #mma1> loc(#loc162)
    %198 = tt.splat %176 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc193)
    %199 = tt.addptr %53, %198 : tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc193)
    %200 = tt.load %199, %192 : tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc194)
    %201 = tt.expand_dims %200 {axis = 0 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi64, #blocked2> loc(#loc195)
    %202 = tt.broadcast %201 : tensor<1x64xi64, #blocked2> -> tensor<64x64xi64, #blocked2> loc(#loc164)
    %203 = arith.subi %103, %202 : tensor<64x64xi64, #blocked2> loc(#loc164)
    %204 = arith.sitofp %203 : tensor<64x64xi64, #blocked2> to tensor<64x64xf32, #blocked2> loc(#loc165)
    %205 = arith.addf %204, %104 : tensor<64x64xf32, #blocked2> loc(#loc165)
    %206 = arith.cmpf ogt, %205, %cst_1 : tensor<64x64xf32, #blocked2> loc(#loc196)
    %207 = arith.select %206, %205, %cst_1 : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2> loc(#loc197)
    %208 = arith.mulf %207, %107 : tensor<64x64xf32, #blocked2> loc(#loc167)
    %209 = math.sqrt %208 : tensor<64x64xf32, #blocked2> loc(#loc198)
    %210 = arith.mulf %209, %109 : tensor<64x64xf32, #blocked2> loc(#loc169)
    %211 = arith.fptosi %210 : tensor<64x64xf32, #blocked2> to tensor<64x64xi32, #blocked2> loc(#loc199)
    %212 = arith.cmpi sgt, %211, %cst_2 : tensor<64x64xi32, #blocked2> loc(#loc200)
    %213 = arith.select %212, %211, %cst_2 : tensor<64x64xi1, #blocked2>, tensor<64x64xi32, #blocked2> loc(#loc201)
    %214 = tt.expand_dims %192 {axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi1, #blocked2> loc(#loc202)
    %215 = tt.broadcast %214 : tensor<1x64xi1, #blocked2> -> tensor<64x64xi1, #blocked2> loc(#loc172)
    %216 = arith.andi %112, %215 : tensor<64x64xi1, #blocked2> loc(#loc172)

    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    %217 = triton_gpu.local_gather %62[%213], %216 : (<1x2048xbf16, #shared, #triton_gpu.shared_memory>, tensor<64x64xi32, #blocked2>, tensor<64x64xi1, #blocked2>) -> tensor<64x64xbf16, #blocked2> loc(#loc201)

    %218 = arith.extf %217 : tensor<64x64xbf16, #blocked2> to tensor<64x64xf32, #blocked2> loc(#loc203)
    %219 = arith.addf %218, %cst_9 : tensor<64x64xf32, #blocked2> loc(#loc203)
    %220 = arith.addi %18, %198 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc204)
    %221 = arith.extsi %220 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc205)
    %222 = arith.cmpi slt, %221, %115 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc205)
    %223 = arith.select %222, %221, %115 : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>, tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc206)
    %224 = tt.expand_dims %223 {axis = 0 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi64, #blocked2> loc(#loc207)
    %225 = tt.broadcast %224 : tensor<1x64xi64, #blocked2> -> tensor<64x64xi64, #blocked2> loc(#loc177)
    %226 = arith.subi %225, %120 : tensor<64x64xi64, #blocked2> loc(#loc177)
    %227 = arith.addi %226, %122 : tensor<64x64xi64, #blocked2> loc(#loc178)
    %228 = arith.subi %227, %cst_3 : tensor<64x64xi64, #blocked2> loc(#loc208)
    %229 = arith.cmpi sgt, %228, %cst_4 : tensor<64x64xi64, #blocked2> loc(#loc209)
    %230 = arith.select %229, %228, %cst_4 : tensor<64x64xi1, #blocked2>, tensor<64x64xi64, #blocked2> loc(#loc210)
    %231 = arith.cmpi slt, %230, %126 : tensor<64x64xi64, #blocked2> loc(#loc181)
    %232 = arith.select %231, %230, %126 : tensor<64x64xi1, #blocked2>, tensor<64x64xi64, #blocked2> loc(#loc211)

    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    // CHECK: llvm.load %{{.*}} {alignment = 2 : i64} : !llvm.ptr<3> -> vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<1xbf16>
    // CHECK-NEXT: llvm.mlir.constant(0.000000e+00 : bf16) : bf16
    // CHECK-NEXT: llvm.select %{{.*}}, %{{.*}}, %{{.*}} : i1, bf16
    %233 = triton_gpu.local_gather %134[%232], %216 : (<1x4096xbf16, #shared, #triton_gpu.shared_memory>, tensor<64x64xi64, #blocked2>, tensor<64x64xi1, #blocked2>) -> tensor<64x64xbf16, #blocked2> loc(#loc201)

    %234 = arith.extf %233 : tensor<64x64xbf16, #blocked2> to tensor<64x64xf32, #blocked2> loc(#loc212)
    %235 = arith.addf %219, %234 : tensor<64x64xf32, #blocked2> loc(#loc212)
    %236 = triton_gpu.convert_layout %235 {allocation.offset = 94208 : i32} : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #mma1> loc(#loc212)
    %237 = arith.addf %197, %236 : tensor<64x64xf32, #mma1> loc(#loc213)
    %238 = arith.subf %cst_12, %237 : tensor<64x64xf32, #mma1> loc(#loc214)
    %239 = math.exp %238 : tensor<64x64xf32, #mma1> loc(#loc215)
    %240 = arith.addf %239, %cst_11 : tensor<64x64xf32, #mma1> loc(#loc216)
    %241 = tt.extern_elementwise %237, %240 {libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<64x64xf32, #mma1>, tensor<64x64xf32, #mma1>) -> tensor<64x64xf32, #mma1> loc(#loc217)
    %242 = arith.mulf %241, %137 : tensor<64x64xf32, #mma1> loc(#loc184)
    %243 = tt.splat %176 : i32 -> tensor<1x64xi32, #mma1> loc(#loc218)
    %244 = arith.addi %243, %140 : tensor<1x64xi32, #mma1> loc(#loc218)
    %245 = tt.broadcast %244 : tensor<1x64xi32, #mma1> -> tensor<64x64xi32, #mma1> loc(#loc187)
    %246 = arith.cmpi sge, %141, %245 : tensor<64x64xi32, #mma1> loc(#loc187)
    %247 = arith.extsi %244 : tensor<1x64xi32, #mma1> to tensor<1x64xi64, #mma1> loc(#loc188)
    %248 = arith.cmpi slt, %247, %142 : tensor<1x64xi64, #mma1> loc(#loc188)
    %249 = arith.cmpi eq, %245, %141 : tensor<64x64xi32, #mma1> loc(#loc219)
    %250 = tt.broadcast %248 : tensor<1x64xi1, #mma1> -> tensor<64x64xi1, #mma1> loc(#loc220)
    %251 = arith.ori %250, %249 : tensor<64x64xi1, #mma1> loc(#loc220)
    %252 = arith.andi %246, %251 : tensor<64x64xi1, #mma1> loc(#loc221)
    %253 = arith.select %252, %242, %cst_12 : tensor<64x64xi1, #mma1>, tensor<64x64xf32, #mma1> loc(#loc222)
    %254 = triton_gpu.async_wait %183 {num = 0 : i32} loc(#loc189)
    %255 = triton_gpu.memdesc_subview %146[%189, %c0_i32, %c0_i32] : !tt.memdesc<2x64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc189)
    %256 = arith.truncf %253 : tensor<64x64xf32, #mma1> to tensor<64x64xbf16, #mma1> loc(#loc223)
    %257 = triton_gpu.convert_layout %256 : tensor<64x64xbf16, #mma1> -> tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> loc(#loc224)
    triton_nvidia_gpu.fence_async_shared {bCluster = false} loc(#loc224)
    %258 = triton_nvidia_gpu.warp_group_dot %257, %255, %177 {isAsync = true} : tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<64x128xf32, #mma> loc(#loc224)
    %259 = arith.addi %178, %c64_i64 : i64 loc(#loc138)
    %260 = arith.addi %179, %c64_i64 : i64 loc(#loc139)
    %261 = arith.addi %180, %c1_i32 : i32 loc(#loc94)
    %262 = arith.cmpi slt, %261, %c2_i32 : i32 loc(#loc94)
    %263 = arith.select %262, %261, %c0_i32 : i32 loc(#loc94)
    %264 = tt.splat %259 : i64 -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc161)
    %265 = arith.addi %264, %66 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc161)
    %266 = tt.expand_dims %265 {axis = 0 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi64, #blocked1> loc(#loc161)
    %267 = arith.muli %266, %99 : tensor<1x64xi64, #blocked1> loc(#loc161)
    %268 = tt.broadcast %267 : tensor<1x64xi64, #blocked1> -> tensor<128x64xi64, #blocked1> loc(#loc161)
    %269 = arith.addi %98, %268 : tensor<128x64xi64, #blocked1> loc(#loc161)
    %270 = tt.addptr %96, %269 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi64, #blocked1> loc(#loc161)
    %271 = arith.cmpi sge, %266, %cst_0 : tensor<1x64xi64, #blocked1> loc(#loc161)
    %272 = arith.cmpi slt, %266, %100 : tensor<1x64xi64, #blocked1> loc(#loc161)
    %273 = arith.andi %271, %272 : tensor<1x64xi1, #blocked1> loc(#loc161)
    %274 = tt.broadcast %273 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc161)
    %275 = triton_gpu.memdesc_subview %145[%263, %c0_i32, %c0_i32] : !tt.memdesc<2x128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc161)
    %276 = tt.splat %186 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc94)
    %277 = arith.andi %276, %274 : tensor<128x64xi1, #blocked1> loc(#loc94)
    %278 = triton_gpu.async_copy_global_to_local %270, %275 mask %277 other %cst_7 : tensor<128x64x!tt.ptr<bf16>, #blocked1> -> <128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc161)
    %279 = triton_gpu.async_commit_group %278 loc(#loc161)
    %280 = tt.splat %260 : i64 -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc189)
    %281 = arith.addi %280, %65 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc189)
    %282 = tt.expand_dims %281 {axis = 1 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked> loc(#loc189)
    %283 = arith.muli %282, %144 : tensor<64x1xi64, #blocked> loc(#loc189)
    %284 = tt.broadcast %283 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked> loc(#loc189)
    %285 = arith.addi %284, %77 : tensor<64x128xi64, #blocked> loc(#loc189)
    %286 = tt.addptr %143, %285 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi64, #blocked> loc(#loc189)
    %287 = arith.cmpi sge, %282, %cst : tensor<64x1xi64, #blocked> loc(#loc189)
    %288 = arith.cmpi slt, %282, %81 : tensor<64x1xi64, #blocked> loc(#loc189)
    %289 = arith.andi %287, %288 : tensor<64x1xi1, #blocked> loc(#loc189)
    %290 = tt.broadcast %289 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc189)
    %291 = triton_gpu.memdesc_subview %146[%263, %c0_i32, %c0_i32] : !tt.memdesc<2x64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc189)
    %292 = tt.splat %186 : i1 -> tensor<64x128xi1, #blocked> loc(#loc94)
    %293 = arith.andi %292, %290 : tensor<64x128xi1, #blocked> loc(#loc94)
    %294 = triton_gpu.async_copy_global_to_local %286, %291 mask %293 other %cst_6 : tensor<64x128x!tt.ptr<bf16>, #blocked> -> <64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc189)
    %295 = triton_gpu.async_commit_group %294 loc(#loc189)
    %296 = arith.addi %176, %c64_i32 : i32 loc(#loc94)
    cf.br ^bb7(%296, %258, %259, %260, %263, %189, %279, %295 : i32, tensor<64x128xf32, #mma>, i64, i64, i32, i32, !triton_gpu.async.token, !triton_gpu.async.token) loc(#loc94)
  ^bb9:  // pred: ^bb7
    %297 = triton_nvidia_gpu.warp_group_dot_wait %177 {pendings = 0 : i32} : tensor<64x128xf32, #mma> loc(#loc94)
    %298 = triton_gpu.async_wait  {num = 0 : i32} loc(#loc94)
    triton_gpu.local_dealloc %145 : !tt.memdesc<2x128x64xbf16, #shared1, #triton_gpu.shared_memory, mutable> loc(#loc94)
    triton_gpu.local_dealloc %146 : !tt.memdesc<2x64x128xbf16, #shared, #triton_gpu.shared_memory, mutable> loc(#loc94)
    cf.cond_br %91, ^bb10, ^bb11 loc(#loc140)
  ^bb10:  // pred: ^bb9
    %299 = arith.subi %33, %90 : i64 loc(#loc141)
    %300 = arith.trunci %299 : i64 to i32 loc(#loc142)
    %301 = arith.extsi %300 : i32 to i64 loc(#loc143)
    %302 = arith.addi %178, %301 : i64 loc(#loc143)
    %303 = arith.addi %179, %301 : i64 loc(#loc144)
    %304 = arith.subi %8, %10 : i32 loc(#loc225)
    %305 = tt.splat %304 : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc226)
    %306 = arith.cmpi slt, %18, %305 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc226)
    %307 = tt.splat %302 : i64 -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc227)
    %308 = arith.addi %307, %66 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> loc(#loc227)
    %309 = tt.expand_dims %308 {axis = 0 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi64, #blocked1> loc(#loc227)
    %310 = arith.muli %309, %99 : tensor<1x64xi64, #blocked1> loc(#loc227)
    %311 = tt.broadcast %310 : tensor<1x64xi64, #blocked1> -> tensor<128x64xi64, #blocked1> loc(#loc227)
    %312 = arith.addi %98, %311 : tensor<128x64xi64, #blocked1> loc(#loc227)
    %313 = tt.addptr %96, %312 : tensor<128x64x!tt.ptr<bf16>, #blocked1>, tensor<128x64xi64, #blocked1> loc(#loc227)
    %314 = arith.cmpi sge, %309, %cst_0 : tensor<1x64xi64, #blocked1> loc(#loc227)
    %315 = arith.cmpi slt, %309, %100 : tensor<1x64xi64, #blocked1> loc(#loc227)
    %316 = arith.andi %314, %315 : tensor<1x64xi1, #blocked1> loc(#loc227)
    %317 = tt.broadcast %316 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc227)
    %318 = tt.load %313, %317, %cst_7 : tensor<128x64x!tt.ptr<bf16>, #blocked1> loc(#loc227)
    %319 = triton_gpu.local_alloc %318 {allocation.offset = 0 : i32} : (tensor<128x64xbf16, #blocked1>) -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> loc(#loc227)
    triton_nvidia_gpu.fence_async_shared {bCluster = false} loc(#loc228)
    %320 = triton_nvidia_gpu.warp_group_dot %86, %319, %cst_12 : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma1> loc(#loc228)
    %321 = arith.mulf %320, %101 : tensor<64x64xf32, #mma1> loc(#loc229)
    %322 = tt.addptr %53, %22 : tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc230)
    %323 = tt.load %322, %306 : tensor<64x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc231)
    %324 = tt.expand_dims %323 {axis = 0 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi64, #blocked2> loc(#loc232)
    %325 = tt.broadcast %324 : tensor<1x64xi64, #blocked2> -> tensor<64x64xi64, #blocked2> loc(#loc233)
    %326 = arith.subi %103, %325 : tensor<64x64xi64, #blocked2> loc(#loc233)
    %327 = arith.sitofp %326 : tensor<64x64xi64, #blocked2> to tensor<64x64xf32, #blocked2> loc(#loc234)
    %328 = arith.addf %327, %104 : tensor<64x64xf32, #blocked2> loc(#loc234)
    %329 = arith.cmpf ogt, %328, %cst_1 : tensor<64x64xf32, #blocked2> loc(#loc235)
    %330 = arith.select %329, %328, %cst_1 : tensor<64x64xi1, #blocked2>, tensor<64x64xf32, #blocked2> loc(#loc236)
    %331 = arith.mulf %330, %107 : tensor<64x64xf32, #blocked2> loc(#loc237)
    %332 = math.sqrt %331 : tensor<64x64xf32, #blocked2> loc(#loc238)
    %333 = arith.mulf %332, %109 : tensor<64x64xf32, #blocked2> loc(#loc239)
    %334 = arith.fptosi %333 : tensor<64x64xf32, #blocked2> to tensor<64x64xi32, #blocked2> loc(#loc240)
    %335 = arith.cmpi sgt, %334, %cst_2 : tensor<64x64xi32, #blocked2> loc(#loc241)
    %336 = arith.select %335, %334, %cst_2 : tensor<64x64xi1, #blocked2>, tensor<64x64xi32, #blocked2> loc(#loc242)
    %337 = arith.cmpi slt, %336, %110 : tensor<64x64xi32, #blocked2> loc(#loc243)
    %338 = arith.select %337, %336, %110 : tensor<64x64xi1, #blocked2>, tensor<64x64xi32, #blocked2> loc(#loc244)
    %339 = tt.expand_dims %306 {axis = 0 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi1, #blocked2> loc(#loc245)
    %340 = tt.broadcast %339 : tensor<1x64xi1, #blocked2> -> tensor<64x64xi1, #blocked2> loc(#loc246)
    %341 = arith.andi %112, %340 : tensor<64x64xi1, #blocked2> loc(#loc246)
    %342 = tt.addptr %113, %338 : tensor<64x64x!tt.ptr<bf16>, #blocked2>, tensor<64x64xi32, #blocked2> loc(#loc247)
    %343 = tt.load %342, %341 : tensor<64x64x!tt.ptr<bf16>, #blocked2> loc(#loc248)
    %344 = arith.extf %343 : tensor<64x64xbf16, #blocked2> to tensor<64x64xf32, #blocked2> loc(#loc249)
    %345 = arith.addf %344, %cst_9 : tensor<64x64xf32, #blocked2> loc(#loc249)
    %346 = arith.extsi %26 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc250)
    %347 = arith.cmpi slt, %346, %115 : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc250)
    %348 = arith.select %347, %346, %115 : tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>, tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> loc(#loc251)
    %349 = tt.expand_dims %348 {axis = 0 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi64, #blocked2> loc(#loc252)
    %350 = tt.broadcast %349 : tensor<1x64xi64, #blocked2> -> tensor<64x64xi64, #blocked2> loc(#loc253)
    %351 = arith.subi %350, %120 : tensor<64x64xi64, #blocked2> loc(#loc253)
    %352 = arith.addi %351, %122 : tensor<64x64xi64, #blocked2> loc(#loc254)
    %353 = arith.subi %352, %cst_3 : tensor<64x64xi64, #blocked2> loc(#loc255)
    %354 = arith.cmpi sgt, %353, %cst_4 : tensor<64x64xi64, #blocked2> loc(#loc256)
    %355 = arith.select %354, %353, %cst_4 : tensor<64x64xi1, #blocked2>, tensor<64x64xi64, #blocked2> loc(#loc257)
    %356 = arith.cmpi slt, %355, %126 : tensor<64x64xi64, #blocked2> loc(#loc258)
    %357 = arith.select %356, %355, %126 : tensor<64x64xi1, #blocked2>, tensor<64x64xi64, #blocked2> loc(#loc259)
    %358 = tt.addptr %127, %357 : tensor<64x64x!tt.ptr<bf16>, #blocked2>, tensor<64x64xi64, #blocked2> loc(#loc260)
    %359 = tt.load %358, %341 : tensor<64x64x!tt.ptr<bf16>, #blocked2> loc(#loc261)
    %360 = arith.extf %359 : tensor<64x64xbf16, #blocked2> to tensor<64x64xf32, #blocked2> loc(#loc262)
    %361 = arith.addf %345, %360 : tensor<64x64xf32, #blocked2> loc(#loc262)
    %362 = triton_gpu.convert_layout %361 {allocation.offset = 0 : i32} : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #mma1> loc(#loc262)
    %363 = arith.addf %321, %362 : tensor<64x64xf32, #mma1> loc(#loc263)
    %364 = arith.subf %cst_12, %363 : tensor<64x64xf32, #mma1> loc(#loc264)
    %365 = math.exp %364 : tensor<64x64xf32, #mma1> loc(#loc265)
    %366 = arith.addf %365, %cst_11 : tensor<64x64xf32, #mma1> loc(#loc266)
    %367 = tt.extern_elementwise %363, %366 {libname = "", libpath = "", pure = true, symbol = "__nv_fast_fdividef"} : (tensor<64x64xf32, #mma1>, tensor<64x64xf32, #mma1>) -> tensor<64x64xf32, #mma1> loc(#loc267)
    %368 = arith.mulf %367, %137 : tensor<64x64xf32, #mma1> loc(#loc268)
    %369 = tt.splat %10 : i32 -> tensor<1x64xi32, #mma1> loc(#loc269)
    %370 = arith.addi %369, %140 : tensor<1x64xi32, #mma1> loc(#loc269)
    %371 = tt.broadcast %370 : tensor<1x64xi32, #mma1> -> tensor<64x64xi32, #mma1> loc(#loc270)
    %372 = arith.cmpi sge, %141, %371 : tensor<64x64xi32, #mma1> loc(#loc270)
    %373 = arith.extsi %370 : tensor<1x64xi32, #mma1> to tensor<1x64xi64, #mma1> loc(#loc271)
    %374 = arith.cmpi slt, %373, %142 : tensor<1x64xi64, #mma1> loc(#loc271)
    %375 = arith.cmpi eq, %371, %141 : tensor<64x64xi32, #mma1> loc(#loc272)
    %376 = tt.broadcast %374 : tensor<1x64xi1, #mma1> -> tensor<64x64xi1, #mma1> loc(#loc273)
    %377 = arith.ori %376, %375 : tensor<64x64xi1, #mma1> loc(#loc273)
    %378 = arith.andi %372, %377 : tensor<64x64xi1, #mma1> loc(#loc274)
    %379 = arith.select %378, %368, %cst_12 : tensor<64x64xi1, #mma1>, tensor<64x64xf32, #mma1> loc(#loc275)
    %380 = tt.splat %303 : i64 -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc276)
    %381 = arith.addi %380, %65 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> loc(#loc276)
    %382 = tt.expand_dims %381 {axis = 1 : i32} : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked> loc(#loc276)
    %383 = arith.muli %382, %144 : tensor<64x1xi64, #blocked> loc(#loc276)
    %384 = tt.broadcast %383 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked> loc(#loc276)
    %385 = arith.addi %384, %77 : tensor<64x128xi64, #blocked> loc(#loc276)
    %386 = tt.addptr %143, %385 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi64, #blocked> loc(#loc276)
    %387 = arith.cmpi sge, %382, %cst : tensor<64x1xi64, #blocked> loc(#loc276)
    %388 = arith.cmpi slt, %382, %81 : tensor<64x1xi64, #blocked> loc(#loc276)
    %389 = arith.andi %387, %388 : tensor<64x1xi1, #blocked> loc(#loc276)
    %390 = tt.broadcast %389 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc276)
    %391 = tt.load %386, %390, %cst_6 : tensor<64x128x!tt.ptr<bf16>, #blocked> loc(#loc276)
    %392 = triton_gpu.local_alloc %391 {allocation.offset = 0 : i32} : (tensor<64x128xbf16, #blocked>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> loc(#loc276)
    %393 = arith.truncf %379 : tensor<64x64xf32, #mma1> to tensor<64x64xbf16, #mma1> loc(#loc277)
    %394 = triton_gpu.convert_layout %393 : tensor<64x64xbf16, #mma1> -> tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> loc(#loc278)
    triton_nvidia_gpu.fence_async_shared {bCluster = false} loc(#loc278)
    %395 = triton_nvidia_gpu.warp_group_dot %394, %392, %297 : tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma1}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma> loc(#loc278)
    cf.br ^bb12(%395 : tensor<64x128xf32, #mma>) loc(#loc140)
  ^bb11:  // pred: ^bb9
    cf.br ^bb12(%297 : tensor<64x128xf32, #mma>) loc(#loc140)
  ^bb12(%396: tensor<64x128xf32, #mma> loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":569:11)):  // 2 preds: ^bb10, ^bb11
    cf.br ^bb13 loc(#loc140)
  ^bb13:  // pred: ^bb12
    %397 = tt.expand_dims %25 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc149)
    %398 = tt.splat %4 : i64 -> tensor<64x1xi64, #blocked> loc(#loc150)
    %399 = arith.extsi %397 : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked> loc(#loc150)
    %400 = arith.addi %398, %399 : tensor<64x1xi64, #blocked> loc(#loc150)
    %401 = arith.extsi %arg18 : i32 to i64 loc(#loc151)
    %402 = tt.splat %401 : i64 -> tensor<64x1xi64, #blocked> loc(#loc151)
    %403 = arith.muli %400, %402 : tensor<64x1xi64, #blocked> loc(#loc151)
    %404 = arith.muli %2, %arg19 : i32 loc(#loc152)
    %405 = arith.extsi %404 : i32 to i64 loc(#loc153)
    %406 = tt.splat %405 : i64 -> tensor<64x1xi64, #blocked> loc(#loc153)
    %407 = arith.addi %403, %406 : tensor<64x1xi64, #blocked> loc(#loc153)
    %408 = tt.expand_dims %72 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked> loc(#loc154)
    %409 = tt.broadcast %407 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked> loc(#loc155)
    %410 = arith.extsi %408 : tensor<1x128xi32, #blocked> to tensor<1x128xi64, #blocked> loc(#loc155)
    %411 = tt.broadcast %410 : tensor<1x128xi64, #blocked> -> tensor<64x128xi64, #blocked> loc(#loc155)
    %412 = arith.addi %409, %411 : tensor<64x128xi64, #blocked> loc(#loc155)
    %413 = tt.splat %arg8 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked> loc(#loc156)
    %414 = tt.addptr %413, %412 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi64, #blocked> loc(#loc156)
    %415 = tt.expand_dims %47 {axis = 1 : i32} : tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi1, #blocked> loc(#loc157)
    %416 = tt.broadcast %415 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked> loc(#loc158)
    %417 = arith.truncf %396 : tensor<64x128xf32, #mma> to tensor<64x128xbf16, #mma> loc(#loc158)
    %418 = triton_gpu.convert_layout %417 {allocation.offset = 0 : i32} : tensor<64x128xbf16, #mma> -> tensor<64x128xbf16, #blocked> loc(#loc158)
    tt.store %414, %418, %416 : tensor<64x128x!tt.ptr<bf16>, #blocked> loc(#loc158)
    tt.return loc(#loc159)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":423:27)
#loc3 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":424:22)
#loc4 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":425:21)
#loc5 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":426:38)
#loc6 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":426:24)
#loc7 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":427:44)
#loc8 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":427:22)
#loc9 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":428:25)
#loc10 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":428:39)
#loc11 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":435:32)
#loc12 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":435:37)
#loc13 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":436:18)
#loc14 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":437:8)
#loc15 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":439:42)
#loc16 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":439:28)
#loc17 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":442:36)
#loc18 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":442:23)
#loc19 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":455:29)
#loc20 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":455:21)
#loc21 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":455:53)
#loc22 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":455:41)
#loc23 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":460:12)
#loc24 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":463:25)
#loc25 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":463:17)
#loc26 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":463:49)
#loc27 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":463:37)
#loc28 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":471:25)
#loc29 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":471:17)
#loc30 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":471:49)
#loc31 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":471:37)
#loc32 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":478:22)
#loc33 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":480:33)
#loc34 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":480:25)
#loc35 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":480:45)
#loc36 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":481:45)
#loc37 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":483:39)
#loc38 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":483:27)
#loc39 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":94:12)
#loc40 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":95:12)
#loc41 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":96:12)
#loc42 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":97:12)
#loc43 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":98:12)
#loc44 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":99:12)
#loc45 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":100:12)
#loc46 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":496:16)
#loc47 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":501:29)
#loc48 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":501:51)
#loc49 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":501:41)
#loc50 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":501:57)
#loc51 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":501:67)
#loc52 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":502:21)
#loc54 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":503:42)
#loc55 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":503:29)
#loc56 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":505:29)
#loc57 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":254:16)
#loc58 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":562:12)
#loc59 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":255:47)
#loc60 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":263:22)
#loc61 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":263:33)
#loc62 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":264:22)
#loc63 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":266:29)
#loc64 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":266:23)
#loc65 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":271:29)
#loc66 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":271:23)
#loc67 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":274:31)
#loc68 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":277:28)
#loc69 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":277:41)
#loc70 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":276:21)
#loc71 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":288:37)
#loc72 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":290:24)
#loc73 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":305:62)
#loc74 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":305:51)
#loc75 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":305:73)
#loc76 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":308:37)
#loc77 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":308:51)
#loc78 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":308:33)
#loc79 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":315:21)
#loc80 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":170:12)
#loc81 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":171:12)
#loc82 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":172:12)
#loc83 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":173:12)
#loc84 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":174:12)
#loc85 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":175:12)
#loc86 = loc("/home/plotfi/.triton/override/e12d7953749865175c74b535c0175303bab899d8d164efbce73d029e789f06ef/_ragged_hstu_attn_fwd.ttgir":176:12)
#loc87 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":327:56)
#loc88 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":327:50)
#loc89 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":330:30)
#loc90 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":330:60)
#loc91 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":330:43)
#loc92 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":336:45)
#loc93 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":349:16)
#loc94 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":517:36)
#loc95 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":255:19)
#loc104 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":252:32)
#loc105 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":252:22)
#loc106 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":260:43)
#loc107 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":260:31)
#loc108 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":263:38)
#loc109 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":265:31)
#loc110 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":265:41)
#loc111 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":270:29)
#loc112 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":272:23)
#loc113 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":273:31)
#loc114 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":273:38)
#loc115 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":277:48)
#loc116 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":279:36)
#loc117 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":285:38)
#loc118 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":293:37)
#loc119 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":295:24)
#loc120 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":305:40)
#loc121 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":305:87)
#loc122 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":306:51)
#loc123 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":306:66)
#loc124 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":310:20)
#loc125 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":318:36)
#loc126 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":319:18)
#loc127 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":327:42)
#loc128 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":327:41)
#loc129 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":327:34)
#loc130 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":327:28)
#loc131 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":330:53)
#loc132 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":337:49)
#loc133 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":337:20)
#loc134 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":336:16)
#loc135 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":345:40)
#loc136 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":350:19)
#loc137 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":351:24)
#loc138 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":564:46)
#loc139 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":565:46)
#loc141 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":572:34)
#loc142 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":572:46)
#loc143 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":573:50)
#loc144 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":574:50)
#loc145 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":624:20)
#loc146 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":274:48)
#loc147 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":276:16)
#loc148 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":315:16)
#loc149 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":646:32)
#loc150 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":646:25)
#loc151 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":646:44)
#loc152 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":647:22)
#loc153 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":647:14)
#loc154 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":648:23)
#loc155 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":648:14)
#loc156 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":650:25)
#loc157 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":651:56)
#loc158 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":651:27)
#loc159 = loc("/home/plotfi/opt/dev/TRITON-SMEM-H2/triton/ttgir-override-testbed/ragged_hstu_test_bed/hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py":629:4)
#loc160 = loc(fused[#loc48, #loc49])
#loc161 = loc(callsite(#loc57 at #loc58))
#loc162 = loc(callsite(#loc59 at #loc58))
#loc163 = loc(callsite(#loc60 at #loc58))
#loc164 = loc(callsite(#loc61 at #loc58))
#loc165 = loc(callsite(#loc62 at #loc58))
#loc166 = loc(callsite(#loc63 at #loc58))
#loc167 = loc(callsite(#loc64 at #loc58))
#loc168 = loc(callsite(#loc65 at #loc58))
#loc169 = loc(callsite(#loc66 at #loc58))
#loc170 = loc(callsite(#loc67 at #loc58))
#loc171 = loc(callsite(#loc68 at #loc58))
#loc172 = loc(callsite(#loc69 at #loc58))
#loc173 = loc(callsite(#loc70 at #loc58))
#loc174 = loc(callsite(#loc71 at #loc58))
#loc175 = loc(callsite(#loc72 at #loc58))
#loc176 = loc(callsite(#loc73 at #loc58))
#loc177 = loc(callsite(#loc74 at #loc58))
#loc178 = loc(callsite(#loc75 at #loc58))
#loc179 = loc(callsite(#loc76 at #loc58))
#loc180 = loc(callsite(#loc77 at #loc58))
#loc181 = loc(callsite(#loc78 at #loc58))
#loc182 = loc(callsite(#loc79 at #loc58))
#loc183 = loc(callsite(#loc87 at #loc58))
#loc184 = loc(callsite(#loc88 at #loc58))
#loc185 = loc(callsite(#loc89 at #loc58))
#loc186 = loc(callsite(#loc90 at #loc58))
#loc187 = loc(callsite(#loc91 at #loc58))
#loc188 = loc(callsite(#loc92 at #loc58))
#loc189 = loc(callsite(#loc93 at #loc58))
#loc190 = loc(callsite(#loc95 at #loc58))
#loc191 = loc(callsite(#loc104 at #loc58))
#loc192 = loc(callsite(#loc105 at #loc58))
#loc193 = loc(callsite(#loc106 at #loc58))
#loc194 = loc(callsite(#loc107 at #loc58))
#loc195 = loc(callsite(#loc108 at #loc58))
#loc196 = loc(callsite(#loc109 at #loc58))
#loc197 = loc(callsite(#loc110 at #loc58))
#loc198 = loc(callsite(#loc111 at #loc58))
#loc199 = loc(callsite(#loc112 at #loc58))
#loc200 = loc(callsite(#loc113 at #loc58))
#loc201 = loc(callsite(#loc114 at #loc58))
#loc202 = loc(callsite(#loc115 at #loc58))
#loc203 = loc(callsite(#loc116 at #loc58))
#loc204 = loc(callsite(#loc117 at #loc58))
#loc205 = loc(callsite(#loc118 at #loc58))
#loc206 = loc(callsite(#loc119 at #loc58))
#loc207 = loc(callsite(#loc120 at #loc58))
#loc208 = loc(callsite(#loc121 at #loc58))
#loc209 = loc(callsite(#loc122 at #loc58))
#loc210 = loc(callsite(#loc123 at #loc58))
#loc211 = loc(callsite(#loc124 at #loc58))
#loc212 = loc(callsite(#loc125 at #loc58))
#loc213 = loc(callsite(#loc126 at #loc58))
#loc214 = loc(callsite(#loc127 at #loc58))
#loc215 = loc(callsite(#loc128 at #loc58))
#loc216 = loc(callsite(#loc129 at #loc58))
#loc217 = loc(callsite(#loc130 at #loc58))
#loc218 = loc(callsite(#loc131 at #loc58))
#loc219 = loc(callsite(#loc132 at #loc58))
#loc220 = loc(callsite(#loc133 at #loc58))
#loc221 = loc(callsite(#loc134 at #loc58))
#loc222 = loc(callsite(#loc135 at #loc58))
#loc223 = loc(callsite(#loc136 at #loc58))
#loc224 = loc(callsite(#loc137 at #loc58))
#loc225 = loc(callsite(#loc104 at #loc145))
#loc226 = loc(callsite(#loc105 at #loc145))
#loc227 = loc(callsite(#loc57 at #loc145))
#loc228 = loc(callsite(#loc95 at #loc145))
#loc229 = loc(callsite(#loc59 at #loc145))
#loc230 = loc(callsite(#loc106 at #loc145))
#loc231 = loc(callsite(#loc107 at #loc145))
#loc232 = loc(callsite(#loc108 at #loc145))
#loc233 = loc(callsite(#loc61 at #loc145))
#loc234 = loc(callsite(#loc62 at #loc145))
#loc235 = loc(callsite(#loc109 at #loc145))
#loc236 = loc(callsite(#loc110 at #loc145))
#loc237 = loc(callsite(#loc64 at #loc145))
#loc238 = loc(callsite(#loc111 at #loc145))
#loc239 = loc(callsite(#loc66 at #loc145))
#loc240 = loc(callsite(#loc112 at #loc145))
#loc241 = loc(callsite(#loc113 at #loc145))
#loc242 = loc(callsite(#loc114 at #loc145))
#loc243 = loc(callsite(#loc67 at #loc145))
#loc244 = loc(callsite(#loc146 at #loc145))
#loc245 = loc(callsite(#loc115 at #loc145))
#loc246 = loc(callsite(#loc69 at #loc145))
#loc247 = loc(callsite(#loc70 at #loc145))
#loc248 = loc(callsite(#loc147 at #loc145))
#loc249 = loc(callsite(#loc116 at #loc145))
#loc250 = loc(callsite(#loc71 at #loc145))
#loc251 = loc(callsite(#loc72 at #loc145))
#loc252 = loc(callsite(#loc120 at #loc145))
#loc253 = loc(callsite(#loc74 at #loc145))
#loc254 = loc(callsite(#loc75 at #loc145))
#loc255 = loc(callsite(#loc121 at #loc145))
#loc256 = loc(callsite(#loc122 at #loc145))
#loc257 = loc(callsite(#loc123 at #loc145))
#loc258 = loc(callsite(#loc78 at #loc145))
#loc259 = loc(callsite(#loc124 at #loc145))
#loc260 = loc(callsite(#loc79 at #loc145))
#loc261 = loc(callsite(#loc148 at #loc145))
#loc262 = loc(callsite(#loc125 at #loc145))
#loc263 = loc(callsite(#loc126 at #loc145))
#loc264 = loc(callsite(#loc127 at #loc145))
#loc265 = loc(callsite(#loc128 at #loc145))
#loc266 = loc(callsite(#loc129 at #loc145))
#loc267 = loc(callsite(#loc130 at #loc145))
#loc268 = loc(callsite(#loc88 at #loc145))
#loc269 = loc(callsite(#loc131 at #loc145))
#loc270 = loc(callsite(#loc91 at #loc145))
#loc271 = loc(callsite(#loc92 at #loc145))
#loc272 = loc(callsite(#loc132 at #loc145))
#loc273 = loc(callsite(#loc133 at #loc145))
#loc274 = loc(callsite(#loc134 at #loc145))
#loc275 = loc(callsite(#loc135 at #loc145))
#loc276 = loc(callsite(#loc93 at #loc145))
#loc277 = loc(callsite(#loc136 at #loc145))
#loc278 = loc(callsite(#loc137 at #loc145))

