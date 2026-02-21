# SG14 Demonstrators

Performance demonstrators presented at the SG14 (Low Latency/Games/Embedded/Financial Trading) session, February 2026.

These implementations are **conformance-verified** against the §4 canonical expression specification in P4016R0. Each produces identical golden hex values for the same `(N, L, seed)` inputs.

## Files

| File | Platform | Godbolt | Build flags |
|------|----------|---------|-------------|
| `x86_avx2.cpp` | x86-64 (SSE2/AVX2/AVX512) | [godbolt.org/z/jbYqf1Eez](https://godbolt.org/z/jbYqf1Eez) | `-O3 -std=c++20 -ffp-contract=off -fno-fast-math` |
| `arm_neon.cpp` | AArch64 (NEON) | [godbolt.org/z/v369Mbnvh](https://godbolt.org/z/v369Mbnvh) | `-O3 -std=c++20 -march=armv8-a -ffp-contract=off -fno-fast-math` |
| `cuda_reduce.cu` | CUDA (NVCC) | [godbolt.org/z/x58GzE73q](https://godbolt.org/z/x58GzE73q) | `-O3 -std=c++17 --fmad=false` |

## Golden Reference Values

N = 1,000,000 doubles, seed = `0x243F6A8885A308D3`:

| Configuration | Hex |
|---------------|-----|
| L=16 (NARROW) | `0x40618f71f6379380` |
| L=128 (WIDE)  | `0x40618f71f6379397` |

All three platforms produce identical results.

## What These Demonstrate Beyond the Paper's Appendix K

1. **Performance benchmarks** — throughput (GB/s) comparisons against `std::accumulate`, `std::reduce` (all execution policies), and NVIDIA CUB `DeviceReduce::Sum`.
2. **Robustness test suite** — length-sweep reproducibility across 40+ input sizes (0 to 12288+) on both base and hostile (±1e16 cancellation at prime strides 7, 11) datasets.
3. **Cross-ISA identity** — three architectures, identical hex output, verifiable with a single click on Compiler Explorer.

## Performance Summary

### x86 (Compiler Explorer, AVX2, N=1M)

| Variant | Throughput | vs accumulate |
|---------|-----------|---------------|
| `std::accumulate` | 5.4 GB/s | baseline |
| `std::reduce` | 21.4 GB/s | +297% |
| Canonical (L=16, single-threaded) | 26.5 GB/s | +391% |

### CUDA (vs CUB)

The canonical GPU implementation runs approximately 20% slower than CUB `DeviceReduce::Sum` (~80% of CUB throughput). This is the cost of determinism — CUB is free to reassociate across warps and blocks, while the canonical kernel must preserve the specified expression tree.

## §4 Conformance Verification

All three demonstrators have been reviewed against the P4016R0 §4 normative specification:

- **§4.2.3** Canonical iterative pairwise tree (binary counter stack with "older on left" combine order)
- **§4.3.1** Lane partitioning by `i mod L`
- **§4.3.2** Fixed-length leaves with `K = ceil(N/L)`
- **§4.2.2** Absent-operand propagation (count-aware combine / NaN sentinel on CUDA)
- **§4.4** Two-stage reduction (per-lane tree + cross-lane tree)
- **§4.5** Init placement: `op(init, R)` applied once after tree completion

The 8-block unroll optimisation (pushing at binary counter level 3) preserves the canonical tree shape — the parenthesisation `((b0+b1)+(b2+b3))+((b4+b5)+(b6+b7))` is identical to what the shift-reduce algorithm produces for 8 consecutive blocks.
