# SG14 Performance Demonstrators

Production-quality implementations of the §4 canonical reduction, presented to
SG14 (Low Latency / Games / Embedded / Financial Trading).  These complement the
Appendix K reference demonstrators by adding:

- **Throughput benchmarks** against `std::accumulate`, `std::reduce`, and CUB
- **Robustness stress tests** — length sweep + hostile cancellation (±1e16 at prime strides 7/11 + cluster)
- **Cross-ISA golden-value verification** — all three produce identical hex for the same seed/N/L

## Files

| File | Platform | Godbolt | Notes |
|------|----------|---------|-------|
| `x86_avx2.cpp` | x86-64 | [link](https://godbolt.org/z/jGThKbrqq) | SSE2/AVX2/AVX512 multiversion; 26.5 GB/s single-thread |
| `arm_neon.cpp` | AArch64 | [link](https://godbolt.org/z/5n6rWGbq5) | NEON 8-block unroll; cross-platform golden match |
| `cuda.cu` | CUDA | [link](https://godbolt.org/z/x58GzE73q) | 3-phase kernel; ~80% of CUB throughput |

## Golden values

All demonstrators use SEED = `0x243F6A8885A308D3`, N = 1,000,000 doubles:

- **L=16 (NARROW):** `0x40618f71f6379380`
- **L=128 (WIDE):** `0x40618f71f6379397`

These match the Appendix K reference implementations and each other.

## §4 Conformance

All three demonstrators have been verified against §4:

- §4.3.1 Lane partitioning (`i mod L`) ✓
- §4.2.3 Canonical iterative pairwise tree (binary counter stack) ✓
- §4.2.2 Absent operand propagation (count-aware combine) ✓
- §4.4 Two-stage reduction (per-lane + cross-lane) ✓
- §4.5 Init placement (`op(init, R)`) ✓

## Running on Compiler Explorer

### Getting a clean session

Compiler Explorer remembers your last session.  If it opens in CMake mode or
with stale settings, **open godbolt.org in an incognito / private browser
window** — this gives you a fresh single-file session.

### Compiler and flags

**x86:**
```
Compiler: x86-64 clang (trunk)  — or GCC
Flags:    -O3 -std=c++20 -ffp-contract=off -fno-fast-math
```

**ARM / AArch64:**
```
Compiler: ARM64 GCC (trunk)  — or AArch64 clang
Flags:    -O3 -std=c++20 -march=armv8-a -ffp-contract=off -fno-fast-math
```

**CUDA:**
```
Compiler: NVCC (trunk)
Flags:    -O3 -std=c++17 --fmad=false
```

### CMake mode troubleshooting

If the Godbolt link opens in CMake mode, you may see a link error like
`-lfmtd: not found`.  The code does not use fmt — this is a CE environment
artefact.  Fix: open an incognito window, paste the source, select the compiler,
and enter the flags above in the "Compiler options" box.

## Relationship to Appendix K

The paper's Appendix K contains seven Godbolt demonstrators that serve as
**minimal reference implementations** — they prove the spec is implementable.

These SG14 demonstrators are **performance-oriented variants** of the same
algorithm, adding benchmarks, hostile-data stress tests, and ISA multiversion
dispatch.  They produce identical results to the Appendix K links for the same
seed, N, and L.
