// ============================================================================
// Deterministic Parallel Reduction — TRUE MT + SIMD Multiversion
// EXTENDED CORRECTNESS EDITION
//
// Reference implementation for WG21 proposal P4016R1
// "Deterministic Parallel Reduction: Canonical Compute Sequence"
//
// Same algorithm as the base demonstrator. Extended with:
//   - Hostile input sequences (near-cancellation, large dynamic range,
//     alternating sign, Kahan-style ill-conditioned sums, subnormals)
//   - Sweep over ~30 awkward sizes: primes, N<L, boundary±1, 8-block
//     unroll boundaries, chunk-partition stress cases
//   - Every size × {T=2, T=3, T=4} × {L=16, L=128} verified bitwise
//     against single-threaded reference
//   - Original N=1M golden-value verification preserved
//   - Performance comparison: std::accumulate, std::reduce,
//     std::reduce(par_unseq) [when TBB available], deterministic ST/MT
//
// Canonical semantics:
//   1) Partition into blocks of L lanes
//   2) Binary-counter stack replay (pairwise tree, carry last if odd)
//   3) Cross-lane pairwise fold
//
// MT design:
//   - Partition blocks at power-of-2 aligned boundaries
//   - Each thread performs local stack replay (parallel)
//   - Merge thread states deterministically (serial, left-to-right)
//   - Result is bitwise identical to single-threaded for any T
//
// ============================================================================
// BUILD INSTRUCTIONS
//
// Godbolt (Clang trunk, CMake mode):
//   Compiler:  Clang (trunk)
//   CXX flags: -O3 -std=c++20 -ffp-contract=off -fno-fast-math
//              -DFORCE_AVX2=1 -pthread -DNO_PAR_POLICIES=1 -mavx2
//
// Local build (GCC/Clang with TBB for par_unseq comparison):
//   g++ -O3 -std=c++20 -pthread -ffp-contract=off -fno-fast-math \
//       -DNO_PAR_POLICIES=0 -o det_reduce mt_extended_correctness.cpp -ltbb
//
// Local build (without TBB):
//   g++ -O3 -std=c++20 -pthread -ffp-contract=off -fno-fast-math \
//       -DNO_PAR_POLICIES=1 -o det_reduce mt_extended_correctness.cpp
//
// CRITICAL FLAGS:
//   -ffp-contract=off   Prevents FMA contraction (changes rounding)
//   -fno-fast-math      Preserves IEEE 754 semantics (no reassociation)
//   -pthread            Required for std::thread (MT paths)
//
// OPTIONAL FLAGS:
//   -DFORCE_AVX2=1      Force AVX2 codegen (Godbolt: no runtime dispatch)
//   -DFORCE_AVX512=1    Force AVX-512 codegen
//   -DNO_PAR_POLICIES=1 Skip std::reduce(par_unseq) benchmark
//   -DNO_PAR_POLICIES=0 Enable std::reduce(par_unseq) (needs -ltbb)
//   -mavx2 / -mavx512f  ISA target (when using FORCE_* macros)
//
// NOTE ON par_unseq:
//   Clang's <execution> header exists even without TBB, but par_unseq
//   falls back to sequential execution — making it slower than
//   std::reduce(no policy).  Use -DNO_PAR_POLICIES=1 on Godbolt.
//   For a genuine parallel baseline, build locally with TBB.
//
// VERIFIED CONFIGURATIONS:
//   Godbolt: Clang trunk, AVX2, -DNO_PAR_POLICIES=1     870/870 PASS
//   Local:   GCC 13, AVX-512, -DNO_PAR_POLICIES=0+TBB   870/870 PASS
//   Local:   Clang 17, AVX2, -DNO_PAR_POLICIES=0+TBB    870/870 PASS
//
// GOLDEN VALUES (any conforming platform, same SEED):
//   ST L=16:   0x40618f71f6379380
//   ST L=128:  0x40618f71f6379397
//
// ============================================================================

#include <algorithm>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
// par_unseq support: requires TBB on GCC/libstdc++, built-in on some libc++.
// Control: -DNO_PAR_POLICIES=1 to disable, -DNO_PAR_POLICIES=0 to force enable.
// If not defined, auto-detect via __has_include.
#ifdef NO_PAR_POLICIES
#  if NO_PAR_POLICIES == 0
#    include <execution>
#    define HAS_PAR_UNSEQ 1
#  else
#    define HAS_PAR_UNSEQ 0
#  endif
#elif __has_include(<execution>)
#  include <execution>
#  define HAS_PAR_UNSEQ 1
#else
#  define HAS_PAR_UNSEQ 0
#endif
#include <numeric>
#include <thread>
#include <vector>

#if __has_include(<execution>)
  #include <execution>
  #define HAS_EXECUTION 1
#else
  #define HAS_EXECUTION 0
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  #include <immintrin.h>
  #define HAS_X86 1
#else
  #define HAS_X86 0
#endif

// ------------------------------ Knobs ---------------------------------------
#ifndef PERF_WARMUP
  #define PERF_WARMUP 3
#endif
#ifndef PERF_TRIALS
  #define PERF_TRIALS 3
#endif
#ifndef PERF_INNER
  #define PERF_INNER 20
#endif
#ifndef MAX_THREADS
  #define MAX_THREADS 4
#endif

// ------------------------------ RNG -----------------------------------------
class CrossPlatformRNG {
    uint64_t state_;
public:
    explicit CrossPlatformRNG(uint64_t seed) : state_(seed) {}
    uint64_t next_u64() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return state_;
    }
    double next_double() {
        uint64_t bits = next_u64();
        int64_t mantissa = (int64_t)(bits >> 11) - (1LL << 52);
        return (double)mantissa / (double)(1LL << 52);
    }
};

// ------------------------------ Utilities -----------------------------------
static inline uint64_t as_u64(double x) {
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

static inline const char* yesno(bool b) { return b ? "YES" : "NO"; }

template<typename T>
static inline void do_not_optimize(T const& value) {
#if defined(__aarch64__)
    asm volatile("" : : "r"(value) : "memory");
#else
    asm volatile("" : : "r,m"(value) : "memory");
#endif
}

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    double ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

template<class F>
static double bench(const char* name, size_t N, F&& fn) {
    for (int i = 0; i < PERF_WARMUP; ++i) { double r = fn(); do_not_optimize(r); }
    double best_ms = 1e100;
    for (int trial = 0; trial < PERF_TRIALS; ++trial) {
        Timer t;
        for (int i = 0; i < PERF_INNER; ++i) { double r = fn(); do_not_optimize(r); }
        best_ms = std::min(best_ms, t.ms() / (double)PERF_INNER);
    }
    double gbs = (N * sizeof(double) / 1e9) / (best_ms / 1000.0);
    std::printf("  %-52s %8.3f ms   %6.2f GB/s\n", name, best_ms, gbs);
    return best_ms;
}

// Canonical scalar pairwise reduce
static inline double balanced_tree_reduce_scalars(double* data, size_t n) {
    if (n == 0) return 0.0;
    if (n == 1) return data[0];
    if (n == 2) return data[0] + data[1];
    while (n > 1) {
        size_t half = n / 2;
        for (size_t i = 0; i < half; ++i)
            data[i] = data[2*i] + data[2*i + 1];
        if (n & 1) { data[half] = data[n - 1]; n = half + 1; }
        else       { n = half; }
    }
    return data[0];
}

// ============================================================================
// ISA-specific kernels with multiversion dispatch
// ============================================================================
using CombineFn = void(*)(const double*, size_t, const double*, size_t, double*, size_t*);
using Reduce8Fn = void(*)(const double*, double*);

struct Ops {
    const char* name;
    CombineFn combine;
    Reduce8Fn reduce8_L16;
    Reduce8Fn reduce8_L128;
};

// -------------------- Scalar fallbacks --------------------
static void combine_scalar(const double* left, size_t lc,
                           const double* right, size_t rc,
                           double* out, size_t* outc) {
    const size_t common = (lc < rc) ? lc : rc;
    *outc = (lc > rc) ? lc : rc;
    for (size_t j = 0; j < common; ++j) out[j] = left[j] + right[j];
    if (lc > common) for (size_t j = common; j < lc; ++j) out[j] = left[j];
    else             for (size_t j = common; j < rc; ++j) out[j] = right[j];
}

static void reduce8_L16_scalar(const double* data, double* out) {
    constexpr size_t L = 16;
    const double* b0=data+0*L; const double* b1=data+1*L;
    const double* b2=data+2*L; const double* b3=data+3*L;
    const double* b4=data+4*L; const double* b5=data+5*L;
    const double* b6=data+6*L; const double* b7=data+7*L;
    for (size_t j = 0; j < L; ++j)
        out[j] = ((b0[j]+b1[j])+(b2[j]+b3[j])) + ((b4[j]+b5[j])+(b6[j]+b7[j]));
}

static void reduce8_L128_scalar(const double* data, double* out) {
    constexpr size_t L = 128;
    const double* b0=data+0*L; const double* b1=data+1*L;
    const double* b2=data+2*L; const double* b3=data+3*L;
    const double* b4=data+4*L; const double* b5=data+5*L;
    const double* b6=data+6*L; const double* b7=data+7*L;
    for (size_t j = 0; j < L; ++j)
        out[j] = ((b0[j]+b1[j])+(b2[j]+b3[j])) + ((b4[j]+b5[j])+(b6[j]+b7[j]));
}

// -------------------- SSE2 --------------------
#if HAS_X86
__attribute__((target("sse2")))
static void combine_sse2(const double* left, size_t lc, const double* right, size_t rc,
                         double* out, size_t* outc) {
    const size_t common = (lc < rc) ? lc : rc;
    *outc = (lc > rc) ? lc : rc;
    size_t j = 0;
    for (; j + 2 <= common; j += 2) {
        __m128d a = _mm_loadu_pd(left+j); __m128d b = _mm_loadu_pd(right+j);
        _mm_storeu_pd(out+j, _mm_add_pd(a, b));
    }
    for (; j < common; ++j) out[j] = left[j] + right[j];
    if (lc > common) for (size_t k=common; k<lc; ++k) out[k] = left[k];
    else             for (size_t k=common; k<rc; ++k) out[k] = right[k];
}

__attribute__((target("sse2")))
static void reduce8_L16_sse2(const double* data, double* out) {
    constexpr size_t L = 16;
    const double* b0=data+0*L; const double* b1=data+1*L;
    const double* b2=data+2*L; const double* b3=data+3*L;
    const double* b4=data+4*L; const double* b5=data+5*L;
    const double* b6=data+6*L; const double* b7=data+7*L;
    for (size_t j = 0; j < L; j += 2) {
        __m128d v0=_mm_loadu_pd(b0+j); __m128d v1=_mm_loadu_pd(b1+j);
        __m128d v2=_mm_loadu_pd(b2+j); __m128d v3=_mm_loadu_pd(b3+j);
        __m128d v4=_mm_loadu_pd(b4+j); __m128d v5=_mm_loadu_pd(b5+j);
        __m128d v6=_mm_loadu_pd(b6+j); __m128d v7=_mm_loadu_pd(b7+j);
        _mm_storeu_pd(out+j, _mm_add_pd(_mm_add_pd(_mm_add_pd(v0,v1),_mm_add_pd(v2,v3)),
                                         _mm_add_pd(_mm_add_pd(v4,v5),_mm_add_pd(v6,v7))));
    }
}

__attribute__((target("sse2")))
static void reduce8_L128_sse2(const double* data, double* out) {
    constexpr size_t L = 128;
    const double* b0=data+0*L; const double* b1=data+1*L;
    const double* b2=data+2*L; const double* b3=data+3*L;
    const double* b4=data+4*L; const double* b5=data+5*L;
    const double* b6=data+6*L; const double* b7=data+7*L;
    for (size_t j = 0; j < L; j += 2) {
        __m128d v0=_mm_loadu_pd(b0+j); __m128d v1=_mm_loadu_pd(b1+j);
        __m128d v2=_mm_loadu_pd(b2+j); __m128d v3=_mm_loadu_pd(b3+j);
        __m128d v4=_mm_loadu_pd(b4+j); __m128d v5=_mm_loadu_pd(b5+j);
        __m128d v6=_mm_loadu_pd(b6+j); __m128d v7=_mm_loadu_pd(b7+j);
        _mm_storeu_pd(out+j, _mm_add_pd(_mm_add_pd(_mm_add_pd(v0,v1),_mm_add_pd(v2,v3)),
                                         _mm_add_pd(_mm_add_pd(v4,v5),_mm_add_pd(v6,v7))));
    }
}
#endif

// -------------------- AVX2 --------------------
#if HAS_X86
__attribute__((target("avx2")))
static void combine_avx2(const double* left, size_t lc, const double* right, size_t rc,
                         double* out, size_t* outc) {
    const size_t common = (lc < rc) ? lc : rc;
    *outc = (lc > rc) ? lc : rc;
    size_t j = 0;
    for (; j + 4 <= common; j += 4) {
        __m256d a = _mm256_loadu_pd(left+j); __m256d b = _mm256_loadu_pd(right+j);
        _mm256_storeu_pd(out+j, _mm256_add_pd(a, b));
    }
    for (; j < common; ++j) out[j] = left[j] + right[j];
    if (lc > common) for (size_t k=common; k<lc; ++k) out[k] = left[k];
    else             for (size_t k=common; k<rc; ++k) out[k] = right[k];
}

__attribute__((target("avx2")))
static void reduce8_L16_avx2(const double* data, double* out) {
    constexpr size_t L = 16;
    const double* b0=data+0*L; const double* b1=data+1*L;
    const double* b2=data+2*L; const double* b3=data+3*L;
    const double* b4=data+4*L; const double* b5=data+5*L;
    const double* b6=data+6*L; const double* b7=data+7*L;
    for (size_t j = 0; j < L; j += 4) {
        __m256d v0=_mm256_loadu_pd(b0+j); __m256d v1=_mm256_loadu_pd(b1+j);
        __m256d v2=_mm256_loadu_pd(b2+j); __m256d v3=_mm256_loadu_pd(b3+j);
        __m256d v4=_mm256_loadu_pd(b4+j); __m256d v5=_mm256_loadu_pd(b5+j);
        __m256d v6=_mm256_loadu_pd(b6+j); __m256d v7=_mm256_loadu_pd(b7+j);
        _mm256_storeu_pd(out+j, _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v0,v1),_mm256_add_pd(v2,v3)),
                                               _mm256_add_pd(_mm256_add_pd(v4,v5),_mm256_add_pd(v6,v7))));
    }
}

__attribute__((target("avx2")))
static void reduce8_L128_avx2(const double* data, double* out) {
    constexpr size_t L = 128;
    const double* b0=data+0*L; const double* b1=data+1*L;
    const double* b2=data+2*L; const double* b3=data+3*L;
    const double* b4=data+4*L; const double* b5=data+5*L;
    const double* b6=data+6*L; const double* b7=data+7*L;
    for (size_t j = 0; j < L; j += 4) {
        __m256d v0=_mm256_loadu_pd(b0+j); __m256d v1=_mm256_loadu_pd(b1+j);
        __m256d v2=_mm256_loadu_pd(b2+j); __m256d v3=_mm256_loadu_pd(b3+j);
        __m256d v4=_mm256_loadu_pd(b4+j); __m256d v5=_mm256_loadu_pd(b5+j);
        __m256d v6=_mm256_loadu_pd(b6+j); __m256d v7=_mm256_loadu_pd(b7+j);
        _mm256_storeu_pd(out+j, _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v0,v1),_mm256_add_pd(v2,v3)),
                                               _mm256_add_pd(_mm256_add_pd(v4,v5),_mm256_add_pd(v6,v7))));
    }
}
#endif

// -------------------- AVX512 --------------------
#if HAS_X86
__attribute__((target("avx512f")))
static void combine_avx512(const double* left, size_t lc, const double* right, size_t rc,
                           double* out, size_t* outc) {
    const size_t common = (lc < rc) ? lc : rc;
    *outc = (lc > rc) ? lc : rc;
    size_t j = 0;
    for (; j + 8 <= common; j += 8) {
        __m512d a = _mm512_loadu_pd(left+j); __m512d b = _mm512_loadu_pd(right+j);
        _mm512_storeu_pd(out+j, _mm512_add_pd(a, b));
    }
    for (; j < common; ++j) out[j] = left[j] + right[j];
    if (lc > common) for (size_t k=common; k<lc; ++k) out[k] = left[k];
    else             for (size_t k=common; k<rc; ++k) out[k] = right[k];
}

__attribute__((target("avx512f")))
static void reduce8_L16_avx512(const double* data, double* out) {
    constexpr size_t L = 16;
    const double* b0=data+0*L; const double* b1=data+1*L;
    const double* b2=data+2*L; const double* b3=data+3*L;
    const double* b4=data+4*L; const double* b5=data+5*L;
    const double* b6=data+6*L; const double* b7=data+7*L;
    for (size_t j = 0; j < L; j += 8) {
        __m512d v0=_mm512_loadu_pd(b0+j); __m512d v1=_mm512_loadu_pd(b1+j);
        __m512d v2=_mm512_loadu_pd(b2+j); __m512d v3=_mm512_loadu_pd(b3+j);
        __m512d v4=_mm512_loadu_pd(b4+j); __m512d v5=_mm512_loadu_pd(b5+j);
        __m512d v6=_mm512_loadu_pd(b6+j); __m512d v7=_mm512_loadu_pd(b7+j);
        _mm512_storeu_pd(out+j, _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(v0,v1),_mm512_add_pd(v2,v3)),
                                               _mm512_add_pd(_mm512_add_pd(v4,v5),_mm512_add_pd(v6,v7))));
    }
}

__attribute__((target("avx512f")))
static void reduce8_L128_avx512(const double* data, double* out) {
    constexpr size_t L = 128;
    const double* b0=data+0*L; const double* b1=data+1*L;
    const double* b2=data+2*L; const double* b3=data+3*L;
    const double* b4=data+4*L; const double* b5=data+5*L;
    const double* b6=data+6*L; const double* b7=data+7*L;
    for (size_t j = 0; j < L; j += 8) {
        __m512d v0=_mm512_loadu_pd(b0+j); __m512d v1=_mm512_loadu_pd(b1+j);
        __m512d v2=_mm512_loadu_pd(b2+j); __m512d v3=_mm512_loadu_pd(b3+j);
        __m512d v4=_mm512_loadu_pd(b4+j); __m512d v5=_mm512_loadu_pd(b5+j);
        __m512d v6=_mm512_loadu_pd(b6+j); __m512d v7=_mm512_loadu_pd(b7+j);
        _mm512_storeu_pd(out+j, _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(v0,v1),_mm512_add_pd(v2,v3)),
                                               _mm512_add_pd(_mm512_add_pd(v4,v5),_mm512_add_pd(v6,v7))));
    }
}
#endif

// -------------------- Dispatch --------------------
static const Ops OPS_SCALAR { "SCALAR", &combine_scalar, &reduce8_L16_scalar, &reduce8_L128_scalar };
#if HAS_X86
static const Ops OPS_SSE2   { "SSE2",   &combine_sse2,   &reduce8_L16_sse2,   &reduce8_L128_sse2   };
static const Ops OPS_AVX2   { "AVX2",   &combine_avx2,   &reduce8_L16_avx2,   &reduce8_L128_avx2   };
static const Ops OPS_AVX512 { "AVX512", &combine_avx512, &reduce8_L16_avx512, &reduce8_L128_avx512 };
#endif

static const Ops& select_ops() {
#if HAS_X86 && (defined(__clang__) || defined(__GNUC__))
    if (__builtin_cpu_supports("avx512f")) return OPS_AVX512;
    if (__builtin_cpu_supports("avx2"))    return OPS_AVX2;
    if (__builtin_cpu_supports("sse2"))    return OPS_SSE2;
#endif
    return OPS_SCALAR;
}

// ============================================================================
// Stack State for Multi-Threaded Merge
// ============================================================================
template<size_t L>
struct alignas(64) StackState {
    static constexpr size_t MAX_DEPTH = 32;
    alignas(64) double buckets[MAX_DEPTH][L];
    size_t counts[MAX_DEPTH] = {};
    uint32_t mask = 0;
};

static inline bool depth_overflow(size_t level, size_t max_depth) {
    return (level >= max_depth) || (level >= 32);
}

// ============================================================================
// Single-Threaded Reduction with 8-block unrolling + SIMD
// ============================================================================
template<size_t L>
static double deterministic_reduce_ST(const Ops& ops, const double* first, size_t N) {
    constexpr size_t MAX_DEPTH = 32;
    constexpr size_t GROUP_SIZE = 8;
    constexpr size_t GROUP_LEVEL = 3;

    if (N == 0) return 0.0;
    const size_t num_blocks = (N + L - 1) / L;

    if (num_blocks == 1) {
        alignas(64) double lanes[L] = {};
        for (size_t j = 0; j < N; ++j) lanes[j] = first[j];
        return balanced_tree_reduce_scalars(lanes, N);
    }

    alignas(64) double stack[MAX_DEPTH][L];
    size_t stack_counts[MAX_DEPTH] = {};
    uint32_t valid_mask = 0;

    alignas(64) double buf0[L], buf1[L];
    double* current = buf0; double* temp = buf1;
    size_t current_count = 0;
    auto swap_buffers = [&]() { double* t = current; current = temp; temp = t; };

    auto merge_into_top = [&](const double* rhs, size_t rc) {
        constexpr size_t top = MAX_DEPTH - 1;
        if (!(valid_mask & (1u << top))) {
            for (size_t j = 0; j < rc; ++j) stack[top][j] = rhs[j];
            stack_counts[top] = rc; valid_mask |= (1u << top); return;
        }
        size_t outc = 0;
        ops.combine(stack[top], stack_counts[top], rhs, rc, stack[top], &outc);
        stack_counts[top] = outc;
    };

    const size_t full_blocks = N / L;
    const size_t unrollable_groups = full_blocks / GROUP_SIZE;

    for (size_t g = 0; g < unrollable_groups; ++g) {
        const double* gs = first + g * GROUP_SIZE * L;
        if constexpr (L == 16) ops.reduce8_L16(gs, current);
        else                   ops.reduce8_L128(gs, current);
        current_count = L;
        size_t level = GROUP_LEVEL;
        while (!depth_overflow(level, MAX_DEPTH) && (valid_mask & (1u << level))) {
            size_t tc = 0;
            ops.combine(stack[level], stack_counts[level], current, current_count, temp, &tc);
            valid_mask &= ~(1u << level); swap_buffers(); current_count = tc; ++level;
        }
        if (depth_overflow(level, MAX_DEPTH)) merge_into_top(current, current_count);
        else {
            for (size_t j = 0; j < current_count; ++j) stack[level][j] = current[j];
            stack_counts[level] = current_count; valid_mask |= (1u << level);
        }
    }

    for (size_t b = unrollable_groups * GROUP_SIZE; b < num_blocks; ++b) {
        const size_t es = b * L, ee = std::min(es + L, N);
        current_count = ee - es;
        for (size_t j = 0; j < current_count; ++j) current[j] = first[es + j];
        size_t level = 0;
        while (!depth_overflow(level, MAX_DEPTH) && (valid_mask & (1u << level))) {
            size_t tc = 0;
            ops.combine(stack[level], stack_counts[level], current, current_count, temp, &tc);
            valid_mask &= ~(1u << level); swap_buffers(); current_count = tc; ++level;
        }
        if (depth_overflow(level, MAX_DEPTH)) merge_into_top(current, current_count);
        else {
            for (size_t j = 0; j < current_count; ++j) stack[level][j] = current[j];
            stack_counts[level] = current_count; valid_mask |= (1u << level);
        }
    }

    bool have = false;
    for (size_t level = 0; level < MAX_DEPTH; ++level) {
        if (!(valid_mask & (1u << level))) continue;
        if (!have) {
            for (size_t j = 0; j < stack_counts[level]; ++j) current[j] = stack[level][j];
            current_count = stack_counts[level]; have = true;
        } else {
            size_t tc = 0;
            ops.combine(stack[level], stack_counts[level], current, current_count, temp, &tc);
            swap_buffers(); current_count = tc;
        }
    }
    return balanced_tree_reduce_scalars(current, current_count);
}

// ============================================================================
// Multi-Threaded Local Replay
// ============================================================================
template<size_t L>
static StackState<L> replay_range(const Ops& ops, const double* first, size_t N,
                                   size_t block_start, size_t block_end, size_t /*total_blocks*/) {
    constexpr size_t MAX_DEPTH = StackState<L>::MAX_DEPTH;
    constexpr size_t GROUP_SIZE = 8;
    constexpr size_t GROUP_LEVEL = 3;
    StackState<L> S{};
    alignas(64) double buf0[L], buf1[L];
    double* current = buf0; double* temp = buf1;
    size_t current_count = 0;
    auto swap_buffers = [&]() { double* t = current; current = temp; temp = t; };
    auto merge_into_top = [&](const double* rhs, size_t rc) {
        constexpr size_t top = MAX_DEPTH - 1;
        if (!(S.mask & (1u << top))) {
            for (size_t j = 0; j < rc; ++j) S.buckets[top][j] = rhs[j];
            S.counts[top] = rc; S.mask |= (1u << top); return;
        }
        size_t outc = 0;
        ops.combine(S.buckets[top], S.counts[top], rhs, rc, S.buckets[top], &outc);
        S.counts[top] = outc;
    };

    size_t full_in_range = 0;
    for (size_t b = block_start; b < block_end; ++b)
        if (b * L + L <= N) ++full_in_range;
    const size_t ug = full_in_range / GROUP_SIZE;

    for (size_t g = 0; g < ug; ++g) {
        const double* gs = first + (block_start + g * GROUP_SIZE) * L;
        if constexpr (L == 16) ops.reduce8_L16(gs, current);
        else                   ops.reduce8_L128(gs, current);
        current_count = L;
        size_t level = GROUP_LEVEL;
        while (!depth_overflow(level, MAX_DEPTH) && (S.mask & (1u << level))) {
            size_t tc = 0;
            ops.combine(S.buckets[level], S.counts[level], current, current_count, temp, &tc);
            S.mask &= ~(1u << level); swap_buffers(); current_count = tc; ++level;
        }
        if (depth_overflow(level, MAX_DEPTH)) merge_into_top(current, current_count);
        else {
            for (size_t j = 0; j < current_count; ++j) S.buckets[level][j] = current[j];
            S.counts[level] = current_count; S.mask |= (1u << level);
        }
    }

    for (size_t b = block_start + ug * GROUP_SIZE; b < block_end; ++b) {
        const size_t es = b * L, ee = std::min(es + L, N);
        current_count = ee - es;
        for (size_t j = 0; j < current_count; ++j) current[j] = first[es + j];
        size_t level = 0;
        while (!depth_overflow(level, MAX_DEPTH) && (S.mask & (1u << level))) {
            size_t tc = 0;
            ops.combine(S.buckets[level], S.counts[level], current, current_count, temp, &tc);
            S.mask &= ~(1u << level); swap_buffers(); current_count = tc; ++level;
        }
        if (depth_overflow(level, MAX_DEPTH)) merge_into_top(current, current_count);
        else {
            for (size_t j = 0; j < current_count; ++j) S.buckets[level][j] = current[j];
            S.counts[level] = current_count; S.mask |= (1u << level);
        }
    }
    return S;
}

template<size_t L>
static void merge_states(const Ops& ops, StackState<L>& A, const StackState<L>& B) {
    constexpr size_t MAX_DEPTH = StackState<L>::MAX_DEPTH;
    alignas(64) double buf0[L], buf1[L];
    double* current = buf0; double* temp = buf1;
    auto swap_buffers = [&]() { double* t = current; current = temp; temp = t; };
    for (size_t k = 0; k < MAX_DEPTH; ++k) {
        if (!(B.mask & (1u << k))) continue;
        for (size_t j = 0; j < B.counts[k]; ++j) current[j] = B.buckets[k][j];
        size_t cc = B.counts[k];
        size_t level = k;
        while (!depth_overflow(level, MAX_DEPTH) && (A.mask & (1u << level))) {
            size_t tc = 0;
            ops.combine(A.buckets[level], A.counts[level], current, cc, temp, &tc);
            A.mask &= ~(1u << level); swap_buffers(); cc = tc; ++level;
        }
        if (depth_overflow(level, MAX_DEPTH)) {
            level = MAX_DEPTH - 1;
            if (A.mask & (1u << level)) {
                size_t outc = 0;
                ops.combine(A.buckets[level], A.counts[level], current, cc, A.buckets[level], &outc);
                A.counts[level] = outc;
            } else {
                for (size_t j = 0; j < cc; ++j) A.buckets[level][j] = current[j];
                A.counts[level] = cc; A.mask |= (1u << level);
            }
        } else {
            for (size_t j = 0; j < cc; ++j) A.buckets[level][j] = current[j];
            A.counts[level] = cc; A.mask |= (1u << level);
        }
    }
}

template<size_t L>
static double fold_state(const Ops& ops, const StackState<L>& S) {
    constexpr size_t MAX_DEPTH = StackState<L>::MAX_DEPTH;
    alignas(64) double buf0[L], buf1[L];
    double* current = buf0; double* temp = buf1;
    size_t cc = 0; bool have = false;
    auto swap_buffers = [&]() { double* t = current; current = temp; temp = t; };
    for (size_t level = 0; level < MAX_DEPTH; ++level) {
        if (!(S.mask & (1u << level))) continue;
        if (!have) {
            for (size_t j = 0; j < S.counts[level]; ++j) current[j] = S.buckets[level][j];
            cc = S.counts[level]; have = true;
        } else {
            size_t tc = 0;
            ops.combine(S.buckets[level], S.counts[level], current, cc, temp, &tc);
            swap_buffers(); cc = tc;
        }
    }
    if (!have) return 0.0;
    return balanced_tree_reduce_scalars(current, cc);
}

static inline size_t choose_chunk_size(size_t num_blocks, size_t T) {
    if (num_blocks <= T) return 1;
    return size_t{1} << (std::bit_width(num_blocks / T) - 1);
}

template<size_t L>
static double deterministic_reduce_MT(const Ops& ops, const double* first, size_t N, size_t T) {
    if (N == 0) return 0.0;
    if (T == 0) T = 1;
    if (T > MAX_THREADS) T = MAX_THREADS;
    const size_t num_blocks = (N + L - 1) / L;
    if (num_blocks <= T) return deterministic_reduce_ST<L>(ops, first, N);
    const size_t C = choose_chunk_size(num_blocks, T);
    const size_t num_chunks = (num_blocks + C - 1) / C;
    const size_t actual_threads = std::min(num_chunks, T);
    std::vector<StackState<L>> states(num_chunks);
    {
        std::vector<std::thread> threads;
        threads.reserve(actual_threads);
        for (size_t t = 0; t < actual_threads; ++t) {
            size_t cs = (num_chunks * t) / actual_threads;
            size_t ce = (num_chunks * (t + 1)) / actual_threads;
            threads.emplace_back([&, cs, ce]() {
                for (size_t chunk = cs; chunk < ce; ++chunk) {
                    size_t b0 = chunk * C, b1 = std::min(b0 + C, num_blocks);
                    states[chunk] = replay_range<L>(ops, first, N, b0, b1, num_blocks);
                }
            });
        }
        for (auto& t : threads) t.join();
    }
    for (size_t i = 1; i < num_chunks; ++i)
        merge_states<L>(ops, states[0], states[i]);
    return fold_state<L>(ops, states[0]);
}

// ============================================================================
// HOSTILE DATA GENERATORS
//
// These produce sequences where any change in associativity (grouping)
// will change the result.  If MT matches ST bitwise on these, the merge
// algorithm is correctly preserving the canonical tree.
// ============================================================================

// Generator 0: uniform random in [-1, 1) — the original baseline
static void gen_uniform(double* out, size_t N, uint64_t seed) {
    CrossPlatformRNG rng(seed);
    for (size_t i = 0; i < N; ++i) out[i] = rng.next_double();
}

// Generator 1: near-cancellation pairs
// Alternates +C and -(C + tiny_perturbation).  The sum is a sequence of
// tiny residuals that are extremely sensitive to grouping order because
// the intermediate magnitudes are ~C while the true answer is ~N*eps*C.
static void gen_cancellation(double* out, size_t N, uint64_t seed) {
    CrossPlatformRNG rng(seed);
    const double C = 1e15;
    for (size_t i = 0; i < N; ++i) {
        double perturbation = rng.next_double();
        out[i] = (i & 1) ? -(C + perturbation * 0.5) : (C + perturbation * 0.5);
    }
}

// Generator 2: exponential staircase
// Values span ~300 orders of magnitude (the full double range).
// Swallowing occurs when small values are added to large ones in the
// wrong order.  Any reassociation changes which values get swallowed.
static void gen_exponential(double* out, size_t N, uint64_t seed) {
    CrossPlatformRNG rng(seed);
    for (size_t i = 0; i < N; ++i) {
        double exponent = (double)((int)(i % 301) - 150);
        double sign = (rng.next_u64() & 1) ? 1.0 : -1.0;
        double mantissa = 0.5 + 0.5 * ((double)(rng.next_u64() >> 12) / (double)(1ULL << 52));
        out[i] = sign * std::ldexp(mantissa, (int)exponent);
    }
}

// Generator 3: Kahan's "ill-conditioned sum" pattern
// All values are +1.0 except every K-th is -(K-1).0, so partial sums
// oscillate between 0 and K-1.  The condition number is O(N).
static void gen_kahan(double* out, size_t N, uint64_t seed) {
    (void)seed;
    const size_t K = 7;
    for (size_t i = 0; i < N; ++i)
        out[i] = (i % K == 0 && i > 0) ? -(double)(K - 1) : 1.0;
}

// Generator 4: subnormal / boundary values
// Mix of subnormals, normals near underflow/overflow, and exact zeros.
static void gen_subnormal(double* out, size_t N, uint64_t seed) {
    CrossPlatformRNG rng(seed);
    for (size_t i = 0; i < N; ++i) {
        switch (i % 5) {
            case 0: out[i] =  5e-324 * (double)(rng.next_u64() % 1000); break;
            case 1: out[i] = -5e-324 * (double)(rng.next_u64() % 1000); break;
            case 2: out[i] =  1e+308 / (double)N; break;
            case 3: out[i] = -1e+308 / (double)N; break;
            case 4: out[i] =  0.0; break;
        }
    }
}

struct Generator {
    const char* name;
    void (*fn)(double*, size_t, uint64_t);
};

static const Generator GENERATORS[] = {
    { "uniform[-1,1)",   gen_uniform      },
    { "cancellation",    gen_cancellation  },
    { "exponential",     gen_exponential   },
    { "kahan",           gen_kahan         },
    { "subnormal/edge",  gen_subnormal     },
};
static constexpr size_t NUM_GENERATORS = sizeof(GENERATORS) / sizeof(GENERATORS[0]);

// ============================================================================
// Constants
// ============================================================================
constexpr uint64_t SEED = 0x243F6A8885A308D3ULL;
constexpr size_t N_PERF = 1'000'000;

constexpr uint64_t EXPECT_NARROW = 0x40618f71f6379380ULL;
constexpr uint64_t EXPECT_WIDE   = 0x40618f71f6379397ULL;

// ============================================================================
// Main
// ============================================================================
int main() {
    std::printf("================================================================\n");
    std::printf("  Deterministic Parallel Reduction — EXTENDED CORRECTNESS\n");
    std::printf("  Hostile data x awkward sizes x odd thread counts\n");
    std::printf("  (Godbolt-safe: max %d threads)\n", MAX_THREADS);
    std::printf("================================================================\n\n");

    const Ops& ops = select_ops();
    std::printf("Selected ISA: %s\n\n", ops.name);

    bool all_ok = true;

    // ==================================================================
    // SECTION 1: Golden-value verification at N=1,000,000
    // ==================================================================
    std::printf("================================================================\n");
    std::printf("SECTION 1: Golden-value verification (N=%zu)\n", N_PERF);
    std::printf("================================================================\n");

    std::vector<double> perf_data(N_PERF);
    gen_uniform(perf_data.data(), N_PERF, SEED);

    const uint64_t exps[5] = {
        0x3fd37de3b20e9fdcULL, 0xbfd2e1595e76077cULL, 0xbfd5c999955b530cULL,
        0xbfe6be1806d7224eULL, 0x3fef95133e17376eULL
    };
    bool data_ok = true;
    for (int i = 0; i < 5; ++i) data_ok &= (as_u64(perf_data[(size_t)i]) == exps[i]);
    std::printf("  Data fingerprint: %s\n", yesno(data_ok));
    all_ok &= data_ok;

    double st16  = deterministic_reduce_ST<16>(ops, perf_data.data(), N_PERF);
    double st128 = deterministic_reduce_ST<128>(ops, perf_data.data(), N_PERF);
    bool st_ok = (as_u64(st16) == EXPECT_NARROW) && (as_u64(st128) == EXPECT_WIDE);
    std::printf("  ST L=16:  0x%016" PRIx64 "  %s\n", as_u64(st16), yesno(as_u64(st16)==EXPECT_NARROW));
    std::printf("  ST L=128: 0x%016" PRIx64 "  %s\n", as_u64(st128), yesno(as_u64(st128)==EXPECT_WIDE));
    all_ok &= st_ok;

    bool mt_golden_ok = true;
    for (size_t T : {2, 3, 4}) {
        double mt16  = deterministic_reduce_MT<16>(ops, perf_data.data(), N_PERF, T);
        double mt128 = deterministic_reduce_MT<128>(ops, perf_data.data(), N_PERF, T);
        bool ok = (as_u64(mt16)==as_u64(st16)) && (as_u64(mt128)==as_u64(st128));
        mt_golden_ok &= ok;
        std::printf("  MT T=%-2zu  L=16: %s   L=128: %s\n", T,
                    yesno(as_u64(mt16)==as_u64(st16)), yesno(as_u64(mt128)==as_u64(st128)));
    }
    all_ok &= mt_golden_ok;
    std::printf("  Golden verdict: %s\n\n", (mt_golden_ok && st_ok) ? "PASS" : "FAIL");

    // ==================================================================
    // SECTION 2: Hostile data x awkward sizes x thread counts
    // ==================================================================

    // Deduplicated, sorted awkward sizes
    const size_t RAW_SIZES[] = {
        // Degenerate: N < L=16
        1, 2, 3, 7, 13, 15,
        // At/near L=16 boundary
        16, 17, 31,
        // At/near L=128 boundary
        127, 128, 129,
        // 8-block unroll boundary for L=16: 8*16=128
        126, 130,
        // Primes that divide awkwardly by L, T, and C
        251, 257, 509, 521, 997, 1009,
        // Near powers of 2 (chunk partition stress)
        1023, 1024, 1025,
        // Moderate primes
        4999, 10007, 65521,
        // Large primes
        100003, 999983, 1000003,
    };
    std::vector<size_t> sizes;
    for (size_t s : RAW_SIZES) {
        bool dup = false;
        for (size_t x : sizes) if (x == s) { dup = true; break; }
        if (!dup) sizes.push_back(s);
    }
    std::sort(sizes.begin(), sizes.end());

    const size_t max_n = sizes.back();
    std::vector<double> hostile_buf(max_n);

    std::printf("================================================================\n");
    std::printf("SECTION 2: Hostile correctness sweep\n");
    std::printf("  %zu generators x %zu sizes x 3 thread counts x 2 lane widths\n",
                NUM_GENERATORS, sizes.size());
    std::printf("  = %zu bitwise comparisons against ST reference\n",
                NUM_GENERATORS * sizes.size() * 3 * 2);
    std::printf("================================================================\n\n");

    size_t total_tests = 0, total_pass = 0;

    for (size_t gi = 0; gi < NUM_GENERATORS; ++gi) {
        const auto& gen = GENERATORS[gi];
        std::printf("  %-18s ", gen.name);

        gen.fn(hostile_buf.data(), max_n, SEED + gi * 0x9E3779B97F4A7C15ULL);

        size_t gen_pass = 0, gen_total = 0;

        for (size_t n : sizes) {
            double ref16  = deterministic_reduce_ST<16>(ops, hostile_buf.data(), n);
            double ref128 = deterministic_reduce_ST<128>(ops, hostile_buf.data(), n);

            for (size_t T : {2, 3, 4}) {
                double mt16 = deterministic_reduce_MT<16>(ops, hostile_buf.data(), n, T);
                ++gen_total;
                if (as_u64(mt16) == as_u64(ref16)) ++gen_pass;
                else std::printf("\n    FAIL: N=%zu T=%zu L=16  ST=0x%016" PRIx64
                                 " MT=0x%016" PRIx64, n, T, as_u64(ref16), as_u64(mt16));

                double mt128 = deterministic_reduce_MT<128>(ops, hostile_buf.data(), n, T);
                ++gen_total;
                if (as_u64(mt128) == as_u64(ref128)) ++gen_pass;
                else std::printf("\n    FAIL: N=%zu T=%zu L=128 ST=0x%016" PRIx64
                                 " MT=0x%016" PRIx64, n, T, as_u64(ref128), as_u64(mt128));
            }
        }

        total_tests += gen_total;
        total_pass  += gen_pass;
        bool gen_ok = (gen_pass == gen_total);
        all_ok &= gen_ok;
        std::printf("%zu/%zu %s\n", gen_pass, gen_total, gen_ok ? "PASS" : "FAIL");
    }
    std::printf("\n  Sweep total: %zu/%zu\n\n", total_pass, total_tests);

    // ==================================================================
    // SECTION 3: Performance (N=1M)
    // ==================================================================
    std::printf("================================================================\n");
    std::printf("SECTION 3: Performance (N=%zu)\n", N_PERF);
    std::printf("================================================================\n\n");

    const double* ptr = perf_data.data();

    double t_acc = bench("std::accumulate", N_PERF, [&]() {
        return std::accumulate(ptr, ptr + N_PERF, 0.0);
    });
    (void)bench("std::reduce (no policy)", N_PERF, [&]() {
        return std::reduce(ptr, ptr + N_PERF, 0.0);
    });
    double t_par = 0;
#if HAS_PAR_UNSEQ
    t_par = bench("std::reduce(par_unseq)  [TBB]", N_PERF, [&]() {
        return std::reduce(std::execution::par_unseq, ptr, ptr + N_PERF, 0.0);
    });
#else
    std::printf("  std::reduce(par_unseq)  [TBB not available — skipped]\n");
#endif
    std::printf("\n");

    double t_st16 = bench("deterministic L=16  (ST, 8-block unroll)", N_PERF, [&]() {
        return deterministic_reduce_ST<16>(ops, ptr, N_PERF);
    });
    double t_st128 = bench("deterministic L=128 (ST, 8-block unroll)", N_PERF, [&]() {
        return deterministic_reduce_ST<128>(ops, ptr, N_PERF);
    });
    std::printf("\n");

    double t_mt16_2 = bench("deterministic L=16  (MT T=2)", N_PERF, [&]() {
        return deterministic_reduce_MT<16>(ops, ptr, N_PERF, 2);
    });
    double t_mt128_2 = bench("deterministic L=128 (MT T=2)", N_PERF, [&]() {
        return deterministic_reduce_MT<128>(ops, ptr, N_PERF, 2);
    });
    double t_mt16_4 = bench("deterministic L=16  (MT T=4)", N_PERF, [&]() {
        return deterministic_reduce_MT<16>(ops, ptr, N_PERF, 4);
    });
    double t_mt128_4 = bench("deterministic L=128 (MT T=4)", N_PERF, [&]() {
        return deterministic_reduce_MT<128>(ops, ptr, N_PERF, 4);
    });

    if (t_par > 0) {
        std::printf("\n  Cost of determinism vs std::reduce(par_unseq), N=%zu:\n", N_PERF);
        std::printf("    det ST L=16:  %.3f / %.3f ms = %.2fx (%+.1f%%)\n",
                    t_st16, t_par, t_st16/t_par, (t_st16/t_par - 1.0)*100.0);
        std::printf("    det MT T=4:   %.3f / %.3f ms = %.2fx (%+.1f%%)\n",
                    t_mt16_4, t_par, t_mt16_4/t_par, (t_mt16_4/t_par - 1.0)*100.0);
    }

    // ==================================================================
    // SECTION 4: Large-N performance (N=10M — shows MT scaling)
    // ==================================================================
    constexpr size_t N_LARGE = 10'000'000;
    std::printf("\n================================================================\n");
    std::printf("SECTION 4: Large-N performance (N=%zu — thread scaling)\n", N_LARGE);
    std::printf("================================================================\n\n");

    std::vector<double> large_data(N_LARGE);
    {
        CrossPlatformRNG rng2(SEED);
        for (size_t i = 0; i < N_LARGE; ++i) large_data[i] = rng2.next_double();
    }
    const double* lptr = large_data.data();

    // Quick MT correctness sanity check at this size
    {
        double lst16   = deterministic_reduce_ST<16>(ops, lptr, N_LARGE);
        double lmt16_3 = deterministic_reduce_MT<16>(ops, lptr, N_LARGE, 3);
        double lmt16_4 = deterministic_reduce_MT<16>(ops, lptr, N_LARGE, 4);
        bool lok = (as_u64(lst16) == as_u64(lmt16_3)) && (as_u64(lst16) == as_u64(lmt16_4));
        std::printf("  MT correctness at N=%zu: T=3 %s  T=4 %s\n\n", N_LARGE,
                    yesno(as_u64(lst16)==as_u64(lmt16_3)), yesno(as_u64(lst16)==as_u64(lmt16_4)));
        all_ok &= lok;
    }

    double tl_acc = bench("std::accumulate", N_LARGE, [&]() {
        return std::accumulate(lptr, lptr + N_LARGE, 0.0);
    });
    (void)bench("std::reduce (no policy)", N_LARGE, [&]() {
        return std::reduce(lptr, lptr + N_LARGE, 0.0);
    });
    double tl_par = 0;
#if HAS_PAR_UNSEQ
    tl_par = bench("std::reduce(par_unseq)  [TBB]", N_LARGE, [&]() {
        return std::reduce(std::execution::par_unseq, lptr, lptr + N_LARGE, 0.0);
    });
#else
    std::printf("  std::reduce(par_unseq)  [TBB not available — skipped]\n");
#endif
    std::printf("\n");

    double tl_st16 = bench("deterministic L=16  (ST)", N_LARGE, [&]() {
        return deterministic_reduce_ST<16>(ops, lptr, N_LARGE);
    });
    double tl_st128 = bench("deterministic L=128 (ST)", N_LARGE, [&]() {
        return deterministic_reduce_ST<128>(ops, lptr, N_LARGE);
    });
    std::printf("\n");

    double tl_mt16_2 = bench("deterministic L=16  (MT T=2)", N_LARGE, [&]() {
        return deterministic_reduce_MT<16>(ops, lptr, N_LARGE, 2);
    });
    double tl_mt128_2 = bench("deterministic L=128 (MT T=2)", N_LARGE, [&]() {
        return deterministic_reduce_MT<128>(ops, lptr, N_LARGE, 2);
    });
    double tl_mt16_3 = bench("deterministic L=16  (MT T=3)", N_LARGE, [&]() {
        return deterministic_reduce_MT<16>(ops, lptr, N_LARGE, 3);
    });
    double tl_mt128_3 = bench("deterministic L=128 (MT T=3)", N_LARGE, [&]() {
        return deterministic_reduce_MT<128>(ops, lptr, N_LARGE, 3);
    });
    double tl_mt16_4 = bench("deterministic L=16  (MT T=4)", N_LARGE, [&]() {
        return deterministic_reduce_MT<16>(ops, lptr, N_LARGE, 4);
    });
    double tl_mt128_4 = bench("deterministic L=128 (MT T=4)", N_LARGE, [&]() {
        return deterministic_reduce_MT<128>(ops, lptr, N_LARGE, 4);
    });

    std::printf("\n  Speedup vs std::accumulate (N=%zu):\n", N_LARGE);
    if (tl_par > 0)
    std::printf("    reduce(par_unseq): %.2fx\n", tl_acc / tl_par);
    std::printf("    det ST  L=16:      %.2fx\n", tl_acc / tl_st16);
    std::printf("    det MT  T=2 L=16:  %.2fx\n", tl_acc / tl_mt16_2);
    std::printf("    det MT  T=3 L=16:  %.2fx\n", tl_acc / tl_mt16_3);
    std::printf("    det MT  T=4 L=16:  %.2fx\n", tl_acc / tl_mt16_4);

    if (tl_par > 0) {
        std::printf("\n  Cost of determinism vs std::reduce(par_unseq), N=%zu:\n", N_LARGE);
        std::printf("    det ST  L=16:  %.3f / %.3f ms = %.2fx (%+.1f%%)\n",
                    tl_st16, tl_par, tl_st16/tl_par, (tl_st16/tl_par - 1.0)*100.0);
        std::printf("    det MT  T=2:   %.3f / %.3f ms = %.2fx (%+.1f%%)\n",
                    tl_mt16_2, tl_par, tl_mt16_2/tl_par, (tl_mt16_2/tl_par - 1.0)*100.0);
        std::printf("    det MT  T=4:   %.3f / %.3f ms = %.2fx (%+.1f%%)\n",
                    tl_mt16_4, tl_par, tl_mt16_4/tl_par, (tl_mt16_4/tl_par - 1.0)*100.0);
    }

    // ==================================================================
    // FINAL SUMMARY
    // ==================================================================
    std::printf("\n================================================================\n");
    std::printf("VERIFICATION SUMMARY\n");
    std::printf("================================================================\n");
    std::printf("ISA:             %s\n", ops.name);
    std::printf("SEED:            0x%016" PRIx64 "\n", SEED);
    std::printf("Data fingerprint:%s\n", yesno(data_ok));
    std::printf("Golden ST:       %s\n", yesno(st_ok));
    std::printf("Golden MT:       %s (T=2,3,4 x L=16,128)\n", yesno(mt_golden_ok));
    std::printf("Hostile sweep:   %zu/%zu\n", total_pass, total_tests);
    std::printf("Generators:      uniform, cancellation, exponential, kahan, subnormal\n");
    std::printf("Sizes tested:    %zu (N=1 to N=%zu)\n", sizes.size(), sizes.back());
    std::printf("Thread counts:   T=2, T=3, T=4\n");
    std::printf("Lane widths:     L=16, L=128\n");
    std::printf("================================================================\n");
    std::printf("\nOverall: %s\n", all_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
