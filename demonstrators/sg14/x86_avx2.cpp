// ============================================================================
// Deterministic Parallel Reduction — x86 single-file Godbolt
// Exact tree order preserved + 8-block unrolling
// Portable baseline binary + runtime dispatch to SSE2 / AVX2 / AVX512 (no SIGILL)
//
// Adds: LENGTH-SWEEP REPRODUCIBILITY on both BASE + HOSTILE (high cancellation)
//   - Multiple lengths (tiny, around L, around k*L boundaries)
//   - Hostile overlay: +1e16 at i%7==0, -1e16 at i%11==0, plus (+,+,-,-) cluster
//
// Flags (safe for CE Run):
//   -O3 -std=c++20 -ffp-contract=off -fno-fast-math
// Optional:
//   -DNO_PAR_POLICIES=1
//
// CHANGES (this revision):
//   1) AVX512 target fixed: use avx512f only (no avx512dq gating mismatch).
//   2) Depth/shift guard: prevents UB (1u<<level) and stack OOB if level >= 32/ MAX_DEPTH.
//      (For the intended demo sizes this never triggers; guard is for robustness.)
//   3) Removed unnecessary zero-fill of current[] tail (combine respects counts).
//   4) Ping-pong buffers for current/temp to avoid copy loops in combine hot path.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

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

namespace std_proposal {
inline constexpr std::size_t deterministic_reduce_narrow = 128;  // L=16 doubles
inline constexpr std::size_t deterministic_reduce_wide   = 1024; // L=128 doubles
}

#ifndef PERF_WARMUP
  #define PERF_WARMUP 5
#endif
#ifndef PERF_TRIALS
  #define PERF_TRIALS 5
#endif
#ifndef PERF_INNER
  #define PERF_INNER 50
#endif

// ============================================================================
// RNG (matches your golden dataset)
// ============================================================================
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

// ============================================================================
// Utilities
// ============================================================================
static inline std::string to_hex(double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%016llx", (unsigned long long)bits);
    return buf;
}

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
    std::printf("  %-40s %8.3f ms   %6.2f GB/s\n", name, best_ms, gbs);
    return best_ms;
}

// ============================================================================
// Balanced scalar tree reduce (exact iterative pairwise + carry-forward)
// ============================================================================
static inline double balanced_tree_reduce_scalars(double* data, size_t n) {
    if (n == 0) return 0.0;
    if (n == 1) return data[0];
    if (n == 2) return data[0] + data[1];

    while (n > 1) {
        size_t half = n / 2;
        for (size_t i = 0; i < half; ++i) {
            data[i] = data[2*i] + data[2*i + 1];
        }
        if (n & 1) {
            data[half] = data[n - 1];
            n = half + 1;
        } else {
            n = half;
        }
    }
    return data[0];
}

// ============================================================================
// High cancellation (hostile) overlay
// ============================================================================
static void inject_cancellation_prime_stride(std::vector<double>& x,
                                             size_t p1 = 7, size_t p2 = 11,
                                             double A = 1e16) {
    for (size_t i = 0; i < x.size(); ++i) {
        if ((i % p1) == 0) x[i] += A;
        if ((i % p2) == 0) x[i] -= A;
    }
}

static void inject_cluster(std::vector<double>& x, double A = 1e16) {
    if (x.size() < 8) return;
    size_t m = x.size() / 2;
    x[m + 0] += A;
    x[m + 1] += A;
    x[m + 2] -= A;
    x[m + 3] -= A;
}

// ============================================================================
// ISA-specific kernels (multiversion) for two L sizes: 16 and 128
// We only use vector adds (no horizontal ops), preserving exact per-lane tree.
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
    else            for (size_t j = common; j < rc; ++j) out[j] = right[j];
}

static void reduce8_L16_scalar(const double* data, double* out) {
    constexpr size_t L = 16;
    const double* b0 = data + 0*L; const double* b1 = data + 1*L;
    const double* b2 = data + 2*L; const double* b3 = data + 3*L;
    const double* b4 = data + 4*L; const double* b5 = data + 5*L;
    const double* b6 = data + 6*L; const double* b7 = data + 7*L;
    for (size_t j = 0; j < L; ++j) {
        double leftv  = (b0[j] + b1[j]) + (b2[j] + b3[j]);
        double rightv = (b4[j] + b5[j]) + (b6[j] + b7[j]);
        out[j] = leftv + rightv;
    }
}

static void reduce8_L128_scalar(const double* data, double* out) {
    constexpr size_t L = 128;
    const double* b0 = data + 0*L; const double* b1 = data + 1*L;
    const double* b2 = data + 2*L; const double* b3 = data + 3*L;
    const double* b4 = data + 4*L; const double* b5 = data + 5*L;
    const double* b6 = data + 6*L; const double* b7 = data + 7*L;
    for (size_t j = 0; j < L; ++j) {
        double leftv  = (b0[j] + b1[j]) + (b2[j] + b3[j]);
        double rightv = (b4[j] + b5[j]) + (b6[j] + b7[j]);
        out[j] = leftv + rightv;
    }
}

// -------------------- SSE2 --------------------
#if HAS_X86
__attribute__((target("sse2")))
static void combine_sse2(const double* left, size_t lc,
                         const double* right, size_t rc,
                         double* out, size_t* outc) {
    const size_t common = (lc < rc) ? lc : rc;
    *outc = (lc > rc) ? lc : rc;

    size_t j = 0;
    for (; j + 2 <= common; j += 2) {
        __m128d a = _mm_loadu_pd(left + j);
        __m128d b = _mm_loadu_pd(right + j);
        _mm_storeu_pd(out + j, _mm_add_pd(a, b));
    }
    for (; j < common; ++j) out[j] = left[j] + right[j];

    if (lc > common) for (size_t k = common; k < lc; ++k) out[k] = left[k];
    else             for (size_t k = common; k < rc; ++k) out[k] = right[k];
}

__attribute__((target("sse2")))
static void reduce8_L16_sse2(const double* data, double* out) {
    constexpr size_t L = 16;
    const double* b0 = data + 0*L; const double* b1 = data + 1*L;
    const double* b2 = data + 2*L; const double* b3 = data + 3*L;
    const double* b4 = data + 4*L; const double* b5 = data + 5*L;
    const double* b6 = data + 6*L; const double* b7 = data + 7*L;
    for (size_t j = 0; j < L; j += 2) {
        __m128d v0 = _mm_loadu_pd(b0 + j);
        __m128d v1 = _mm_loadu_pd(b1 + j);
        __m128d v2 = _mm_loadu_pd(b2 + j);
        __m128d v3 = _mm_loadu_pd(b3 + j);
        __m128d v4 = _mm_loadu_pd(b4 + j);
        __m128d v5 = _mm_loadu_pd(b5 + j);
        __m128d v6 = _mm_loadu_pd(b6 + j);
        __m128d v7 = _mm_loadu_pd(b7 + j);

        __m128d sum01 = _mm_add_pd(v0, v1);
        __m128d sum23 = _mm_add_pd(v2, v3);
        __m128d sum45 = _mm_add_pd(v4, v5);
        __m128d sum67 = _mm_add_pd(v6, v7);

        __m128d leftv  = _mm_add_pd(sum01, sum23);
        __m128d rightv = _mm_add_pd(sum45, sum67);

        _mm_storeu_pd(out + j, _mm_add_pd(leftv, rightv));
    }
}

__attribute__((target("sse2")))
static void reduce8_L128_sse2(const double* data, double* out) {
    constexpr size_t L = 128;
    const double* b0 = data + 0*L; const double* b1 = data + 1*L;
    const double* b2 = data + 2*L; const double* b3 = data + 3*L;
    const double* b4 = data + 4*L; const double* b5 = data + 5*L;
    const double* b6 = data + 6*L; const double* b7 = data + 7*L;
    for (size_t j = 0; j < L; j += 2) {
        __m128d v0 = _mm_loadu_pd(b0 + j);
        __m128d v1 = _mm_loadu_pd(b1 + j);
        __m128d v2 = _mm_loadu_pd(b2 + j);
        __m128d v3 = _mm_loadu_pd(b3 + j);
        __m128d v4 = _mm_loadu_pd(b4 + j);
        __m128d v5 = _mm_loadu_pd(b5 + j);
        __m128d v6 = _mm_loadu_pd(b6 + j);
        __m128d v7 = _mm_loadu_pd(b7 + j);

        __m128d sum01 = _mm_add_pd(v0, v1);
        __m128d sum23 = _mm_add_pd(v2, v3);
        __m128d sum45 = _mm_add_pd(v4, v5);
        __m128d sum67 = _mm_add_pd(v6, v7);

        __m128d leftv  = _mm_add_pd(sum01, sum23);
        __m128d rightv = _mm_add_pd(sum45, sum67);

        _mm_storeu_pd(out + j, _mm_add_pd(leftv, rightv));
    }
}
#endif

// -------------------- AVX2 --------------------
#if HAS_X86
__attribute__((target("avx2")))
static void combine_avx2(const double* left, size_t lc,
                         const double* right, size_t rc,
                         double* out, size_t* outc) {
    const size_t common = (lc < rc) ? lc : rc;
    *outc = (lc > rc) ? lc : rc;

    size_t j = 0;
    for (; j + 4 <= common; j += 4) {
        __m256d a = _mm256_loadu_pd(left + j);
        __m256d b = _mm256_loadu_pd(right + j);
        _mm256_storeu_pd(out + j, _mm256_add_pd(a, b));
    }
    for (; j < common; ++j) out[j] = left[j] + right[j];

    if (lc > common) for (size_t k = common; k < lc; ++k) out[k] = left[k];
    else             for (size_t k = common; k < rc; ++k) out[k] = right[k];
}

__attribute__((target("avx2")))
static void reduce8_L16_avx2(const double* data, double* out) {
    constexpr size_t L = 16;
    const double* b0 = data + 0*L; const double* b1 = data + 1*L;
    const double* b2 = data + 2*L; const double* b3 = data + 3*L;
    const double* b4 = data + 4*L; const double* b5 = data + 5*L;
    const double* b6 = data + 6*L; const double* b7 = data + 7*L;
    for (size_t j = 0; j < L; j += 4) {
        __m256d v0 = _mm256_loadu_pd(b0 + j);
        __m256d v1 = _mm256_loadu_pd(b1 + j);
        __m256d v2 = _mm256_loadu_pd(b2 + j);
        __m256d v3 = _mm256_loadu_pd(b3 + j);
        __m256d v4 = _mm256_loadu_pd(b4 + j);
        __m256d v5 = _mm256_loadu_pd(b5 + j);
        __m256d v6 = _mm256_loadu_pd(b6 + j);
        __m256d v7 = _mm256_loadu_pd(b7 + j);

        __m256d sum01 = _mm256_add_pd(v0, v1);
        __m256d sum23 = _mm256_add_pd(v2, v3);
        __m256d sum45 = _mm256_add_pd(v4, v5);
        __m256d sum67 = _mm256_add_pd(v6, v7);

        __m256d leftv  = _mm256_add_pd(sum01, sum23);
        __m256d rightv = _mm256_add_pd(sum45, sum67);

        _mm256_storeu_pd(out + j, _mm256_add_pd(leftv, rightv));
    }
}

__attribute__((target("avx2")))
static void reduce8_L128_avx2(const double* data, double* out) {
    constexpr size_t L = 128;
    const double* b0 = data + 0*L; const double* b1 = data + 1*L;
    const double* b2 = data + 2*L; const double* b3 = data + 3*L;
    const double* b4 = data + 4*L; const double* b5 = data + 5*L;
    const double* b6 = data + 6*L; const double* b7 = data + 7*L;
    for (size_t j = 0; j < L; j += 4) {
        __m256d v0 = _mm256_loadu_pd(b0 + j);
        __m256d v1 = _mm256_loadu_pd(b1 + j);
        __m256d v2 = _mm256_loadu_pd(b2 + j);
        __m256d v3 = _mm256_loadu_pd(b3 + j);
        __m256d v4 = _mm256_loadu_pd(b4 + j);
        __m256d v5 = _mm256_loadu_pd(b5 + j);
        __m256d v6 = _mm256_loadu_pd(b6 + j);
        __m256d v7 = _mm256_loadu_pd(b7 + j);

        __m256d sum01 = _mm256_add_pd(v0, v1);
        __m256d sum23 = _mm256_add_pd(v2, v3);
        __m256d sum45 = _mm256_add_pd(v4, v5);
        __m256d sum67 = _mm256_add_pd(v6, v7);

        __m256d leftv  = _mm256_add_pd(sum01, sum23);
        __m256d rightv = _mm256_add_pd(sum45, sum67);

        _mm256_storeu_pd(out + j, _mm256_add_pd(leftv, rightv));
    }
}
#endif

// -------------------- AVX512 (FIXED: avx512f only) --------------------
#if HAS_X86
__attribute__((target("avx512f")))
static void combine_avx512(const double* left, size_t lc,
                           const double* right, size_t rc,
                           double* out, size_t* outc) {
    const size_t common = (lc < rc) ? lc : rc;
    *outc = (lc > rc) ? lc : rc;

    size_t j = 0;
    for (; j + 8 <= common; j += 8) {
        __m512d a = _mm512_loadu_pd(left + j);
        __m512d b = _mm512_loadu_pd(right + j);
        _mm512_storeu_pd(out + j, _mm512_add_pd(a, b));
    }
    for (; j < common; ++j) out[j] = left[j] + right[j];

    if (lc > common) for (size_t k = common; k < lc; ++k) out[k] = left[k];
    else             for (size_t k = common; k < rc; ++k) out[k] = right[k];
}

__attribute__((target("avx512f")))
static void reduce8_L16_avx512(const double* data, double* out) {
    constexpr size_t L = 16;
    const double* b0 = data + 0*L; const double* b1 = data + 1*L;
    const double* b2 = data + 2*L; const double* b3 = data + 3*L;
    const double* b4 = data + 4*L; const double* b5 = data + 5*L;
    const double* b6 = data + 6*L; const double* b7 = data + 7*L;
    for (size_t j = 0; j < L; j += 8) {
        __m512d v0 = _mm512_loadu_pd(b0 + j);
        __m512d v1 = _mm512_loadu_pd(b1 + j);
        __m512d v2 = _mm512_loadu_pd(b2 + j);
        __m512d v3 = _mm512_loadu_pd(b3 + j);
        __m512d v4 = _mm512_loadu_pd(b4 + j);
        __m512d v5 = _mm512_loadu_pd(b5 + j);
        __m512d v6 = _mm512_loadu_pd(b6 + j);
        __m512d v7 = _mm512_loadu_pd(b7 + j);

        __m512d sum01 = _mm512_add_pd(v0, v1);
        __m512d sum23 = _mm512_add_pd(v2, v3);
        __m512d sum45 = _mm512_add_pd(v4, v5);
        __m512d sum67 = _mm512_add_pd(v6, v7);

        __m512d leftv  = _mm512_add_pd(sum01, sum23);
        __m512d rightv = _mm512_add_pd(sum45, sum67);

        _mm512_storeu_pd(out + j, _mm512_add_pd(leftv, rightv));
    }
}

__attribute__((target("avx512f")))
static void reduce8_L128_avx512(const double* data, double* out) {
    constexpr size_t L = 128;
    const double* b0 = data + 0*L; const double* b1 = data + 1*L;
    const double* b2 = data + 2*L; const double* b3 = data + 3*L;
    const double* b4 = data + 4*L; const double* b5 = data + 5*L;
    const double* b6 = data + 6*L; const double* b7 = data + 7*L;
    for (size_t j = 0; j < L; j += 8) {
        __m512d v0 = _mm512_loadu_pd(b0 + j);
        __m512d v1 = _mm512_loadu_pd(b1 + j);
        __m512d v2 = _mm512_loadu_pd(b2 + j);
        __m512d v3 = _mm512_loadu_pd(b3 + j);
        __m512d v4 = _mm512_loadu_pd(b4 + j);
        __m512d v5 = _mm512_loadu_pd(b5 + j);
        __m512d v6 = _mm512_loadu_pd(b6 + j);
        __m512d v7 = _mm512_loadu_pd(b7 + j);

        __m512d sum01 = _mm512_add_pd(v0, v1);
        __m512d sum23 = _mm512_add_pd(v2, v3);
        __m512d sum45 = _mm512_add_pd(v4, v5);
        __m512d sum67 = _mm512_add_pd(v6, v7);

        __m512d leftv  = _mm512_add_pd(sum01, sum23);
        __m512d rightv = _mm512_add_pd(sum45, sum67);

        _mm512_storeu_pd(out + j, _mm512_add_pd(leftv, rightv));
    }
}
#endif

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
    return OPS_SCALAR;
#else
    return OPS_SCALAR;
#endif
}

static void print_cpu_support() {
#if HAS_X86 && (defined(__clang__) || defined(__GNUC__))
    std::printf("Runtime CPU support:\n");
    std::printf("  avx512f = %d\n", __builtin_cpu_supports("avx512f") ? 1 : 0);
    std::printf("  avx2    = %d\n", __builtin_cpu_supports("avx2")    ? 1 : 0);
    std::printf("  sse2    = %d\n", __builtin_cpu_supports("sse2")    ? 1 : 0);
#else
    std::printf("Runtime CPU support: (unknown)\n");
#endif
    std::printf("\n");
}

// ============================================================================
// Deterministic reduce specialized for L=16 and L=128, parameterized by Ops.
// Canonical merge order: older partial on the left (stable in time / index).
// Uses a binary-counter stack over blocks, with 8-block pre-reduction at level 3.
//
// Robustness guard:
//  - prevents UB shift (1u<<level) and stack OOB if level >= MAX_DEPTH / 32.
//  - if it ever triggers, we "saturate" merges into the top bucket (MAX_DEPTH-1).
//    (For intended demo N this never triggers.)
// ============================================================================

static inline bool depth_overflow(size_t level, size_t max_depth) {
    return (level >= max_depth) || (level >= 32); // 32-bit mask limitation + shift UB
}

static double deterministic_reduce_L16(const Ops& ops,
                                      const double* __restrict__ first, size_t N, double init) {
    constexpr size_t L = 16;
    constexpr size_t MAX_DEPTH = 32;
    constexpr size_t GROUP_SIZE = 8;
    constexpr size_t GROUP_LEVEL = 3;

    if (N == 0) return init;
    const size_t num_blocks = (N + L - 1) / L;

    if (num_blocks == 1) {
        alignas(64) double lanes[L] = {};
        for (size_t j = 0; j < N; ++j) lanes[j] = first[j];
        return init + balanced_tree_reduce_scalars(lanes, N);
    }

    alignas(64) double stack[MAX_DEPTH][L];
    size_t stack_counts[MAX_DEPTH] = {};
    uint32_t valid_mask = 0;

    // ping-pong buffers (avoid temp+copy loop)
    alignas(64) double buf0[L];
    alignas(64) double buf1[L];
    double* current = buf0;
    double* temp    = buf1;
    size_t current_count = 0;

    auto merge_into_top = [&](const double* rhs, size_t rhs_count) {
        // Ensure top bucket is populated and then keep merging into it.
        constexpr size_t top = MAX_DEPTH - 1;
        if (!(valid_mask & (1u << top))) {
            for (size_t j = 0; j < rhs_count; ++j) stack[top][j] = rhs[j];
            stack_counts[top] = rhs_count;
            valid_mask |= (1u << top);
            return;
        }
        size_t outc = 0;
        ops.combine(stack[top], stack_counts[top], rhs, rhs_count, stack[top], &outc);
        stack_counts[top] = outc;
        valid_mask |= (1u << top);
    };

    const size_t full_blocks = N / L;
    const size_t unrollable_groups = full_blocks / GROUP_SIZE;

    for (size_t g = 0; g < unrollable_groups; ++g) {
        const double* group_start = first + g * GROUP_SIZE * L;

        ops.reduce8_L16(group_start, current);
        current_count = L;

        size_t level = GROUP_LEVEL;
        for (;;) {
            if (depth_overflow(level, MAX_DEPTH)) {
                merge_into_top(current, current_count);
                current_count = 0;
                break;
            }
            const uint32_t bit = (1u << (uint32_t)level);
            if (!(valid_mask & bit)) break;

            size_t temp_count = 0;
            ops.combine(stack[level], stack_counts[level],
                        current, current_count,
                        temp, &temp_count);

            valid_mask &= ~bit;
            current = temp; temp = (current == buf0) ? buf1 : buf0;
            current_count = temp_count;
            ++level;
        }

        if (current_count != 0) {
            if (depth_overflow(level, MAX_DEPTH)) {
                merge_into_top(current, current_count);
            } else {
                for (size_t j = 0; j < current_count; ++j) stack[level][j] = current[j];
                stack_counts[level] = current_count;
                valid_mask |= (1u << (uint32_t)level);
            }
        }
    }

    const size_t start_block = unrollable_groups * GROUP_SIZE;

    for (size_t b = start_block; b < num_blocks; ++b) {
        const size_t elem_start = b * L;
        const size_t elem_end   = (elem_start + L < N) ? (elem_start + L) : N;
        current_count = elem_end - elem_start;

        const double* src = first + elem_start;
        for (size_t j = 0; j < current_count; ++j) current[j] = src[j];
        // NOTE: removed zero-fill tail; combine respects counts.

        size_t level = 0;
        for (;;) {
            if (depth_overflow(level, MAX_DEPTH)) {
                merge_into_top(current, current_count);
                current_count = 0;
                break;
            }
            const uint32_t bit = (1u << (uint32_t)level);
            if (!(valid_mask & bit)) break;

            size_t temp_count = 0;
            ops.combine(stack[level], stack_counts[level],
                        current, current_count,
                        temp, &temp_count);

            valid_mask &= ~bit;
            current = temp; temp = (current == buf0) ? buf1 : buf0;
            current_count = temp_count;
            ++level;
        }

        if (current_count != 0) {
            if (depth_overflow(level, MAX_DEPTH)) {
                merge_into_top(current, current_count);
            } else {
                for (size_t j = 0; j < current_count; ++j) stack[level][j] = current[j];
                stack_counts[level] = current_count;
                valid_mask |= (1u << (uint32_t)level);
            }
        }
    }

    // Final fold: increasing level, always "older" on left
    bool have = false;
    for (size_t level = 0; level < MAX_DEPTH; ++level) {
        const uint32_t bit = (1u << (uint32_t)level);
        if (valid_mask & bit) {
            if (!have) {
                have = true;
                for (size_t j = 0; j < stack_counts[level]; ++j) current[j] = stack[level][j];
                current_count = stack_counts[level];
            } else {
                size_t temp_count = 0;
                ops.combine(stack[level], stack_counts[level],
                            current, current_count,
                            temp, &temp_count);

                current = temp; temp = (current == buf0) ? buf1 : buf0;
                current_count = temp_count;
            }
        }
    }

    return init + balanced_tree_reduce_scalars(current, current_count);
}

static double deterministic_reduce_L128(const Ops& ops,
                                       const double* __restrict__ first, size_t N, double init) {
    constexpr size_t L = 128;
    constexpr size_t MAX_DEPTH = 32;
    constexpr size_t GROUP_SIZE = 8;
    constexpr size_t GROUP_LEVEL = 3;

    if (N == 0) return init;
    const size_t num_blocks = (N + L - 1) / L;

    if (num_blocks == 1) {
        alignas(64) double lanes[L] = {};
        for (size_t j = 0; j < N; ++j) lanes[j] = first[j];
        return init + balanced_tree_reduce_scalars(lanes, N);
    }

    alignas(64) double stack[MAX_DEPTH][L];
    size_t stack_counts[MAX_DEPTH] = {};
    uint32_t valid_mask = 0;

    // ping-pong buffers (avoid temp+copy loop)
    alignas(64) double buf0[L];
    alignas(64) double buf1[L];
    double* current = buf0;
    double* temp    = buf1;
    size_t current_count = 0;

    auto merge_into_top = [&](const double* rhs, size_t rhs_count) {
        constexpr size_t top = MAX_DEPTH - 1;
        if (!(valid_mask & (1u << top))) {
            for (size_t j = 0; j < rhs_count; ++j) stack[top][j] = rhs[j];
            stack_counts[top] = rhs_count;
            valid_mask |= (1u << top);
            return;
        }
        size_t outc = 0;
        ops.combine(stack[top], stack_counts[top], rhs, rhs_count, stack[top], &outc);
        stack_counts[top] = outc;
        valid_mask |= (1u << top);
    };

    const size_t full_blocks = N / L;
    const size_t unrollable_groups = full_blocks / GROUP_SIZE;

    for (size_t g = 0; g < unrollable_groups; ++g) {
        const double* group_start = first + g * GROUP_SIZE * L;

        ops.reduce8_L128(group_start, current);
        current_count = L;

        size_t level = GROUP_LEVEL;
        for (;;) {
            if (depth_overflow(level, MAX_DEPTH)) {
                merge_into_top(current, current_count);
                current_count = 0;
                break;
            }
            const uint32_t bit = (1u << (uint32_t)level);
            if (!(valid_mask & bit)) break;

            size_t temp_count = 0;
            ops.combine(stack[level], stack_counts[level],
                        current, current_count,
                        temp, &temp_count);

            valid_mask &= ~bit;
            current = temp; temp = (current == buf0) ? buf1 : buf0;
            current_count = temp_count;
            ++level;
        }

        if (current_count != 0) {
            if (depth_overflow(level, MAX_DEPTH)) {
                merge_into_top(current, current_count);
            } else {
                for (size_t j = 0; j < current_count; ++j) stack[level][j] = current[j];
                stack_counts[level] = current_count;
                valid_mask |= (1u << (uint32_t)level);
            }
        }
    }

    const size_t start_block = unrollable_groups * GROUP_SIZE;

    for (size_t b = start_block; b < num_blocks; ++b) {
        const size_t elem_start = b * L;
        const size_t elem_end   = (elem_start + L < N) ? (elem_start + L) : N;
        current_count = elem_end - elem_start;

        const double* src = first + elem_start;
        for (size_t j = 0; j < current_count; ++j) current[j] = src[j];
        // NOTE: removed zero-fill tail; combine respects counts.

        size_t level = 0;
        for (;;) {
            if (depth_overflow(level, MAX_DEPTH)) {
                merge_into_top(current, current_count);
                current_count = 0;
                break;
            }
            const uint32_t bit = (1u << (uint32_t)level);
            if (!(valid_mask & bit)) break;

            size_t temp_count = 0;
            ops.combine(stack[level], stack_counts[level],
                        current, current_count,
                        temp, &temp_count);

            valid_mask &= ~bit;
            current = temp; temp = (current == buf0) ? buf1 : buf0;
            current_count = temp_count;
            ++level;
        }

        if (current_count != 0) {
            if (depth_overflow(level, MAX_DEPTH)) {
                merge_into_top(current, current_count);
            } else {
                for (size_t j = 0; j < current_count; ++j) stack[level][j] = current[j];
                stack_counts[level] = current_count;
                valid_mask |= (1u << (uint32_t)level);
            }
        }
    }

    // Final fold: increasing level, always "older" on left
    bool have = false;
    for (size_t level = 0; level < MAX_DEPTH; ++level) {
        const uint32_t bit = (1u << (uint32_t)level);
        if (valid_mask & bit) {
            if (!have) {
                have = true;
                for (size_t j = 0; j < stack_counts[level]; ++j) current[j] = stack[level][j];
                current_count = stack_counts[level];
            } else {
                size_t temp_count = 0;
                ops.combine(stack[level], stack_counts[level],
                            current, current_count,
                            temp, &temp_count);

                current = temp; temp = (current == buf0) ? buf1 : buf0;
                current_count = temp_count;
            }
        }
    }

    return init + balanced_tree_reduce_scalars(current, current_count);
}

// ============================================================================
// Golden constants
// ============================================================================
constexpr uint64_t SEED = 0x243F6A8885A308D3ULL;
constexpr size_t N = 1'000'000;

const char* expected_data[] = {
    "0x3fd37de3b20e9fdc", "0xbfd2e1595e76077c", "0xbfd5c999955b530c",
    "0xbfe6be1806d7224e", "0x3fef95133e17376e"
};

const char* EXPECTED_NARROW = "0x40618f71f6379380";
const char* EXPECTED_WIDE   = "0x40618f71f6379397";

// ============================================================================
// Dataset generator for arbitrary lengths (base + hostile)
// ============================================================================
static std::vector<double> make_data(size_t n, uint64_t seed, bool hostile) {
    std::vector<double> x(n);
    CrossPlatformRNG rng(seed);
    for (size_t i = 0; i < n; ++i) x[i] = rng.next_double();

    if (hostile) {
        inject_cancellation_prime_stride(x, 7, 11, 1e16);
        inject_cluster(x, 1e16);
    }
    return x;
}

// ============================================================================
// Length sweep reproducibility table
// ============================================================================
static void run_length_sweep(const Ops& ops, size_t L, bool hostile) {
    auto det = [&](const std::vector<double>& x) -> double {
        return (L == 16)
            ? deterministic_reduce_L16(ops, x.data(), x.size(), 0.0)
            : deterministic_reduce_L128(ops, x.data(), x.size(), 0.0);
    };

    auto run_one = [&](size_t n) {
        auto x = make_data(n, SEED, hostile);

        double d0 = det(x);
        std::string d0h = to_hex(d0);

        bool stable = true;
        for (int r = 0; r < 3; ++r) {
            if (to_hex(det(x)) != d0h) stable = false;
        }

        double acc = std::accumulate(x.begin(), x.end(), 0.0);
        double red = std::reduce(x.begin(), x.end(), 0.0);

        std::printf("  N=%-6zu det=%s %s  acc=%s  red=%s\n",
                    n,
                    d0h.c_str(),
                    stable ? "✓" : "✗",
                    to_hex(acc).c_str(),
                    to_hex(red).c_str());
    };

    std::printf("L=%zu  dataset=%s\n",
                L,
                hostile ? "HOSTILE (+/-1e16 primes 7,11 + cluster)" : "BASE (RNG only)");

    const size_t fixed[] = {0,1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65};
    for (size_t n : fixed) run_one(n);

    for (int d = -3; d <= 3; ++d) {
        long n = (long)L + d;
        if (n >= 0) run_one((size_t)n);
    }

    for (size_t k : {1UL, 2UL, 3UL, 4UL}) {
        if (k*L > 0) run_one(k*L - 1);
        run_one(k*L);
        run_one(k*L + 1);
    }

    std::printf("\n");
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::printf("════════════════════════════════════════════════════════════════\n");
    std::printf("  Deterministic Parallel Reduction (x86)\n");
    std::printf("  Unrolled + multiversion SIMD (Exact Tree Order Preserved)\n");
    std::printf("════════════════════════════════════════════════════════════════\n\n");

    const Ops& ops = select_ops();

    std::printf("Platform: x86-64\n");
    std::printf("Compiled variants: SCALAR");
#if HAS_X86
    std::printf(" + SSE2 + AVX2 + AVX512");
#endif
    std::printf("\nSelected ISA: %s\n\n", ops.name);

    print_cpu_support();

    std::printf("N = %zu elements\n", N);
    std::printf("NARROW: M=128,  L=16\n");
    std::printf("WIDE:   M=1024, L=128\n\n");

    std::vector<double> data(N);
    CrossPlatformRNG rng(SEED);
    for (size_t i = 0; i < N; ++i) data[i] = rng.next_double();

    std::printf("Data verification:\n");
    bool data_ok = true;
    for (int i = 0; i < 5; ++i) {
        std::string hex = to_hex(data[(size_t)i]);
        bool ok = (hex == expected_data[i]);
        data_ok &= ok;
        std::printf("  data[%d] = %s  %s\n", i, hex.c_str(), ok ? "✓" : "✗");
    }

    std::printf("\nCORRECTNESS:\n");
    double narrow = deterministic_reduce_L16(ops, data.data(), N, 0.0);
    double wide   = deterministic_reduce_L128(ops, data.data(), N, 0.0);

    std::string narrow_hex = to_hex(narrow);
    std::string wide_hex   = to_hex(wide);

    bool narrow_ok = (narrow_hex == EXPECTED_NARROW);
    bool wide_ok   = (wide_hex   == EXPECTED_WIDE);

    std::printf("  NARROW: %s  %s (expected %s)\n", narrow_hex.c_str(), narrow_ok ? "✓" : "✗", EXPECTED_NARROW);
    std::printf("  WIDE:   %s  %s (expected %s)\n", wide_hex.c_str(),   wide_ok ? "✓" : "✗", EXPECTED_WIDE);

    std::printf("\nRUN-TO-RUN STABILITY:\n");
    bool n_stable = true, w_stable = true;
    for (int r = 0; r < 5; ++r) {
        if (to_hex(deterministic_reduce_L16(ops, data.data(), N, 0.0)) != narrow_hex) n_stable = false;
        if (to_hex(deterministic_reduce_L128(ops, data.data(), N, 0.0)) != wide_hex)  w_stable = false;
    }
    std::printf("  NARROW: %s\n", n_stable ? "STABLE ✓" : "UNSTABLE ✗");
    std::printf("  WIDE:   %s\n", w_stable ? "STABLE ✓" : "UNSTABLE ✗");

    std::printf("\nTRANSLATIONAL INVARIANCE:\n");
    bool n_trans = true, w_trans = true;
    for (size_t off : {0UL, 1UL, 7UL, 15UL, 31UL, 63UL}) {
        std::vector<double> buf(N + off + 128);
        std::copy(data.begin(), data.end(), buf.data() + off);

        if (to_hex(deterministic_reduce_L16(ops, buf.data() + off, N, 0.0)) != narrow_hex) n_trans = false;
        if (to_hex(deterministic_reduce_L128(ops, buf.data() + off, N, 0.0)) != wide_hex) w_trans = false;
    }
    std::printf("  NARROW: %s\n", n_trans ? "INVARIANT ✓" : "VARIES ✗");
    std::printf("  WIDE:   %s\n", w_trans ? "INVARIANT ✓" : "VARIES ✗");

    std::printf("\nLENGTH-SWEEP REPRODUCIBILITY (BASE + HOSTILE):\n\n");
    run_length_sweep(ops, 16,  false);
    run_length_sweep(ops, 16,  true);
    run_length_sweep(ops, 128, false);
    run_length_sweep(ops, 128, true);

    std::printf("PERFORMANCE (CE timings vary; best-of-trials):\n\n");

    double t_acc = bench("std::accumulate", N, [&]() {
        return std::accumulate(data.begin(), data.end(), 0.0);
    });

    double t_reduce_np = bench("std::reduce (no policy)", N, [&]() {
        return std::reduce(data.begin(), data.end(), 0.0);
    });

#if HAS_EXECUTION
    double t_seq = bench("std::reduce(seq)", N, [&]() {
        return std::reduce(std::execution::seq, data.begin(), data.end(), 0.0);
    });

    double t_unseq = bench("std::reduce(unseq)", N, [&]() {
        return std::reduce(std::execution::unseq, data.begin(), data.end(), 0.0);
    });

#if !defined(NO_PAR_POLICIES)
    double t_par = bench("std::reduce(par)", N, [&]() {
        return std::reduce(std::execution::par, data.begin(), data.end(), 0.0);
    });

    double t_par_unseq = bench("std::reduce(par_unseq)", N, [&]() {
        return std::reduce(std::execution::par_unseq, data.begin(), data.end(), 0.0);
    });

    (void)t_par; (void)t_par_unseq;
#endif
    (void)t_seq; (void)t_unseq;
#endif

    double t_det_n = bench("deterministic_reduce NARROW (M=128)", N, [&]() {
        return deterministic_reduce_L16(ops, data.data(), N, 0.0);
    });

    double t_det_w = bench("deterministic_reduce WIDE   (M=1024)", N, [&]() {
        return deterministic_reduce_L128(ops, data.data(), N, 0.0);
    });

    std::printf("\nOverhead vs std::accumulate:\n");
    std::printf("  NARROW: %+.1f%%\n", (t_det_n / t_acc - 1.0) * 100.0);
    std::printf("  WIDE:   %+.1f%%\n", (t_det_w / t_acc - 1.0) * 100.0);

    std::printf("\n════════════════════════════════════════════════════════════════\n");
    std::printf("VERIFICATION BLOCK\n");
    std::printf("════════════════════════════════════════════════════════════════\n");
    std::printf("Platform: x86-64\n");
    std::printf("Selected: %s\n", ops.name);
    std::printf("SEED:     0x%016llx\n", (unsigned long long)SEED);
    std::printf("N:        %zu\n", N);
    std::printf("NARROW:   %s  (M=128,  L=16)\n", narrow_hex.c_str());
    std::printf("WIDE:     %s  (M=1024, L=128)\n", wide_hex.c_str());
    std::printf("════════════════════════════════════════════════════════════════\n");

    bool all_pass = data_ok && narrow_ok && wide_ok && n_stable && w_stable && n_trans && w_trans;
    std::printf("\nOverall: %s\n", all_pass ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");

    do_not_optimize(narrow);
    do_not_optimize(wide);
    do_not_optimize(t_acc);
    do_not_optimize(t_reduce_np);
    return all_pass ? 0 : 1;
}
