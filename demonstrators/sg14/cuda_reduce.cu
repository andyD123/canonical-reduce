// ============================================================================
// Canonical Reduce - CUDA FAST (no workspace, single Phase2) + CUB comparison
// - Fixes CUDA Phase2 shared-memory race (ping-pong buffering)
// - FINAL fold (cross-lane) is now CANONICAL (no rev-final special-case)
//     => consistent with spec + x86/ARM demonstrators
// - Adds reproducibility stress tests:
//     * length-sweep across many N (incl. around L and around block boundaries)
//     * hostile high-cancellation overlay (±1e16 at prime strides 7/11 + cluster)
// - Benchmarks against CUB DeviceReduce::Sum (non-deterministic for FP)
//
// Source: godbolt.org/z/x58GzE73q
// Godbolt CUDA flags (suggested): -O3 -std=c++17 --fmad=false
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>

#ifndef NO_CUB
#include <cub/cub.cuh>
#define HAS_CUB 1
#else
#define HAS_CUB 0
#endif

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if(_e != cudaSuccess) { \
        printf("CUDA err: %s\n", cudaGetErrorString(_e)); \
        return 1; \
    } \
} while(0)

struct RNG {
    uint64_t s;
    RNG(uint64_t seed) : s(seed) {}
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return ((int64_t)(s >> 11) - (1LL << 52)) / (double)(1LL << 52);
    }
};

static std::string hex(double d) {
    uint64_t b; std::memcpy(&b, &d, 8);
    char buf[24]; std::snprintf(buf, 24, "0x%016llx", (unsigned long long)b);
    return buf;
}

// Canonical tree_host: iterative pairwise with carry if odd.
static double tree_host(std::vector<double> a) {
    size_t n = a.size();
    if (n == 0) return 0.0;
    while (n > 1) {
        size_t h = n / 2;
        for (size_t i = 0; i < h; ++i) a[i] = a[2*i] + a[2*i+1];
        if (n & 1) { a[h] = a[n-1]; n = h + 1; } else n = h;
    }
    return a[0];
}

// Golden reference: per-lane is canonical tree_host; FINAL fold (cross-lane)
// is now ALSO canonical tree_host over results[0..L).
static double canonical_host(const std::vector<double>& data, int L) {
    size_t N = data.size();
    std::vector<std::vector<double>> lanes(L);
    for (size_t i = 0; i < N; ++i) lanes[i % (size_t)L].push_back(data[i]);
    std::vector<double> results(L);
    for (int j = 0; j < L; ++j) results[j] = tree_host(lanes[j]);
    return tree_host(results);
}

// ============================================================================
// Host-side stress data overlays (deterministic)
// ============================================================================

static std::vector<double> make_base_data(size_t N, uint64_t seed) {
    std::vector<double> x(N);
    RNG rng(seed);
    for (size_t i = 0; i < N; ++i) x[i] = rng.next();
    return x;
}

// Prime-stride cancellation: deterministic, explainable.
static void inject_cancellation_prime_stride(std::vector<double>& x,
                                             size_t p1 = 7, size_t p2 = 11,
                                             double A = 1e16) {
    for (size_t i = 0; i < x.size(); ++i) {
        if ((i % p1) == 0) x[i] += A;
        if ((i % p2) == 0) x[i] -= A;
    }
}

// Cluster cancellation: brutal for grouping order.
static void inject_cluster(std::vector<double>& x, double A = 1e16) {
    if (x.size() < 8) return;
    size_t m = x.size() / 2;
    x[m + 0] += A;
    x[m + 1] += A;
    x[m + 2] -= A;
    x[m + 3] -= A;
}

// ============================================================================
// atomicAdd for double fallback
// ============================================================================
__device__ __forceinline__ double atomicAdd_double(double* addr, double val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return atomicAdd(addr, val);
#else
    unsigned long long* ull = (unsigned long long*)addr;
    unsigned long long old = *ull, assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double((long long)assumed) + val;
        old = atomicCAS(ull, assumed, (unsigned long long)__double_as_longlong(sum));
    } while (assumed != old);
    return __longlong_as_double((long long)old);
#endif
}

// ============================================================================
// Device: exact tree_host semantics for n<=16 (tail segments)
// ============================================================================
__device__ __forceinline__ double tree16_exact(double tmp[16], int n) {
    while (n > 1) {
        int h = n / 2;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < h) tmp[i] = tmp[2*i] + tmp[2*i + 1];
        }
        if (n & 1) tmp[h] = tmp[n - 1];
        n = h + (n & 1);
    }
    return tmp[0];
}

// ============================================================================
// Phase 1 (FAST, no workspace, exact structure alignment)
// CHUNK_SIZE=4096 => per-lane chunk sizes are pow2 for L=16,128.
// ============================================================================
template<int L>
__global__ void phase1_kernel_fast(
    const double* __restrict__ data,
    double* __restrict__ block_lanes,   // [num_blocks][L]
    int N)
{
    constexpr int BLOCK_THREADS = 256;
    constexpr int CHUNK_SIZE = 4096;
    static_assert(CHUNK_SIZE % L == 0, "CHUNK_SIZE must be divisible by L");

    constexpr int MAX_PER_LANE = CHUNK_SIZE / L;
    constexpr int THREADS_PER_LANE = BLOCK_THREADS / L;
    static_assert((MAX_PER_LANE & (MAX_PER_LANE - 1)) == 0, "MAX_PER_LANE must be power-of-two");
    static_assert((THREADS_PER_LANE & (THREADS_PER_LANE - 1)) == 0, "THREADS_PER_LANE must be power-of-two");
    constexpr int ELEMS_PER_THREAD = MAX_PER_LANE / THREADS_PER_LANE;
    static_assert(ELEMS_PER_THREAD == 16, "This kernel expects ELEMS_PER_THREAD==16 for L=16 or 128");

    int tid = threadIdx.x;
    int lane = tid % L;
    int lane_local = tid / L;

    int bid = blockIdx.x;
    int chunk_start = bid * CHUNK_SIZE;
    int chunk_end   = min(chunk_start + CHUNK_SIZE, N);

    int base_p = lane_local * 16;

    double vals[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        int p   = base_p + t;
        int idx = chunk_start + lane + p * L;
        vals[t] = (idx < chunk_end) ? data[idx] : __longlong_as_double(0x7ff8000000000000ULL);
    }

    int active = 0;
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        if (vals[t] == vals[t]) active++;
        else break;
    }

    double seg_sum;
    if (active == 16) {
        double tmp[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) tmp[i] = vals[i];
        #pragma unroll
        for (int stride = 1; stride < 16; stride <<= 1) {
            #pragma unroll
            for (int i = 0; i < 16; i += 2*stride) {
                tmp[i] = tmp[i] + tmp[i + stride];
            }
        }
        seg_sum = tmp[0];
    } else if (active > 0) {
        double tmp[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) tmp[i] = (i < active) ? vals[i] : 0.0;
        seg_sum = tree16_exact(tmp, active);
    } else {
        seg_sum = __longlong_as_double(0x7ff8000000000000ULL);
    }

    __shared__ double segs[256];
    segs[lane_local * L + lane] = seg_sum;
    __syncthreads();

    if (lane_local == 0) {
        double tmp[16];
        int seg_active = 0;

        #pragma unroll
        for (int s = 0; s < THREADS_PER_LANE; ++s) {
            double v = segs[s * L + lane];
            if (v == v) tmp[seg_active++] = v;
            else break;
        }

        double lane_sum = 0.0;
        if (seg_active > 0) {
            int n = seg_active;
            while (n > 1) {
                int h = n / 2;
                for (int i = 0; i < h; ++i) tmp[i] = tmp[2*i] + tmp[2*i + 1];
                if (n & 1) tmp[h] = tmp[n - 1];
                n = h + (n & 1);
            }
            lane_sum = tmp[0];
        } else {
            lane_sum = 0.0;
        }

        block_lanes[(size_t)bid * L + lane] = lane_sum;
    }
}

// ============================================================================
// Phase 2 SINGLE kernel (FIXED): reduce across blocks per lane
// ============================================================================
template<int L>
__global__ void phase2_single_kernel(
    const double* __restrict__ block_lanes,
    double* __restrict__ final_lanes,
    int num_blocks)
{
    int lane = blockIdx.x;
    int tid  = threadIdx.x;

    extern __shared__ double sh[];
    double* src = sh;
    double* dst = sh + num_blocks;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        src[i] = block_lanes[(size_t)i * L + lane];
    }
    __syncthreads();

    int n = num_blocks;
    while (n > 1) {
        int h = n / 2;
        for (int i = tid; i < h; i += blockDim.x) {
            dst[i] = src[2*i] + src[2*i + 1];
        }
        if ((n & 1) && tid == 0) {
            dst[h] = src[n - 1];
        }
        __syncthreads();
        double* tmp = src; src = dst; dst = tmp;
        n = h + (n & 1);
    }

    if (tid == 0) final_lanes[lane] = (num_blocks > 0) ? src[0] : 0.0;
}

// ============================================================================
// Phase 3 (CANONICAL): final fold over L lanes
// ============================================================================
template<int L>
__global__ void phase3_kernel(const double* __restrict__ final_lanes, double* __restrict__ out)
{
    __shared__ double sh[128];
    int tid = threadIdx.x;

    if (tid < L) sh[tid] = final_lanes[tid];
    __syncthreads();

    if (tid == 0) {
        int n = L;
        if (n == 0) { *out = 0.0; return; }
        if (n == 1) { *out = sh[0]; return; }

        while (n > 1) {
            int h = n / 2;
            for (int i = 0; i < h; ++i) sh[i] = sh[2*i] + sh[2*i + 1];
            if (n & 1) sh[h] = sh[n - 1];
            n = h + (n & 1);
        }
        *out = sh[0];
    }
}

// ============================================================================
// Fast non-deterministic baseline (atomic)
// ============================================================================
__global__ void fast_reduce_kernel(const double* data, double* out, int N) {
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = gid; i < N; i += stride) sum += data[i];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd_double(out, sdata[0]);
}

// ============================================================================
// Host wrappers
// ============================================================================
template<int L>
double run_canonical_fast(const double* d_data,
                          double* d_block_lanes,
                          double* d_final_lanes,
                          double* d_out,
                          int N)
{
    constexpr int CHUNK_SIZE = 4096;
    int num_blocks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

    phase1_kernel_fast<L><<<num_blocks, 256>>>(d_data, d_block_lanes, N);

    size_t sh_bytes = (size_t)num_blocks * sizeof(double) * 2;
    phase2_single_kernel<L><<<L, 256, sh_bytes>>>(d_block_lanes, d_final_lanes, num_blocks);

    phase3_kernel<L><<<1, 128>>>(d_final_lanes, d_out);

    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

double run_fast(const double* d_data, double* d_out, int N) {
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(double)));
    fast_reduce_kernel<<<128, 256>>>(d_data, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

#if HAS_CUB
double run_cub(const double* d_data, double* d_out, void* d_temp, size_t temp_bytes, int N) {
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    double result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}
#endif

// ============================================================================
// Benchmark helper
// ============================================================================
template<typename F>
double bench(const char* name, int N, int iters, F&& fn) {
    for (int i = 0; i < 3; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double avg_ms = ms / iters;
    double gb_s = (N * sizeof(double) / 1e9) / (avg_ms / 1000.0);

    printf("  %-40s %8.3f ms  %7.2f GB/s\n", name, avg_ms, gb_s);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return avg_ms;
}

// ============================================================================
// Reproducibility stress tests
// ============================================================================
static bool run_length_sweep_cuda(double* d_data,
                                 double* d_block_lanes,
                                 double* d_final_lanes,
                                 double* d_out,
#if HAS_CUB
                                 void* d_temp, size_t temp_bytes,
#endif
                                 uint64_t seed)
{
    auto run_one = [&](int L, size_t N, bool hostile) -> bool {
        auto h = make_base_data(N, seed);
        if (hostile) {
            inject_cancellation_prime_stride(h, 7, 11, 1e16);
            inject_cluster(h, 1e16);
        }

        CUDA_CHECK(cudaMemcpy(d_data, h.data(), N * sizeof(double), cudaMemcpyHostToDevice));

        double host = canonical_host(h, L);
        double gpu  = (L == 16)
            ? run_canonical_fast<16>(d_data, d_block_lanes, d_final_lanes, d_out, (int)N)
            : run_canonical_fast<128>(d_data, d_block_lanes, d_final_lanes, d_out, (int)N);

        std::string hs = hex(host), gs = hex(gpu);

        bool stable = true;
        for (int r = 0; r < 3; ++r) {
            double gpu2 = (L == 16)
                ? run_canonical_fast<16>(d_data, d_block_lanes, d_final_lanes, d_out, (int)N)
                : run_canonical_fast<128>(d_data, d_block_lanes, d_final_lanes, d_out, (int)N);
            if (hex(gpu2) != gs) stable = false;
        }

        bool match = (hs == gs);
        printf("  L=%-3d N=%-7zu %-8s  host=%s  gpu=%s  %s  %s\n",
               L, N,
               hostile ? "HOSTILE" : "BASE",
               hs.c_str(), gs.c_str(),
               match ? "\xe2\x9c\x93 MATCH" : "\xe2\x9c\x97 MISMATCH",
               stable ? "STABLE" : "UNSTABLE");

        return match && stable;
    };

    std::vector<size_t> Ns = {
        0,1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65
    };

    auto add_around = [&](size_t c) {
        for (int d = -3; d <= 3; ++d) {
            long v = (long)c + d;
            if (v >= 0) Ns.push_back((size_t)v);
        }
    };

    add_around(16);
    add_around(128);
    add_around(4096);
    add_around(8192);
    add_around(12288);

    std::sort(Ns.begin(), Ns.end());
    Ns.erase(std::unique(Ns.begin(), Ns.end()), Ns.end());

    bool all_ok = true;

    printf("\n== LENGTH-SWEEP REPRODUCIBILITY (GPU vs HOST) ==\n");
    printf("    (canonical cross-lane fold; hostile=prime(7,11)+cluster +/-1e16)\n\n");

    for (bool hostile : {false, true}) {
        printf("== Dataset: %s ==\n", hostile ? "HOSTILE" : "BASE");
        for (size_t n : Ns) {
            all_ok &= run_one(16,  n, hostile);
            all_ok &= run_one(128, n, hostile);
        }
        printf("\n");
    }

    return all_ok;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("==================================================================\n");
    printf("  Canonical Reduction - CUDA FAST + CUB comparison (race-fixed)\n");
    printf("  FINAL fold is CANONICAL (no rev-final)\n");
    printf("==================================================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Memory BW: %.0f GB/s (theoretical)\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    constexpr uint64_t SEED = 0x243F6A8885A308D3ULL;
    constexpr int N = 1000000;

    std::vector<double> data(N);
    RNG rng(SEED);
    for (int i = 0; i < N; ++i) data[i] = rng.next();

    printf("N = %d elements (%.2f MB)\n\n", N, N * (int)sizeof(double) / 1e6);

    printf("== HOST REFERENCE (canonical cross-lane) ==\n");
    double host_n = canonical_host(data, 16);
    double host_w = canonical_host(data, 128);
    printf("  NARROW (L=16):  %s\n", hex(host_n).c_str());
    printf("  WIDE  (L=128):  %s\n", hex(host_w).c_str());

    double *d_data = nullptr, *d_out = nullptr, *d_block_lanes = nullptr, *d_final_lanes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_data, data.data(), (size_t)N * sizeof(double), cudaMemcpyHostToDevice));

    int max_blocks = (N + 4096 - 1) / 4096 + 1;
    CUDA_CHECK(cudaMalloc(&d_block_lanes, (size_t)max_blocks * 128 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_final_lanes, 128 * sizeof(double)));

#if HAS_CUB
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, d_out, N);
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
#endif

    printf("\n== GPU CORRECTNESS (N=1e6) ==\n");
    double gpu_n = run_canonical_fast<16>(d_data, d_block_lanes, d_final_lanes, d_out, N);
    double gpu_w = run_canonical_fast<128>(d_data, d_block_lanes, d_final_lanes, d_out, N);
    double gpu_fast = run_fast(d_data, d_out, N);

    std::string hn = hex(host_n), hw = hex(host_w);
    std::string gn = hex(gpu_n),  gw = hex(gpu_w);

    printf("  Canonical FAST (L=16):   %s %s\n", gn.c_str(), (gn == hn) ? "MATCH" : "MISMATCH");
    printf("  Canonical FAST (L=128):  %s %s\n", gw.c_str(), (gw == hw) ? "MATCH" : "MISMATCH");
    printf("  Fast atomic:             %s (non-deterministic)\n", hex(gpu_fast).c_str());

#if HAS_CUB
    double cub_r = run_cub(d_data, d_out, d_temp, temp_bytes, N);
    printf("  CUB:                     %s (non-deterministic)\n", hex(cub_r).c_str());
#endif

    printf("\n== RUN-TO-RUN STABILITY (N=1e6) ==\n");
    bool stable = true;
    for (int i = 0; i < 5; ++i) {
        if (hex(run_canonical_fast<16>(d_data, d_block_lanes, d_final_lanes, d_out, N)) != gn) stable = false;
    }
    printf("  Canonical FAST (L=16):   %s\n", stable ? "STABLE" : "UNSTABLE");

    bool sweep_ok =
        run_length_sweep_cuda(d_data, d_block_lanes, d_final_lanes, d_out,
#if HAS_CUB
                              d_temp, temp_bytes,
#endif
                              SEED);

    printf("\n== PERFORMANCE (N=1e6) ==\n\n");

#if HAS_CUB
    double t_cub = bench("CUB DeviceReduce::Sum", N, 1000, [&]() {
        run_cub(d_data, d_out, d_temp, temp_bytes, N);
    });
#endif

    double t_fast = bench("Fast atomic reduce", N, 1000, [&]() {
        run_fast(d_data, d_out, N);
    });

    double t_narrow = bench("Canonical FAST (L=16)", N, 1000, [&]() {
        run_canonical_fast<16>(d_data, d_block_lanes, d_final_lanes, d_out, N);
    });

    double t_wide = bench("Canonical FAST (L=128)", N, 1000, [&]() {
        run_canonical_fast<128>(d_data, d_block_lanes, d_final_lanes, d_out, N);
    });

    printf("\n== PERFORMANCE vs CUB ==\n");
#if HAS_CUB
    printf("  Canonical FAST (L=16):  %.2fx slower (%.1f%% of CUB throughput)\n",
           t_narrow / t_cub, 100.0 * t_cub / t_narrow);
    printf("  Canonical FAST (L=128): %.2fx slower (%.1f%% of CUB throughput)\n",
           t_wide / t_cub, 100.0 * t_cub / t_wide);
#else
    printf("  (CUB disabled; compile without -DNO_CUB)\n");
#endif

    printf("\n==================================================================\n");
    printf("VERIFICATION BLOCK\n");
    printf("==================================================================\n");
    printf("Platform: CUDA (%s)\n", prop.name);
    printf("SEED:     0x%016llx\n", (unsigned long long)SEED);
    printf("N:        %d\n", N);
    printf("HOST N:   %s  (L=16)\n", hn.c_str());
    printf("GPU  N:   %s  (L=16)\n", gn.c_str());
    printf("HOST W:   %s  (L=128)\n", hw.c_str());
    printf("GPU  W:   %s  (L=128)\n", gw.c_str());
    printf("SWEEP:    %s\n", sweep_ok ? "PASS" : "FAIL");
    printf("==================================================================\n");

#if HAS_CUB
    cudaFree(d_temp);
#endif
    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_block_lanes);
    cudaFree(d_final_lanes);

    bool pass = (gn == hn) && (gw == hw) && stable && sweep_ok;
    printf("\nResult: %s\n", pass ? "ALL PASSED" : "FAILED");
    return pass ? 0 : 1;
}
