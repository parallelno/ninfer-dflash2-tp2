// Kernel-launch host overhead on this machine (Xeon E5-2650 v2, Windows 11 WDDM, CUDA 13.4).
// Build: nvcc -arch=sm_120a -O2 -std=c++20 launch_bench.cu -o launch_bench.exe
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
__global__ void tiny_kernel(float* p, int a, int b, int c, int d) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += (float)(a + b + c + d); }
__global__ void busy_kernel(float* p, int iters) { float v = p[threadIdx.x]; for (int i = 0; i < iters; ++i) v = v * 1.0001f + 0.5f; p[threadIdx.x] = v; }

int main(int argc, char** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    const int N = 2000;
    CK(cudaSetDevice(dev));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float* buf; CK(cudaMalloc(&buf, 1 << 20)); CK(cudaMemset(buf, 0, 1 << 20));
    for (int i = 0; i < 100; ++i) tiny_kernel<<<1, 32, 0, s>>>(buf, 1, 2, 3, 4);
    CK(cudaStreamSynchronize(s));
    // (1) per-launch host cost distribution, tiny kernel, 1 block
    {
        std::vector<double> t(N);
        for (int i = 0; i < N; ++i) { auto a = Clock::now(); tiny_kernel<<<1, 32, 0, s>>>(buf, i, 2, 3, 4); t[i] = us(a, Clock::now()); }
        auto a = Clock::now(); CK(cudaStreamSynchronize(s)); double drain = us(a, Clock::now());
        std::sort(t.begin(), t.end());
        double sum = 0; for (double v : t) sum += v;
        printf("(1) tiny <<<1,32>>>   : mean %6.1f us  p50 %6.1f  p90 %6.1f  p99 %6.1f  max %7.1f  | drain after issue %7.1f us\n", sum / N, t[N / 2], t[N * 9 / 10], t[N * 99 / 100], t[N - 1], drain);
    }
    // (2) same with 1024 blocks (more realistic grid)
    {
        std::vector<double> t(N);
        for (int i = 0; i < N; ++i) { auto a = Clock::now(); tiny_kernel<<<1024, 256, 0, s>>>(buf, i, 2, 3, 4); t[i] = us(a, Clock::now()); }
        auto a = Clock::now(); CK(cudaStreamSynchronize(s)); double drain = us(a, Clock::now());
        std::sort(t.begin(), t.end()); double sum = 0; for (double v : t) sum += v;
        printf("(2) tiny <<<1024,256>>>: mean %6.1f us  p50 %6.1f  p90 %6.1f  p99 %6.1f  max %7.1f  | drain %7.1f us\n", sum / N, t[N / 2], t[N * 9 / 10], t[N * 99 / 100], t[N - 1], drain);
    }
    // (3) GPU-bound: kernels take ~50 us each; does host issue get ahead (async) or stay lockstep?
    {
        std::vector<double> t(N);
        for (int i = 0; i < N; ++i) { auto a = Clock::now(); busy_kernel<<<1, 32, 0, s>>>(buf, 20000); t[i] = us(a, Clock::now()); }
        auto a = Clock::now(); CK(cudaStreamSynchronize(s)); double drain = us(a, Clock::now());
        std::sort(t.begin(), t.end()); double sum = 0; for (double v : t) sum += v;
        printf("(3) busy ~50us kernel  : mean %6.1f us  p50 %6.1f  p90 %6.1f  p99 %6.1f  max %7.1f  | drain %7.1f us (queue depth ~ drain/50us)\n", sum / N, t[N / 2], t[N * 9 / 10], t[N * 99 / 100], t[N - 1], drain);
    }
    // (4) cudaEventRecord + cudaStreamWaitEvent same stream
    {
        cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        auto a = Clock::now();
        for (int i = 0; i < N; ++i) { CK(cudaEventRecord(e, s)); CK(cudaStreamWaitEvent(s, e, 0)); }
        double issue = us(a, Clock::now()); CK(cudaStreamSynchronize(s));
        printf("(4) record+wait pair   : %6.1f us per pair (host)\n", issue / N);
    }
    // (5) cudaMemsetAsync / set scalar style
    {
        auto a = Clock::now();
        for (int i = 0; i < N; ++i) CK(cudaMemsetAsync(buf, 0, 4, s));
        double issue = us(a, Clock::now()); CK(cudaStreamSynchronize(s));
        printf("(5) memsetAsync 4 B    : %6.1f us per call (host)\n", issue / N);
    }
    printf("done\n");
    return 0;
}
