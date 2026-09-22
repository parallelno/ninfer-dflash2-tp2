// Overlap bench: can a cross-device host-staged copy (the production TP2 allreduce transport on a
// no-P2P WDDM pair) run CONCURRENTLY with SM compute on the same GPU, on a second stream?
// Decides whether intra-chunk micro-batch pipelining (compute micro-batch B while micro-batch A's
// allreduce is in flight) can hide the collective time.
//   (1) compute only   (2) staged pull only   (3) both, independent streams
//   (4) pinned relay only   (5) compute + relay
// If (3) ~= max((1),(2)) copy engine and SMs overlap; if (3) ~= (1)+(2) they serialize.
// Build: nvcc -arch=sm_120a -O2 -std=c++20 overlap_bench.cu -o overlap_bench.exe
// Run:   overlap_bench.exe [dev_a dev_b] [payload_mib] [compute_ms_target]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s: %s\n", __FILE__, __LINE__, cudaGetErrorName(e_), cudaGetErrorString(e_)); exit(1); } } while (0)

// DRAM-streaming kernel standing in for the weight-streaming NVFP4 GEMMs; occupies all SMs.
__global__ void stream_kernel(float4* __restrict__ buf, int n, int passes) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int span = gridDim.x * blockDim.x;
    for (int p = 0; p < passes; ++p) {
        for (int i = tid; i < n; i += span) {
            float4 v = buf[i];
            v.x = v.x * 1.0001f + 0.5f; v.y = v.y * 0.9999f - 0.25f;
            v.z = v.z * 1.0002f + 0.125f; v.w = v.w * 0.9998f - 0.0625f;
            buf[i] = v;
        }
    }
}

struct Rank {
    int dev = 0;
    cudaStream_t sC = nullptr, sX = nullptr;
    cudaEvent_t inputs_ready = nullptr, pull_done = nullptr;
    void* src = nullptr; void* staging = nullptr; float4* work = nullptr; void* bounce = nullptr;
};

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int dev_a = argc > 1 ? atoi(argv[1]) : 0;
    const int dev_b = argc > 2 ? atoi(argv[2]) : 1;
    const double payload_mib = argc > 3 ? atof(argv[3]) : 10.0;
    const double target_ms = argc > 4 ? atof(argv[4]) : 3.0;
    const size_t bytes = static_cast<size_t>(payload_mib * 1024.0 * 1024.0);
    constexpr int kN = 32;
    constexpr size_t kWorkBytes = 256ull << 20;
    const int work_n = static_cast<int>(kWorkBytes / sizeof(float4));

    Rank r[2]; r[0].dev = dev_a; r[1].dev = dev_b;
    int can01 = 0, can10 = 0;
    CK(cudaDeviceCanAccessPeer(&can01, dev_a, dev_b)); CK(cudaDeviceCanAccessPeer(&can10, dev_b, dev_a));
    printf("overlap_bench: devices %d,%d  P2P %s  payload %.2f MiB  N=%d\n", dev_a, dev_b,
           (can01 && can10) ? "AVAILABLE" : "unavailable (host-staged)", payload_mib, kN);
    for (auto& k : r) {
        CK(cudaSetDevice(k.dev));
        CK(cudaStreamCreateWithFlags(&k.sC, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&k.sX, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&k.inputs_ready, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&k.pull_done, cudaEventDisableTiming));
        CK(cudaMalloc(&k.src, bytes)); CK(cudaMalloc(&k.staging, bytes)); CK(cudaMalloc(&k.work, kWorkBytes));
        CK(cudaMemset(k.src, 1, bytes)); CK(cudaMemset(k.work, 0, kWorkBytes));
        CK(cudaHostAlloc(&k.bounce, bytes, cudaHostAllocPortable));
    }
    auto sync_all = [&] { for (auto& k : r) { CK(cudaSetDevice(k.dev)); CK(cudaDeviceSynchronize()); } };

    int passes = 1;
    {
        CK(cudaSetDevice(r[0].dev));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int p = 1; p <= 64; p *= 2) {
            stream_kernel<<<1024, 256, 0, r[0].sC>>>(r[0].work, work_n, p);
            CK(cudaEventRecord(e0, r[0].sC));
            stream_kernel<<<1024, 256, 0, r[0].sC>>>(r[0].work, work_n, p);
            CK(cudaEventRecord(e1, r[0].sC));
            CK(cudaStreamSynchronize(r[0].sC));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            passes = p;
            if (ms >= target_ms) { printf("compute kernel: passes=%d -> %.2f ms per launch\n", p, ms); break; }
        }
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
    auto issue_compute = [&](int n) {
        for (int i = 0; i < n; ++i)
            for (auto& k : r) { CK(cudaSetDevice(k.dev)); stream_kernel<<<1024, 256, 0, k.sC>>>(k.work, work_n, passes); }
    };
    // Production staged pull with the 4-event choreography, all on stream X.
    auto issue_staged = [&](int n) {
        for (int i = 0; i < n; ++i) {
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(r[q].dev)); CK(cudaEventRecord(r[q].inputs_ready, r[q].sX)); }
            for (int q = 0; q < 2; ++q) {
                CK(cudaSetDevice(r[q].dev));
                CK(cudaStreamWaitEvent(r[q].sX, r[1 - q].inputs_ready, 0));
                CK(cudaMemcpyAsync(r[q].staging, r[1 - q].src, bytes, cudaMemcpyDeviceToDevice, r[q].sX));
                CK(cudaEventRecord(r[q].pull_done, r[q].sX));
            }
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(r[q].dev)); CK(cudaStreamWaitEvent(r[q].sX, r[1 - q].pull_done, 0)); }
        }
    };
    auto issue_relay = [&](int n) {
        for (int i = 0; i < n; ++i) {
            for (int q = 0; q < 2; ++q) {
                CK(cudaSetDevice(r[q].dev));
                CK(cudaMemcpyAsync(r[q].bounce, r[q].src, bytes, cudaMemcpyDeviceToHost, r[q].sX));
                CK(cudaEventRecord(r[q].inputs_ready, r[q].sX));
            }
            for (int q = 0; q < 2; ++q) {
                CK(cudaSetDevice(r[q].dev));
                CK(cudaStreamWaitEvent(r[q].sX, r[1 - q].inputs_ready, 0));
                CK(cudaMemcpyAsync(r[q].staging, r[1 - q].bounce, bytes, cudaMemcpyHostToDevice, r[q].sX));
                CK(cudaEventRecord(r[q].pull_done, r[q].sX));
            }
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(r[q].dev)); CK(cudaStreamWaitEvent(r[q].sX, r[1 - q].pull_done, 0)); }
        }
    };
    auto timed = [&](const char* label, auto&& issue) {
        sync_all(); issue(2); sync_all();
        const auto t0 = std::chrono::steady_clock::now();
        issue(kN);
        const auto ti = std::chrono::steady_clock::now();
        sync_all();
        const auto t1 = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double issue_ms = std::chrono::duration<double, std::milli>(ti - t0).count();
        printf("  %-30s: %8.2f ms  (%6.3f ms/iter; host issue %6.2f ms)\n", label, wall, wall / kN, issue_ms);
        return wall;
    };
    const double tc = timed("(1) compute only", issue_compute);
    const double tx = timed("(2) staged pull only", issue_staged);
    const double tb = timed("(3) compute + staged pull", [&](int n) { issue_compute(n); issue_staged(n); });
    const double tb2 = timed("(3b) staged pull + compute", [&](int n) { issue_staged(n); issue_compute(n); });
    const double tr = timed("(4) pinned relay only", issue_relay);
    const double tbr = timed("(5) compute + pinned relay", [&](int n) { issue_compute(n); issue_relay(n); });
    printf("\n  staged: sum=%.1f max=%.1f measured=%.1f/%.1f -> copy hidden %.0f%%\n", tc + tx, tc > tx ? tc : tx, tb, tb2, 100.0 * (tc + tx - tb) / tx);
    printf("  relay : sum=%.1f max=%.1f measured=%.1f -> copy hidden %.0f%%\n", tc + tr, tc > tr ? tc : tr, tbr, 100.0 * (tc + tr - tbr) / tr);
    for (auto& k : r) {
        CK(cudaSetDevice(k.dev));
        CK(cudaFree(k.src)); CK(cudaFree(k.staging)); CK(cudaFree(k.work)); CK(cudaFreeHost(k.bounce));
        CK(cudaStreamDestroy(k.sC)); CK(cudaStreamDestroy(k.sX));
        CK(cudaEventDestroy(k.inputs_ready)); CK(cudaEventDestroy(k.pull_done));
    }
    printf("done\n");
    return 0;
}
