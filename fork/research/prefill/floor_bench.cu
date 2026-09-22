// Per-chunk fixed-cost floor bench. A TP2 prefill chunk, with transfers DISABLED, still costs
// ~0.2 s at 90 tokens. Candidates: (a) 128 x the 4-event cross-device choreography,
// (b) ~2000 tiny kernel launches across two devices from one host thread on WDDM,
// (c) host-blocking inside cross-device cudaMemcpyAsync.
// Build: nvcc -arch=sm_120a -O2 -std=c++20 floor_bench.cu -o floor_bench.exe
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
__global__ void tiny_kernel(float* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1.0f; }

int main(int argc, char** argv) {
    const int da = argc > 1 ? atoi(argv[1]) : 0, db = argc > 2 ? atoi(argv[2]) : 1;
    int dev[2] = {da, db}; cudaStream_t s[2]; cudaEvent_t ready[2], done[2]; float* buf[2]; float* other[2];
    for (int q = 0; q < 2; ++q) {
        CK(cudaSetDevice(dev[q])); CK(cudaStreamCreateWithFlags(&s[q], cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ready[q], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&done[q], cudaEventDisableTiming));
        CK(cudaMalloc(&buf[q], 1 << 20)); CK(cudaMalloc(&other[q], 1 << 20));
    }
    auto sync = [&] { for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaDeviceSynchronize()); } };
    const int N = 128;
    auto run = [&](const char* label, auto&& body) {
        sync(); body(8); sync();
        auto t0 = Clock::now(); body(N); auto ti = Clock::now(); sync(); auto t1 = Clock::now();
        printf("  %-52s: %7.2f ms total, %6.1f us/iter (host issue %6.2f ms)\n", label, ms(t0, t1), 1000.0 * ms(t0, t1) / N, ms(t0, ti));
    };
    // (a) 4-event choreography + tiny combine kernel, no copy (== NINFER_TP2_DIAG_NO_TRANSFER path)
    run("(a) 128x 4-event choreography, no copy, tiny kernel", [&](int n) {
        for (int i = 0; i < n; ++i) {
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaEventRecord(ready[q], s[q])); }
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaStreamWaitEvent(s[q], ready[1 - q], 0)); CK(cudaEventRecord(done[q], s[q])); }
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaStreamWaitEvent(s[q], done[1 - q], 0)); tiny_kernel<<<1, 32, 0, s[q]>>>(buf[q]); }
        }
    });
    // (a2) same-device events only (no cross-device wait)
    run("(a2) 128x same-device record+wait, tiny kernel", [&](int n) {
        for (int i = 0; i < n; ++i)
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaEventRecord(ready[q], s[q])); CK(cudaStreamWaitEvent(s[q], ready[q], 0)); CK(cudaEventRecord(done[q], s[q])); CK(cudaStreamWaitEvent(s[q], done[q], 0)); tiny_kernel<<<1, 32, 0, s[q]>>>(buf[q]); }
    });
    // (b) kernel launches only: 16 per rank per iter (~one layer's worth), alternating devices
    run("(b) 128x 32 tiny launches alternating devices", [&](int n) {
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 16; ++k)
                for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); tiny_kernel<<<1, 32, 0, s[q]>>>(buf[q]); }
    });
    run("(b2) 128x 32 tiny launches, device 0 only", [&](int n) {
        CK(cudaSetDevice(dev[0]));
        for (int i = 0; i < n; ++i) for (int k = 0; k < 32; ++k) tiny_kernel<<<1, 32, 0, s[0]>>>(buf[0]);
    });
    // (c) 128x choreography WITH a 64 KiB cross-device copy (decode-ish size, eager)
    run("(c) 128x choreography + 64 KiB staged pull", [&](int n) {
        for (int i = 0; i < n; ++i) {
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaEventRecord(ready[q], s[q])); }
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaStreamWaitEvent(s[q], ready[1 - q], 0)); CK(cudaMemcpyAsync(other[q], buf[1 - q], 64 << 10, cudaMemcpyDeviceToDevice, s[q])); CK(cudaEventRecord(done[q], s[q])); }
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaStreamWaitEvent(s[q], done[1 - q], 0)); tiny_kernel<<<1, 32, 0, s[q]>>>(buf[q]); }
        }
    });
    // (c2) 128x choreography + 900 KiB (a 90-token chunk's payload)
    run("(c2) 128x choreography + 900 KiB staged pull", [&](int n) {
        for (int i = 0; i < n; ++i) {
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaEventRecord(ready[q], s[q])); }
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaStreamWaitEvent(s[q], ready[1 - q], 0)); CK(cudaMemcpyAsync(other[q], buf[1 - q], 900 << 10, cudaMemcpyDeviceToDevice, s[q])); CK(cudaEventRecord(done[q], s[q])); }
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaStreamWaitEvent(s[q], done[1 - q], 0)); tiny_kernel<<<1, 32, 0, s[q]>>>(buf[q]); }
        }
    });
    // (d) pageable H2D of 4 KiB (like copy_i32 of ids/positions per chunk) - is it host-blocking?
    {
        static int host_ids[1024]; sync(); CK(cudaSetDevice(dev[0]));
        auto t0 = Clock::now();
        for (int i = 0; i < 32; ++i) CK(cudaMemcpyAsync(buf[0], host_ids, sizeof(host_ids), cudaMemcpyHostToDevice, s[0]));
        auto ti = Clock::now(); CK(cudaStreamSynchronize(s[0])); auto t1 = Clock::now();
        printf("  %-52s: %7.2f ms total, %6.1f us/iter (host issue %6.2f ms)\n", "(d) 32x pageable 4 KiB H2D memcpyAsync", ms(t0, t1), 1000.0 * ms(t0, t1) / 32, ms(t0, ti));
    }
    printf("done\n");
    return 0;
}
