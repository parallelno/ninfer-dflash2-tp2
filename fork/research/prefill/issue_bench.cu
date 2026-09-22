// Host-issue bench: how long does the calling thread block inside ONE cross-device
// cudaMemcpyAsync(D2D over UVA) on a no-P2P pair, and does issuing the two directions from two
// host threads change the pair time? (The production allreduce issues both pulls from one thread.)
// Build: nvcc -arch=sm_120a -O2 -std=c++20 issue_bench.cu -o issue_bench.exe
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
    const int da = argc > 1 ? atoi(argv[1]) : 0, db = argc > 2 ? atoi(argv[2]) : 1;
    const double mib = argc > 3 ? atof(argv[3]) : 10.0;
    const size_t bytes = (size_t)(mib * 1024 * 1024);
    int dev[2] = {da, db}; void* src[2]; void* dst[2]; void* pin[2]; cudaStream_t s[2];
    for (int q = 0; q < 2; ++q) {
        CK(cudaSetDevice(dev[q])); CK(cudaStreamCreateWithFlags(&s[q], cudaStreamNonBlocking));
        CK(cudaMalloc(&src[q], bytes)); CK(cudaMalloc(&dst[q], bytes)); CK(cudaHostAlloc(&pin[q], bytes, cudaHostAllocPortable));
    }
    auto sync = [&] { for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaDeviceSynchronize()); } };
    printf("issue_bench: %.1f MiB payload\n", mib);
    const int N = 16;
    // (a) single staged pull dev1->dev0: host block vs completion
    for (int rep = 0; rep < 2; ++rep) {
        sync(); CK(cudaSetDevice(dev[0]));
        double issue = 0, total = 0;
        for (int i = 0; i < N; ++i) {
            auto t0 = Clock::now();
            CK(cudaMemcpyAsync(dst[0], src[1], bytes, cudaMemcpyDeviceToDevice, s[0]));
            auto t1 = Clock::now(); CK(cudaStreamSynchronize(s[0])); auto t2 = Clock::now();
            issue += ms(t0, t1); total += ms(t0, t2);
        }
        if (rep) printf("  (a) staged D2D single pull      : host blocked %6.3f ms of %6.3f ms per call\n", issue / N, total / N);
    }
    // (b) pinned D2H single: host block
    {
        sync(); CK(cudaSetDevice(dev[0])); double issue = 0, total = 0;
        for (int i = 0; i < N; ++i) {
            auto t0 = Clock::now(); CK(cudaMemcpyAsync(pin[0], src[0], bytes, cudaMemcpyDeviceToHost, s[0]));
            auto t1 = Clock::now(); CK(cudaStreamSynchronize(s[0])); auto t2 = Clock::now();
            issue += ms(t0, t1); total += ms(t0, t2);
        }
        printf("  (b) pinned D2H single           : host blocked %6.3f ms of %6.3f ms per call\n", issue / N, total / N);
    }
    // (c) pair of staged pulls, ONE thread (production order)
    {
        sync(); double total = 0;
        for (int i = 0; i < N; ++i) {
            auto t0 = Clock::now();
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaMemcpyAsync(dst[q], src[1 - q], bytes, cudaMemcpyDeviceToDevice, s[q])); }
            sync(); total += ms(t0, Clock::now());
        }
        printf("  (c) staged pair, one host thread: %6.3f ms per pair\n", total / N);
    }
    // (d) pair of staged pulls, TWO threads issuing simultaneously
    {
        sync(); double total = 0;
        for (int i = 0; i < N; ++i) {
            auto t0 = Clock::now();
            std::thread th[2];
            for (int q = 0; q < 2; ++q) th[q] = std::thread([&, q] { CK(cudaSetDevice(dev[q])); CK(cudaMemcpyAsync(dst[q], src[1 - q], bytes, cudaMemcpyDeviceToDevice, s[q])); CK(cudaStreamSynchronize(s[q])); });
            for (auto& t : th) t.join();
            total += ms(t0, Clock::now());
        }
        printf("  (d) staged pair, two host threads: %6.3f ms per pair\n", total / N);
    }
    // (e) pinned relay pair (D2H both, then H2D both), one thread
    {
        sync(); double total = 0;
        for (int i = 0; i < N; ++i) {
            auto t0 = Clock::now();
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaMemcpyAsync(pin[q], src[q], bytes, cudaMemcpyDeviceToHost, s[q])); }
            sync();
            for (int q = 0; q < 2; ++q) { CK(cudaSetDevice(dev[q])); CK(cudaMemcpyAsync(dst[q], pin[1 - q], bytes, cudaMemcpyHostToDevice, s[q])); }
            sync(); total += ms(t0, Clock::now());
        }
        printf("  (e) pinned relay pair (host-synced phases): %6.3f ms per pair\n", total / N);
    }
    printf("done\n");
    return 0;
}
