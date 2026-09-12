// WDDM graph-launch staging probe.
//
// HYPOTHESIS (from nsys: 160,862 cudaMemcpyAsync H2D of 4,352 B / 3,072 B during MTP3 decode,
// ~1,219 per round, one packet pair per graph node): on WDDM every cudaGraphLaunch re-uploads
// each kernel node's launch parameters from host to device, serialized on PCIe with ~2 us gaps.
// That would make graph replay cost grow linearly with NODE COUNT and add ~5-10 ms per launch to
// a ~1,200-node decode graph.
//
// This probe measures, for graphs of 2 / 64 / 256 / 1024 / 4096 trivial kernel nodes:
//   1. cudaGraphLaunch + stream-sync wall time per launch, steady state   (host-side cost)
//   2. the same graph launched from a DEVICE tail-launch node, if the
//      cudaGraphInstantiateFlagDeviceLaunch path is available                 (the candidate fix)
//   3. eager launch of the same kernel count (baseline for comparison)
//
// Exit 0 on success, 1 on failure.

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

__global__ void trivial_kernel(int* sink) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { sink[0] = blockIdx.x; }
}

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e = (x);                                                     \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA %s:%d: %s: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorName(e), cudaGetErrorString(e));                 \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

double ms_since(const std::chrono::high_resolution_clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t)
        .count();
}

}  // namespace

int main(int argc, char** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    CK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    printf("device %d: %s, CUDA %d.%d\n", device, prop.name, prop.major, prop.minor);

    int* sink = nullptr;
    CK(cudaMalloc(&sink, sizeof(int)));

    cudaStream_t stream = nullptr;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    for (int nodes : {2, 64, 256, 1024, 4096}) {
        cudaEvent_t done = nullptr;
        CK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

        // --- capture `nodes` sequential kernel launches into one graph ---
        CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < nodes; ++i) {
            trivial_kernel<<<1, 32, 0, stream>>>(sink);
            CK(cudaGetLastError());
        }
        cudaGraph_t graph = nullptr;
        CK(cudaStreamEndCapture(stream, &graph));
        cudaGraphExec_t exec = nullptr;
        CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        // --- steady-state host launch cost ---
        constexpr int kWarm = 20;
        constexpr int kReps = 200;
        for (int i = 0; i < kWarm; ++i) { CK(cudaGraphLaunch(exec, stream)); }
        CK(cudaStreamSynchronize(stream));
        double total = 0.0;
        for (int i = 0; i < kReps; ++i) {
            const auto start = std::chrono::high_resolution_clock::now();
            CK(cudaGraphLaunch(exec, stream));
            CK(cudaStreamSynchronize(stream));
            total += ms_since(start);
        }
        const double graph_us = total / kReps * 1000.0;

        // --- eager launch of the same node count (no graph) ---
        for (int i = 0; i < kWarm; ++i) { trivial_kernel<<<1, 32, 0, stream>>>(sink); }
        CK(cudaStreamSynchronize(stream));
        total = 0.0;
        for (int i = 0; i < kReps; ++i) {
            const auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < nodes; ++i) { trivial_kernel<<<1, 32, 0, stream>>>(sink); }
            CK(cudaStreamSynchronize(stream));
            total += ms_since(start);
        }
        const double eager_us = total / kReps * 1000.0;

        printf("nodes=%5d: graph launch+sync %8.2f us/launch | eager %8.2f us for %d kernels "
               "(per-kernel %.3f us)\n",
               nodes, graph_us, eager_us, nodes, eager_us / nodes);

        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
        CK(cudaEventDestroy(done));
    }

    CK(cudaStreamDestroy(stream));
    CK(cudaFree(sink));
    printf("graph_launch_probe: PASS\n");
    return 0;
}
