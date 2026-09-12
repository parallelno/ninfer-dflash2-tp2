// TP2 communication benchmark: the REAL collective the tensor-parallel decode program issues.
//
// Benchmarks ninfer::ops::allreduce_sum (src/ops/common/allreduce.cu) exactly as the model runs
// it: two DeviceContexts from one ExecutionContext, a 10 KiB BF16 buffer (the small-reduction
// size that dominates TP2 decode: 128 reductions per token = 64 layers x 2 row-parallel
// projections), a live PeerEvents set, and -- because the production path replays these
// collectives inside a captured CUDA graph -- both the eager sequence and a graph-captured
// sequence are measured.
//
// Reported:
//   single reduction latency (µs, averaged over many isolated, stream-synchronized calls)
//   128 consecutive reductions   (total ms, µs per reduction) -- one decode token's collectives
//   the same 128 through a captured cudaGraphExec replay
//
// The reduction payload is 10240 bytes = 5120 BF16 elements.
//
// Build (workspace-local CUDA 13.1, links against the built ninfer libraries):
//   vcvars64 && nvcc -arch=sm_120a -O2 -std=c++20 -I<repo>/include -I<repo>/src ^
//        tools/tp2/reduce_bench.cu ^
//        <build>/src/ninfer_ops.lib <build>/src/ninfer_core.lib ^
//        -lcudart_static -o build/diagnostics/reduce_bench.exe
//
// Run: reduce_bench.exe [dev_a dev_b]   # defaults 0 1
//
// Exit 0 on success, 1 on hard CUDA/library failure.

#include "core/device.h"            // ninfer::DeviceContext / ExecutionContext
#include "core/tensor.h"            // ninfer::Tensor
#include "ninfer/ops/allreduce.h"    // ninfer::ops::allreduce_sum / PeerEvents / enable_peer_access

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kReps128 = 128;        // collectives per decode token: 64 layers x 2 projections
constexpr int kElems   = 5120;      // 5120 BF16 elements = 10240 bytes = 10 KiB

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e = (x);                                                     \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA %s:%d: %s: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorName(e), cudaGetErrorString(e));                 \
            exit(1);                                                             \
        }                                                                       \
    } while (0)

struct DeviceBuffers {
    void* buffer = nullptr;
    void* staging = nullptr;
};

} // namespace

int main(int argc, char** argv) {
    const int dev_a = argc > 1 ? atoi(argv[1]) : 0;
    const int dev_b = argc > 2 ? atoi(argv[2]) : 1;
    printf("reduce_bench: devices %d,%d  payload %d bytes (%d BF16 elems)\n", dev_a, dev_b,
           (int)(kElems * sizeof(unsigned short)), kElems);

    // Two live DeviceContexts, exactly as the engine constructs them for --tp 2 --devices A,B.
    ninfer::ExecutionContext ec(std::vector<int>{dev_a, dev_b});
    printf("devices initialized: %s (sm %d.%d) + %s (sm %d.%d)\n", ec.dev[0]->props.name,
           ec.dev[0]->props.major, ec.dev[0]->props.minor, ec.dev[1]->props.name,
           ec.dev[1]->props.major, ec.dev[1]->props.minor);

    // Report the actual P2P state, then keep running on the transport CUDA chose. P2P is an
    // optimization, never a requirement (the fork's design contract).
    const bool peer = ninfer::ops::enable_peer_access(ec);
    printf("P2P peer access: %s\n",
           peer ? "ENABLED (direct PCIe path)" : "UNAVAILABLE (host-staged transport)");

    ninfer::ops::PeerEvents events(ec);

    // 10 KiB BF16 buffers on each rank, plus staging of the same shape (allreduce_sum contract).
    DeviceBuffers bufs[2];
    for (int rank = 0; rank < 2; ++rank) {
        CK(cudaSetDevice(ec.dev[rank]->device));
        CK(cudaMalloc(&bufs[rank].buffer, kElems * sizeof(unsigned short)));
        CK(cudaMalloc(&bufs[rank].staging, kElems * sizeof(unsigned short)));
        CK(cudaMemset(bufs[rank].buffer, 0, kElems * sizeof(unsigned short)));
        CK(cudaMemset(bufs[rank].staging, 0, kElems * sizeof(unsigned short)));
    }

    std::array<ninfer::Tensor, 2> buffer{};
    std::array<ninfer::Tensor, 2> staging{};
    for (int rank = 0; rank < 2; ++rank) {
        buffer[rank]  = ninfer::Tensor(bufs[rank].buffer, ninfer::DType::BF16, {kElems});
        staging[rank] = ninfer::Tensor(bufs[rank].staging, ninfer::DType::BF16, {kElems});
    }

    // --- 1. Single-reduction latency: issue, synchronize both devices, event-time. ---
    {
        cudaEvent_t start{}, stop{};
        CK(cudaSetDevice(ec.dev[0]->device));
        CK(cudaEventCreate(&start));
        CK(cudaEventCreate(&stop));
        constexpr int kIters = 200;
        float total_ms = 0.0F;
        // warm-up
        for (int i = 0; i < 10; ++i) {
            ninfer::ops::allreduce_sum(buffer, staging, ec, events);
        }
        ec.dev[0]->synchronize();
        ec.dev[1]->synchronize();
        for (int i = 0; i < kIters; ++i) {
            CK(cudaSetDevice(ec.dev[0]->device));
            CK(cudaEventRecord(start, ec.dev[0]->stream));
            ninfer::ops::allreduce_sum(buffer, staging, ec, events);
            CK(cudaEventRecord(stop, ec.dev[0]->stream));
            ec.dev[0]->synchronize();
            ec.dev[1]->synchronize();
            float ms = 0.0F;
            CK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
        const float per_us = total_ms * 1000.0F / kIters;
        printf("single 10 KiB allreduce_sum  : %8.2f us mean (over %d isolated, synchronized calls)\n",
               per_us, kIters);
        CK(cudaEventDestroy(start));
        CK(cudaEventDestroy(stop));
    }

    // --- 2. 128 consecutive reductions: one decode token's collective traffic, eager issue. ---
    {
        cudaEvent_t start{}, stop{};
        CK(cudaSetDevice(ec.dev[0]->device));
        CK(cudaEventCreate(&start));
        CK(cudaEventCreate(&stop));
        constexpr int kTokens = 32;   // 32 tokens x 128 reductions
        for (int i = 0; i < 10; ++i) {
            ninfer::ops::allreduce_sum(buffer, staging, ec, events);
        }
        ec.dev[0]->synchronize();
        ec.dev[1]->synchronize();
        CK(cudaEventRecord(start, ec.dev[0]->stream));
        for (int t = 0; t < kTokens; ++t) {
            for (int i = 0; i < kReps128; ++i) {
                ninfer::ops::allreduce_sum(buffer, staging, ec, events);
            }
        }
        CK(cudaEventRecord(stop, ec.dev[0]->stream));
        ec.dev[0]->synchronize();
        ec.dev[1]->synchronize();
        float ms = 0.0F;
        CK(cudaEventElapsedTime(&ms, start, stop));
        const float total_for_128 = ms / kTokens;
        printf("128 consecutive reductions : %8.2f ms total, %6.2f us per reduction (eager)\n",
               total_for_128, total_for_128 * 1000.0F / kReps128);
        CK(cudaEventDestroy(start));
        CK(cudaEventDestroy(stop));
    }

    // --- 3. The same 128 reductions replayed from a captured CUDA graph, both device streams
    //        enrolled in one capture (the production decode program's shape; see allreduce.h and
    //        tools/tp2/capture_probe.cu for the fork/join choreography this mirrors). ---
    {
        for (int i = 0; i < 5; ++i) {
            ninfer::ops::allreduce_sum(buffer, staging, ec, events);
        }
        ec.dev[0]->synchronize();
        ec.dev[1]->synchronize();

        // Each event is created with its owning device current: fork on rank 0, join on rank 1.
        cudaEvent_t fork{}, join{};
        CK(cudaSetDevice(ec.dev[0]->device));
        CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        CK(cudaSetDevice(ec.dev[1]->device));
        CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));

        cudaGraph_t graph{};
        cudaGraphExec_t exec{};
        // Begin capture on rank 0's stream (the origin), then fork rank 1's stream into the
        // same capture by recording on the origin and making the peer wait on it.
        CK(cudaSetDevice(ec.dev[0]->device));
        const cudaError_t began =
            cudaStreamBeginCapture(ec.dev[0]->stream, cudaStreamCaptureModeThreadLocal);
        if (began != cudaSuccess) {
            printf("graph capture begin: FAILED (%s) -- eager path only\n",
                   cudaGetErrorName(began));
            cudaGetLastError();
        } else {
            const cudaError_t forked = cudaEventRecord(fork, ec.dev[0]->stream);
            CK(cudaSetDevice(ec.dev[1]->device));
            const cudaError_t enrolled = forked != cudaSuccess
                                              ? forked
                                              : cudaStreamWaitEvent(ec.dev[1]->stream, fork, 0);
            if (enrolled != cudaSuccess) {
                printf("peer stream enrollment: FAILED (%s) -- cross-device capture unavailable\n",
                       cudaGetErrorName(enrolled));
                cudaGetLastError();
                cudaGraph_t discard{};
                cudaStreamEndCapture(ec.dev[0]->stream, &discard);
                if (discard != nullptr) { cudaGraphDestroy(discard); }
            } else {
                bool body_ok = true;
                for (int i = 0; i < kReps128 && body_ok; ++i) {
                    cudaError_t status = cudaSuccess;
                    try {
                        ninfer::ops::allreduce_sum(buffer, staging, ec, events);
                    } catch (const std::exception& e) {
                        fprintf(stderr, "allreduce_sum threw inside capture: %s\n", e.what());
                        status = cudaErrorUnknown;
                    }
                    if (status == cudaSuccess) { status = cudaGetLastError(); }
                    if (status != cudaSuccess) {
                        printf("collective %d inside capture: %s\n", i, cudaGetErrorName(status));
                        cudaGetLastError();
                        body_ok = false;
                    }
                }
                if (!body_ok) {
                    cudaGraph_t discard{};
                    cudaStreamEndCapture(ec.dev[0]->stream, &discard);
                    if (discard != nullptr) { cudaGraphDestroy(discard); }
                } else {
                    // The join is mandatory: without it EndCapture reports
                    // cudaErrorStreamCaptureUnjoined.
                    CK(cudaSetDevice(ec.dev[1]->device));
                    const cudaError_t joined = cudaEventRecord(join, ec.dev[1]->stream);
                    CK(cudaSetDevice(ec.dev[0]->device));
                    const cudaError_t rejoined = joined != cudaSuccess
                                                     ? joined
                                                     : cudaStreamWaitEvent(ec.dev[0]->stream, join, 0);
                    const cudaError_t ended =
                        rejoined != cudaSuccess ? rejoined
                                                : cudaStreamEndCapture(ec.dev[0]->stream, &graph);
                    if (ended != cudaSuccess || graph == nullptr) {
                        printf("capture end: %s\n", cudaGetErrorName(ended));
                        cudaGetLastError();
                    } else {
                        size_t node_count = 0;
                        cudaGraphGetNodes(graph, nullptr, &node_count);
                        printf("graph capture of 128 reductions: OK (%zu nodes)\n", node_count);
                        const cudaError_t made = cudaGraphInstantiate(&exec, graph, 0);
                        if (made != cudaSuccess) {
                            printf("cudaGraphInstantiate: %s\n", cudaGetErrorName(made));
                            cudaGetLastError();
                        } else {
                            constexpr int kReplays = 32;
                            cudaEvent_t start{}, stop{};
                            CK(cudaEventCreate(&start));
                            CK(cudaEventCreate(&stop));
                            for (int i = 0; i < 3; ++i) {
                                CK(cudaGraphLaunch(exec, ec.dev[0]->stream));
                            }
                            ec.dev[0]->synchronize();
                            ec.dev[1]->synchronize();
                            CK(cudaEventRecord(start, ec.dev[0]->stream));
                            for (int i = 0; i < kReplays; ++i) {
                                CK(cudaGraphLaunch(exec, ec.dev[0]->stream));
                            }
                            CK(cudaEventRecord(stop, ec.dev[0]->stream));
                            ec.dev[0]->synchronize();
                            ec.dev[1]->synchronize();
                            float ms = 0.0F;
                            CK(cudaEventElapsedTime(&ms, start, stop));
                            ms /= kReplays;
                            printf("128 reductions via graph   : %8.2f ms total, %6.2f us per "
                                   "reduction (replayed)\n",
                                   ms, ms * 1000.0F / kReps128);
                            CK(cudaEventDestroy(start));
                            CK(cudaEventDestroy(stop));
                            CK(cudaGraphExecDestroy(exec));
                            CK(cudaGraphDestroy(graph));
                        }
                    }
                }
            }
        }
        CK(cudaEventDestroy(fork));
        CK(cudaEventDestroy(join));
    }

    for (int rank = 0; rank < 2; ++rank) {
        CK(cudaSetDevice(ec.dev[rank]->device));
        CK(cudaFree(bufs[rank].buffer));
        CK(cudaFree(bufs[rank].staging));
    }
    printf("reduce_bench: done\n");
    return 0;
}
