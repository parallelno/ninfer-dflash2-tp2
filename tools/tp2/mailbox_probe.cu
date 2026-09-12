// TP2 mailbox-transport probe: kernel-driven exchange through pinned host memory.
//
// HYPOTHESIS. On this machine (Windows 11 WDDM, 2x RTX 5060 Ti, no P2P) the production
// allreduce_sum costs ~277 us per 10 KiB reduction because every one of its 128 instances per
// decode token walks a four-hop cross-device event chain plus a driver-staged copy-engine
// transfer whose WDDM submission overhead dwarfs the 3.3 us of useful payload time. A pair of
// small kernels -- one per device, running CONCURRENTLY -- can instead publish their partial
// directly into pinned host memory, signal through a spin flag, read the peer's partial, and
// sum locally, with no events, no copy engine, and no driver round trip inside the exchange:
//
//   rank r kernel:  store partial_r -> host_payload[r]
//                   __threadfence_system(); __syncthreads();   (all publishes complete)
//                   flag[r] = 1                                   (release)
//                   poll flag[1-r] == 1; fence;                   (acquire)
//                   out_r = partial_r + host_payload[1-r]
//
// Both kernels run at the same time, so the critical path is one PCIe write + one flag
// round-trip + one PCIe read of the payload, entirely GPU-side. This probe measures whether
// that beats the staged path on THIS machine before anything touches the engine.
//
// Measured here, per payload (10 KiB decode activation, 40 KiB MTP-verify width):
//   1. one eager exchange pair                       (us/op)
//   2. 128 consecutive eager exchange pairs          (ms total, us/op)
//   3. 128 captured into ONE cross-device CUDA graph (ms/replay, us/op)  <- the decode shape
//   4. the same 128 through the STAGED memcpy path inside a graph       <- current transport
//   5. flag round-trip latency alone (empty payload)
//   6. correctness vs a CPU reference, eager and through a replayed graph
//   7. pinned default (WB) vs write-combined host memory (via --write-combined)
//
// Build (workspace-local CUDA 13.1; no engine libraries):
//   vcvars64 && nvcc -arch=sm_120a -O2 -std=c++20 tools/tp2/mailbox_probe.cu ^
//        -lcudart_static -o build/diagnostics/mailbox_probe.exe
//
// Run: mailbox_probe.exe [dev_a dev_b] [--write-combined]     # defaults 0 1
//
// Exit 0 on success (a losing comparison is still a result), 1 on hard failure (CUDA error,
// hang detection, wrong sum).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// The engine's production exchange kernel and launch geometry -- the same header the
// allreduce collective compiles -- so the numbers here measure the shipped code, not a
// lookalike.
#include "ops/kernel/peer_exchange.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kElems5120 = 5120;             // 5120 BF16 = 10 KiB, the decode activation
constexpr int kElemsMtp  = 4 * kElems5120;   // MTP verify width 4 = 40 KiB
constexpr int kChain     = 128;              // collectives per decode token (64 layers x 2)

// Geometry sweep: 40 KiB exchanges currently launch 3 blocks (kPeerGroup=4). The publish phase is
// host-write bandwidth bound and the consume phase is latency bound; more blocks spread the read
// across more SMs. This sweep finds the best block count on THIS machine.
int g_override_blocks = -1;

#define CK(x)                                                                    \
    do {                                                                         \
        cudaError_t e = (x);                                                     \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA %s:%d: %s: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorName(e), cudaGetErrorString(e));                 \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

struct HostSlab {
    void* raw                 = nullptr;
    // Per-slot isolation: every exchange pair k gets its own payload areas and flag words, so
    // pair k+1's publish can never clobber pair k's payload mid-read and no poller can observe
    // a stale flag from a different pair. This mirrors the engine design, where each captured
    // allreduce call site owns a distinct slot.
    __nv_bfloat16* payload[2] = {nullptr, nullptr};  // [rank][slot][elem]
    std::uint32_t* flag[2]    = {nullptr, nullptr};  // [rank][slot]
    std::uint32_t* hang_all   = nullptr;             // aggregate hang word
    int slots                 = 0;
    int slot_elems            = 0;
    std::size_t bytes         = 0;
};

// One pinned host slab: per-slot payload areas for both ranks, then the flag/hang words. The
// payload and flag layout matches ops::PeerMailbox's slab (256-byte slot stride, words after the
// payload areas), at probe scale.
HostSlab alloc_host_slab(int slots, int slot_elems, bool write_combined) {
    HostSlab slab;
    slab.slots      = slots;
    slab.slot_elems = slot_elems;
    const std::size_t payload_bytes =
        static_cast<std::size_t>(slots) * 2 * slot_elems * sizeof(__nv_bfloat16);
    const std::size_t words_bytes =
        static_cast<std::size_t>(slots) * 2 * sizeof(std::uint32_t) + sizeof(std::uint32_t);
    slab.bytes = payload_bytes + words_bytes;
    const unsigned flags =
        write_combined ? (cudaHostAllocWriteCombined | cudaHostAllocMapped) : cudaHostAllocMapped;
    CK(cudaHostAlloc(&slab.raw, slab.bytes, flags));
    std::memset(slab.raw, 0, slab.bytes);
    auto* base      = static_cast<std::uint8_t*>(slab.raw);
    slab.payload[0] = reinterpret_cast<__nv_bfloat16*>(base);
    slab.payload[1] = slab.payload[0] + static_cast<std::size_t>(slots) * slot_elems;
    auto* words      = reinterpret_cast<std::uint32_t*>(base + payload_bytes);
    slab.flag[0]     = words;
    slab.flag[1]     = slab.flag[0] + slots;
    slab.hang_all    = slab.flag[1] + slots;
    return slab;
}

void free_host_slab(HostSlab& slab) {
    if (slab.raw != nullptr) { CK(cudaFreeHost(slab.raw)); }
    slab = HostSlab{};
}

struct DeviceSide {
    __nv_bfloat16* partial = nullptr;  // rank's own operand
    __nv_bfloat16* result  = nullptr;  // rank's sum output
};

double ms_since(std::chrono::high_resolution_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - start)
        .count();
}

} // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);  // a pipe is block-buffered; hangs must localize
    const int dev_a = argc > 1 ? atoi(argv[1]) : 0;
    const int dev_b = argc > 2 ? atoi(argv[2]) : 1;
    bool write_combined = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--write-combined") == 0) { write_combined = true; }
    }
    printf("mailbox_probe: devices %d,%d  host memory: %s\n", dev_a, dev_b,
           write_combined ? "write-combined" : "default (WB)");

    int peer_forward = 0;
    int peer_reverse = 0;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--blocks=", 9) == 0) { g_override_blocks = atoi(argv[i] + 9); }
    }
    CK(cudaDeviceCanAccessPeer(&peer_forward, dev_a, dev_b));
    CK(cudaDeviceCanAccessPeer(&peer_reverse, dev_b, dev_a));
    printf("cudaDeviceCanAccessPeer: %d->%d=%d  %d->%d=%d  (0 => no direct P2P; the exchange "
           "rides pinned host memory)%s\n",
           dev_a, dev_b, peer_forward, dev_b, dev_a, peer_reverse,
           g_override_blocks > 0 ? "  [geometry override: see per-exchange lines]" : "");

    // ---- buffers -----------------------------------------------------------------------
    // Buffers are sized for the WIDE (40 KiB MTP-verify) shape: the wide section must be able
    // to run in-bounds, and the 10 KiB sections simply use a prefix of the same allocation.
    const int elems = kElems5120;
    DeviceSide side[2];
    for (int rank = 0; rank < 2; ++rank) {
        const int dev = rank == 0 ? dev_a : dev_b;
        CK(cudaSetDevice(dev));
        CK(cudaMalloc(&side[rank].partial, kElemsMtp * sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&side[rank].result, kElemsMtp * sizeof(__nv_bfloat16)));
    }

    HostSlab slab = alloc_host_slab(kChain, kElemsMtp, write_combined);
    printf("pinned host slab: %.1f KiB (%s), %d slots x %d elems\n", slab.bytes / 1024.0,
           write_combined ? "WC" : "WB", kChain, kElemsMtp);

    for (int rank = 0; rank < 2; ++rank) {
        void* device_ptr = nullptr;
        CK(cudaHostGetDevicePointer(&device_ptr, slab.flag[rank], 0));
        if (device_ptr != static_cast<void*>(slab.flag[rank])) {
            printf("NOTE: UVA identity mapping absent for flags (host %p -> device %p)\n",
                   static_cast<void*>(slab.flag[rank]), device_ptr);
        }
    }

    // Deterministic operands and the CPU reference sum. The 40 KiB region is filled so the wide
    // exchange reads defined values everywhere; the 10 KiB checks only look at the prefix.
    std::vector<__nv_bfloat16> host_partial[2];
    for (int rank = 0; rank < 2; ++rank) {
        host_partial[rank].resize(kElemsMtp);
        for (int i = 0; i < kElemsMtp; ++i) {
            const float v = (rank == 0 ? 1.0f : -0.5f) + 0.001f * static_cast<float>(i % 977);
            host_partial[rank][i] = __float2bfloat16_rn(v);
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        const int dev = rank == 0 ? dev_a : dev_b;
        CK(cudaSetDevice(dev));
        CK(cudaMemcpy(side[rank].partial, host_partial[rank].data(),
                      kElemsMtp * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    }

    // ---- launch helpers ----------------------------------------------------------------
    cudaStream_t stream[2];
    for (int rank = 0; rank < 2; ++rank) {
        const int dev = rank == 0 ? dev_a : dev_b;
        CK(cudaSetDevice(dev));
        CK(cudaStreamCreateWithFlags(&stream[rank], cudaStreamNonBlocking));
    }
    // Per-slot arrival counters on each device, like the engine's PeerMailbox.
    std::uint32_t* arrival[2] = {nullptr, nullptr};
    for (int rank = 0; rank < 2; ++rank) {
        const int dev = rank == 0 ? dev_a : dev_b;
        CK(cudaSetDevice(dev));
        CK(cudaMalloc(&arrival[rank], kChain * sizeof(std::uint32_t)));
        CK(cudaMemset(arrival[rank], 0, kChain * sizeof(std::uint32_t)));
    }

    // Both ranks' exchange kernels launch with no event dependency on each other -- the spin
    // protocol IS the ordering -- so inside a capture they become two independent graph nodes.
    // Each call takes its own slot: publish to slab.payload[rank] + slot, signal
    // slab.flag[rank][slot], poll the peer's flag for the same slot. This is the engine's
    // ops::detail::peer_exchange_sum_kernel with ops::detail::peer_exchange_blocks geometry --
    // the shipped code, not a lookalike.
    auto launch_pair = [&](int slot, int elem_count) {
        __nv_bfloat16* publish[2]  = {slab.payload[0] + static_cast<std::size_t>(slot) *
                                                          slab.slot_elems,
                                      slab.payload[1] + static_cast<std::size_t>(slot) *
                                                            slab.slot_elems};
        std::uint32_t* flag_ptr[2] = {&slab.flag[0][slot], &slab.flag[1][slot]};
        const int bytes            = elem_count * static_cast<int>(sizeof(__nv_bfloat16));
        const int vecs             = bytes / static_cast<int>(sizeof(ninfer::ops::detail::PeerVec));
        const int blocks           = g_override_blocks > 0 ? g_override_blocks
                                          : ninfer::ops::detail::peer_exchange_blocks(bytes);
        for (int rank = 0; rank < 2; ++rank) {
            const int dev = rank == 0 ? dev_a : dev_b;
            CK(cudaSetDevice(dev));
            ninfer::ops::detail::peer_exchange_sum_kernel<<<blocks, 256, 0, stream[rank]>>>(
                reinterpret_cast<ninfer::ops::detail::PeerVecBf16*>(side[rank].partial),
                reinterpret_cast<ninfer::ops::detail::PeerVecBf16*>(side[rank].result),
                reinterpret_cast<ninfer::ops::detail::PeerVecBf16*>(publish[rank]),
                flag_ptr[rank],
                reinterpret_cast<const ninfer::ops::detail::PeerVecBf16*>(publish[1 - rank]),
                flag_ptr[1 - rank], arrival[rank] + slot, slab.hang_all, vecs);
            CK(cudaGetLastError());
        }
    };

    auto sync_both = [&] {
        for (int rank = 0; rank < 2; ++rank) {
            const int dev = rank == 0 ? dev_a : dev_b;
            CK(cudaSetDevice(dev));
            CK(cudaStreamSynchronize(stream[rank]));
        }
    };

    auto verify = [&](const char* label) {
        bool pass = true;
        for (int rank = 0; rank < 2; ++rank) {
            const int dev = rank == 0 ? dev_a : dev_b;
            CK(cudaSetDevice(dev));
            std::vector<__nv_bfloat16> out(elems);
            CK(cudaMemcpy(out.data(), side[rank].result, elems * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < elems; ++i) {
                const __nv_bfloat16 expected = __float2bfloat16_rn(
                    __bfloat162float(host_partial[0][i]) + __bfloat162float(host_partial[1][i]));
                if (__bfloat162float(out[i]) != __bfloat162float(expected)) {
                    if (bad < 4) {
                        printf("  %s rank %d MISMATCH at %d: got %.6f expected %.6f\n", label, rank,
                               i, __bfloat162float(out[i]), __bfloat162float(expected));
                    }
                    ++bad;
                }
            }
            if (bad != 0) {
                printf("  %s rank %d: %d/%d elements wrong\n", label, rank, bad, elems);
                pass = false;
            }
        }
        if (pass) { printf("  %s: sums exact vs BF16 reference on both ranks\n", label); }
        return pass;
    };

    // Host-side reset between chains: pinned WB words are coherent with the GPUs' PCIe view.
    auto reset_flags = [&] {
        for (int slot = 0; slot < kChain; ++slot) { slab.flag[0][slot] = 0; slab.flag[1][slot] = 0; }
        *slab.hang_all = 0;
    };

    auto report_hang = [&](const char* where) {
        if (*slab.hang_all != 0) {
            printf("HANG GUARD at %s (poller gave up waiting for the peer)\n", where);
            return true;
        }
        return false;
    };

    bool ok = true;

    // ---- 1. single eager exchange -------------------------------------------------------
    {
        reset_flags();
        const auto start = std::chrono::high_resolution_clock::now();
        launch_pair(0, elems);
        sync_both();
        printf("single exchange pair (launch->both-sync): %.2f us\n", ms_since(start) * 1000.0);
        ok = ok && verify("eager single");
        if (report_hang("eager single")) { return 1; }
    }

    // ---- 2. 128 consecutive eager exchanges --------------------------------------------
    {
        reset_flags();
        for (int i = 0; i < kChain; ++i) { launch_pair(i, elems); }
        const auto start = std::chrono::high_resolution_clock::now();
        sync_both();
        const double ms = ms_since(start);
        printf("eager chain x%d: %.3f ms total, %.2f us per exchange\n", kChain, ms,
               ms * 1000.0 / kChain);
        ok = ok && verify("eager chain");
        if (report_hang("eager chain")) { return 1; }
    }

    // ---- 3. the same 128 inside ONE cross-device CUDA graph ------------------------------
    {
        cudaEvent_t fork = nullptr;
        cudaEvent_t join = nullptr;
        CK(cudaSetDevice(dev_a));
        CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        // The join event is recorded on the PEER's stream, so it must be created with the peer
        // device current: a CUDA event belongs to the context it was created in.
        CK(cudaSetDevice(dev_b));
        CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));

        reset_flags();
        printf("[capture] begin cross-device capture...\n");
        CK(cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeThreadLocal));
        CK(cudaEventRecord(fork, stream[0]));
        CK(cudaSetDevice(dev_b));
        CK(cudaStreamWaitEvent(stream[1], fork, 0));
        for (int i = 0; i < kChain; ++i) { launch_pair(i, elems); }
        CK(cudaSetDevice(dev_b));
        CK(cudaEventRecord(join, stream[1]));
        CK(cudaSetDevice(dev_a));
        CK(cudaStreamWaitEvent(stream[0], join, 0));
        cudaGraph_t graph = nullptr;
        cudaError_t end = cudaStreamEndCapture(stream[0], &graph);
        if (end != cudaSuccess) {
            fprintf(stderr, "capture failed: %s\n", cudaGetErrorString(end));
            return 1;
        }
        printf("[capture] end ok, instantiating...\n");
        cudaGraphExec_t exec = nullptr;
        CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        printf("[capture] instantiated.\n");

        size_t node_count = 0;
        CK(cudaGraphGetNodes(graph, nullptr, &node_count));
        printf("captured cross-device graph: %zu nodes (%d exchange pairs)\n", node_count, kChain);

        // Warm replay, then timed replays. Flags reset on the host between replays -- the same
        // point the engine resets at, once both devices have retired the previous round.
        reset_flags();
        printf("[replay] warm launch...\n");
        CK(cudaGraphLaunch(exec, stream[0]));
        sync_both();
        printf("[replay] warm done.\n");
        ok = ok && verify("graph replay 1");
        reset_flags();

        constexpr int kReplays = 5;
        double total = 0.0;
        for (int r = 0; r < kReplays; ++r) {
            reset_flags();
            const auto start = std::chrono::high_resolution_clock::now();
            CK(cudaGraphLaunch(exec, stream[0]));
            CK(cudaStreamSynchronize(stream[0]));
            total += ms_since(start);
        }
        printf("graph replay x%d (%d exchanges each): %.3f ms/replay, %.2f us per exchange\n",
               kReplays, kChain, total / kReplays, total / kReplays * 1000.0 / kChain);
        ok = ok && verify("graph replay N");
        if (report_hang("graph")) { return 1; }

        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
        CK(cudaEventDestroy(fork));
        CK(cudaEventDestroy(join));
    }

    // ---- 4. comparison: staged memcpy path inside a graph (current transport's transfer) --
    // The production collective is cudaMemcpyAsync(D2D over UVA) between two event hops; this
    // measures what the copy engine pays for the same 10 KiB x 2 directions inside one graph,
    // which is the number the mailbox has to beat.
    {
        for (int rank = 0; rank < 2; ++rank) {
            const int dev = rank == 0 ? dev_a : dev_b;
            CK(cudaSetDevice(dev));
            CK(cudaMemcpy(side[rank].result, side[rank].partial, elems * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice));
        }
        CK(cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < kChain; ++i) {
            for (int rank = 0; rank < 2; ++rank) {
                const int dev = rank == 0 ? dev_a : dev_b;
                CK(cudaSetDevice(dev));
                // Pull form, like the engine: rank r reads the peer's operand into its own store.
                CK(cudaMemcpyAsync(side[rank].result, side[1 - rank].partial,
                                   elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice,
                                   stream[rank]));
            }
        }
        cudaGraph_t graph = nullptr;
        cudaError_t end = cudaStreamEndCapture(stream[0], &graph);
        if (end != cudaSuccess) {
            fprintf(stderr, "memcpy capture failed: %s\n", cudaGetErrorString(end));
            return 1;
        }
        cudaGraphExec_t exec = nullptr;
        CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        CK(cudaGraphLaunch(exec, stream[0]));
        CK(cudaStreamSynchronize(stream[0]));
        double total = 0.0;
        constexpr int kReplays = 5;
        for (int r = 0; r < kReplays; ++r) {
            const auto start = std::chrono::high_resolution_clock::now();
            CK(cudaGraphLaunch(exec, stream[0]));
            CK(cudaStreamSynchronize(stream[0]));
            total += ms_since(start);
        }
        printf("STAGED memcpy graph (128 x 2 x 10 KiB pulls): %.3f ms/replay, %.2f us per "
               "pull-pair\n",
               total / kReplays, total / kReplays * 1000.0 / kChain);
        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
    }

    // ---- 5. flag round-trip alone --------------------------------------------------------
    {
        reset_flags();
        launch_pair(0, 0);
        sync_both();
        double total = 0.0;
        constexpr int kReps = 200;
        for (int i = 0; i < kReps; ++i) {
            reset_flags();
            const auto start = std::chrono::high_resolution_clock::now();
            launch_pair(0, 0);
            sync_both();
            total += ms_since(start);
        }
        printf("empty exchange (flag RTT + launch floor): %.2f us\n", total / kReps * 1000.0);
        if (report_hang("empty")) { return 1; }
    }

    // ---- 6. MTP-verify width: 40 KiB -----------------------------------------------------
    {
        // The slab's slots are sized for the 40 KiB shape, so this exchange is in-bounds and its
        // numeric result is checkable at the 10 KiB prefix (operands only fill the first
        // quarter of the payload area; the remaining lanes re-read the same values).
        auto launch_wide = [&](int slot) {
            __nv_bfloat16* publish[2]  = {slab.payload[0] + static_cast<std::size_t>(slot) *
                                                              slab.slot_elems,
                                          slab.payload[1] + static_cast<std::size_t>(slot) *
                                                                slab.slot_elems};
            std::uint32_t* flag_ptr[2] = {&slab.flag[0][slot], &slab.flag[1][slot]};
            const int bytes            = kElemsMtp * static_cast<int>(sizeof(__nv_bfloat16));
            const int vecs = bytes / static_cast<int>(sizeof(ninfer::ops::detail::PeerVec));
            const int blocks = g_override_blocks > 0 ? g_override_blocks
                                  : ninfer::ops::detail::peer_exchange_blocks(bytes);
            for (int rank = 0; rank < 2; ++rank) {
                const int dev = rank == 0 ? dev_a : dev_b;
                CK(cudaSetDevice(dev));
                ninfer::ops::detail::peer_exchange_sum_kernel<<<blocks, 256, 0, stream[rank]>>>(
                    reinterpret_cast<ninfer::ops::detail::PeerVecBf16*>(side[rank].partial),
                    reinterpret_cast<ninfer::ops::detail::PeerVecBf16*>(side[rank].result),
                    reinterpret_cast<ninfer::ops::detail::PeerVecBf16*>(publish[rank]),
                    flag_ptr[rank],
                    reinterpret_cast<const ninfer::ops::detail::PeerVecBf16*>(publish[1 - rank]),
                    flag_ptr[1 - rank], arrival[rank] + slot, slab.hang_all, vecs);
                CK(cudaGetLastError());
            }
        };
        reset_flags();
        double total = 0.0;
        constexpr int kReps = 50;
        for (int i = 0; i < kReps; ++i) {
            reset_flags();
            const auto start = std::chrono::high_resolution_clock::now();
            launch_wide(0);
            sync_both();
            total += ms_since(start);
        }
        printf("40 KiB exchange pair (eager): %.2f us\n", total / kReps * 1000.0);
        if (report_hang("40 KiB eager")) { return 1; }

        // The same chain of 128 WIDE exchanges captured into one cross-device graph: this is
        // the MTP verify round's real shape (the decode graph's collectives, at 4 positions).
        {
            cudaEvent_t fork = nullptr;
            cudaEvent_t join = nullptr;
            CK(cudaSetDevice(dev_a));
            CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
            CK(cudaSetDevice(dev_b));
            CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
            reset_flags();
            CK(cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeThreadLocal));
            CK(cudaEventRecord(fork, stream[0]));
            CK(cudaSetDevice(dev_b));
            CK(cudaStreamWaitEvent(stream[1], fork, 0));
            for (int i = 0; i < kChain; ++i) { launch_wide(i); }
            CK(cudaEventRecord(join, stream[1]));
            CK(cudaSetDevice(dev_a));
            CK(cudaStreamWaitEvent(stream[0], join, 0));
            cudaGraph_t wide_graph = nullptr;
            cudaError_t end = cudaStreamEndCapture(stream[0], &wide_graph);
            if (end != cudaSuccess) {
                fprintf(stderr, "wide capture failed: %s\n", cudaGetErrorString(end));
                return 1;
            }
            cudaGraphExec_t wide_exec = nullptr;
            CK(cudaGraphInstantiate(&wide_exec, wide_graph, nullptr, nullptr, 0));
            reset_flags();
            CK(cudaGraphLaunch(wide_exec, stream[0]));
            sync_both();
            constexpr int kWideReplays = 5;
            double wide_total = 0.0;
            for (int r = 0; r < kWideReplays; ++r) {
                reset_flags();
                const auto start = std::chrono::high_resolution_clock::now();
                CK(cudaGraphLaunch(wide_exec, stream[0]));
                CK(cudaStreamSynchronize(stream[0]));
                wide_total += ms_since(start);
            }
            printf("40 KiB graph replay x%d (%d exchanges each): %.3f ms/replay, %.2f us per "
                   "exchange\n",
                   kWideReplays, kChain, wide_total / kWideReplays,
                   wide_total / kWideReplays * 1000.0 / kChain);
            if (report_hang("40 KiB graph")) { return 1; }
            CK(cudaGraphExecDestroy(wide_exec));
            CK(cudaGraphDestroy(wide_graph));
            CK(cudaEventDestroy(fork));
            CK(cudaEventDestroy(join));
        }
    }

    free_host_slab(slab);
    for (int rank = 0; rank < 2; ++rank) {
        const int dev = rank == 0 ? dev_a : dev_b;
        CK(cudaSetDevice(dev));
        CK(cudaStreamDestroy(stream[rank]));
        CK(cudaFree(side[rank].partial));
        CK(cudaFree(side[rank].result));
    }

    printf(ok ? "mailbox_probe: PASS\n" : "mailbox_probe: FAIL (numeric mismatch)\n");
    return ok ? 0 : 1;
}
