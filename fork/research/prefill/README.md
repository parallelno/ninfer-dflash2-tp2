# Prefill research #3 — measured stage decomposition and the levers that actually move TTFT

Hardware: 2× RTX 5060 Ti 16 GB, Windows 11 WDDM, no P2P, both GPUs on PCIe 3.0 ×8, host
2× Xeon E5‑2650 v2 (2013, 2.6 GHz). Build `build-tp2-check`, artifact
`model\qwen3_8_27b_nvfp4.dflash2.ninfer`, `--tp 2 --kv-dtype int8 --spec dflash2 --draft-tokens 4`.
Every number below comes from the engine's own `--request-log-jsonl` records (`timings_seconds`,
`engine_timing`) or from a self‑contained CUDA microbench in this directory. Nothing is extrapolated.

Prior work (branches `prefill_boost`, `prefill_boost2`, deleted from `main`) established:
pinned relay +5 %, chunk 4096 +3 %, mailbox transport 2.3× slower, sliced/duplex relay no gain,
cross‑chunk wavefront prototype crashed. This document only adds what was not measured before.

## 1. Baseline (default flags, prefix reuse on, `--max-context 32768`)

| prompt tok | prefill s | prefill units | ms/tok |
|---:|---:|---:|---:|
| 90 | 0.47 | 3 | 5.22 |
| 537 | 1.13 | 3 | 2.10 |
| 1,037 | 1.67 | 4 | 1.61 |
| 4,175 | 3.70 | 7 | 0.89 |
| 8,488 | 6.10 | 11 | 0.72 |
| 16,663 | 12.38 | 19 | 0.74 |

Linear fit: **0.805 s fixed + 0.682 ms/token (1,467 tok/s marginal)**.

## 2. The fixed cost is the prefix‑reuse checkpoint machinery, not HTTP/tokenize/sampling

Same server, `--no-prefix-reuse`:

| prompt tok | prefill s | units | ms/tok |
|---:|---:|---:|---:|
| 90 | 0.25 | 1 | 2.81 |
| 537 | 0.53 | 1 | 0.99 |
| 1,037 | 0.93 | 2 | 0.90 |
| 4,175 | 2.95 | 5 | 0.71 |
| 16,663 | 11.75 | 17 | 0.70 |

Fit: **0.153 s fixed + 0.691 ms/token**. The marginal rate is identical; the intercept drops
**0.805 → 0.153 s (−0.65 s per request)**. Short prompts are 2× faster (90 tok: 470 → 250 ms;
537 tok: 1.13 → 0.53 s).

Mechanism (`request_plan_impl.h::segmented_prefill_chunks`, `program_impl.h::advance_prefill`):
with reuse on, every prompt carries a `TurnClosure` rewrite checkpoint at the generation
prologue (`chat_template.cpp:655`) plus shared‑prefix capture candidates. Each capture frontier
**splits the prompt into extra chunks**: `units.prefill` is always `chunks + 2` with reuse on vs
`chunks` off. A tiny chunk still runs the whole 64‑layer schedule with 128 collectives; a
7‑token chunk measured **212 ms** (earlier `wave8` log) and the chunk floor with all transfers
disabled is ~0.2 s (§4). Two forced tiny chunks ≈ 0.4 s, plus state fork/capture ≈ 0.65 s total.

Partial switches: `--max-shared-prefixes 0` alone → intercept 0.434 s;
`--max-private-continuations 1 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0`
→ 0.283 s (units = chunks+1: only the TurnClosure split remains).

## 3. …but prefix reuse never hits under the default flags — and it is the biggest lever

Default context‑cache flags: `private 2 | shared 4 | anchors 2 | 1 cached device state`. In
every multi‑turn shape tested (identical prompt repeated; conversation extended with
assistant+user; with a system message; with tool_calls/tool results; 5‑turn agent loop) the
server reported `cache 0 (0.0%)`, `prefix_reuse_path=root` **every time**, at both
`--max-context 32768` and `110000`. The JSONL shows why: every request after the first runs a
pressure search with `selected_degradation_units=2`, `private_owners_evicted=1`,
`checkpoints_dropped=1` — the planner evicts the only retained checkpoint before the next turn
can use it. The default configuration therefore pays the full 0.65 s of capture overhead on
every request and recovers nothing.

Configurations that make reuse work (client wall, 24‑48 output tokens):

| flags | identical repeat 4K | +1 turn 4K | agent turns 1‑5 (6.2K→8K) | A→B→A conversations |
|---|---|---|---|---|
| default | 3.9 s (root) | 3.5 s (root) | 5.0–6.3 s (root) | all root |
| **`--max-private-continuations 1 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0`** | **0.84 s** (TTFT 0.38) | **0.85 s** (TTFT 0.46) | **0.9–1.1 s** (TTFT 0.62–0.65, 95 % hit) | A hits, B hits, A‑again: turn 0 root, then hits |
| `--device-state-slots 2` (+ defaults) | 0.62 s | 1.29 s | turn 1 hit, then `shared_stable_prefix` at 1.7–1.9 s under pressure | mixed |
| `--device-state-slots 4 --max-private-continuations 4 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` (fits only at `--max-context ≤ 65536`) | 0.88 s | 0.82 s | turns 1‑3 root, 4‑5 hit | A, B hit; A‑again root |

The bold row is the one to ship for the single‑user agent workload: no extra VRAM (works at
`--max-context 110000`), a 6.5K‑token agent turn goes from **5.3 s to 1.0 s wall**, identical
prompts from 3.9 s to 0.84 s. Removing the shared‑prefix capture work also makes a cache *miss*
~0.15 s cheaper than default.

Correctness check on the reuse path: a turn‑closure hit produced a slightly different greedy
continuation than a from‑root recompute of the same 4,250‑token prompt (same meaning, wording
diverges after ~20 tokens); root‑vs‑root was identical. This is the known TP2 ordinary‑split
last‑bit drift (`tests/.../test_engine_mtp_tp2_real.cpp`) amplified by greedy decoding — the
restored checkpoint and the recomputed residual differ in the last BF16 bit somewhere. Not a
bug, but reuse hits are not bit‑reproducible against a cold run.

The MTP reference (`ninfer-windows-tp2`, MTP K=4, `--max-concurrency 1`) logs
`reuse=full_reset` on all nine requests of the same A→B→A workload: it has no working prefix
reuse, so the 836 tok/s "prefill" in `BENCHMARKS.md` is pure recompute and is beatable ~5× per
agent turn with the flags above.
## 4. Marginal cost decomposition, measured in the server (not extrapolated)

Diagnostic switch added to `src/ops/common/allreduce.cu`: `NINFER_TP2_DIAG_NO_TRANSFER=1`
skips the cross‑device `cudaMemcpyAsync` inside every eager `allreduce_sum` but keeps the
4‑event choreography and the combine kernel (output is numerically wrong; used only to time
the compute‑only schedule). `--no-prefix-reuse` in all rows:

| config | fixed | ms/tok | marginal tok/s | GPU util / power @16K |
|---|---:|---:|---:|---|
| production (1024 chunk) | 0.153 s | 0.691 | 1,446 | 88 % / 84 W |
| no transfer, 1024 chunk | 0.119 s | 0.259 | 3,855 | — |
| no transfer, 4096 chunk | 0.066 s | 0.256 | 3,901 | 99 % / 137 W |
| no transfer, 128 chunk | 0.184 s | 0.874 | 1,145 | — |

Per token: **0.43 ms (62 %) is the collective transfer, 0.26 ms (38 %) is everything else**
(GEMMs, attention/GDN, KV append, DFlash features, host issue). 3.9K tok/s is the hard ceiling
for any scheme that hides but does not remove the transfer (wavefront, micro‑batch overlap):
**+2.7× at most**.

The 128‑chunk row shows the eager schedule is **host‑bound below ~256 tokens per chunk**: with
transfers off, 128‑token chunks cost 0.87 ms/tok vs 0.26 at 1024. `launch_bench.exe` shows
why: on this Xeon+WDDM a kernel launch costs **23 µs median / 60 µs p90**, a same‑stream
`cudaEventRecord+cudaStreamWaitEvent` pair 26 µs, `cudaMemsetAsync` 20 µs. `floor_bench (b)`:
32 tiny launches alternating devices = 1.34 ms → a 64‑layer chunk with ~30 launches per rank
per layer is **~130 ms of pure host issue per chunk regardless of width**. That is most of the
0.15 s residual fixed cost and most of the ~0.2 s per‑chunk floor.

## 5. Can the transfer overlap with compute? Yes — 92‑95 % of it hides

`overlap_bench.exe`: DRAM‑streaming kernels on stream C, production‑style staged pulls with the
4‑event choreography on an independent stream X, both devices:

| payload | compute only | staged pull only | both, independent streams | copy hidden |
|---:|---:|---:|---:|---:|
| 10 MiB (1024‑tok chunk) | 187.5 ms | 113.4 ms | 193.7 ms | **95 %** |
| 5 MiB | 93.4 | 60.4 | 97.4 | 93 % |
| 2.5 MiB | 46.9 | 35.7 | 49.6 | 92 % |

The copy engine and SMs run concurrently on this WDDM machine. **The current prefill gets none
of this** because each layer's `allreduce_sum` sits on the critical path of the same stream:
layer l+1's GEMMs cannot start until layer l's reduce lands.

Caveats measured in the same benches:

- Issue order matters: copies enqueued *before* compute (`3b`) serialize (313 ms vs 194) — WDDM
  drains in submission order when the copy blocks the stream. Compute must be enqueued first.
- The staged D2D pull **blocks the calling host thread for ~45 % of its duration** (0.79 of
  1.75 ms at 10 MiB; `issue_bench (a)`): one thread issuing 128 per chunk spends ~100 ms/chunk
  inside the driver. The pinned‑relay form blocks 24 µs. Issuing the two directions from two
  host threads is *worse* (4.27 vs 3.47 ms per pair) — WDDM serializes them.

## 6. Not worth doing (measured)

- **nsys on the server**: process dies before `--delay` even with `--sample=none`; 55 KB report,
  no CUDA trace. Same as the previous attempt. Use the JSONL timings plus standalone benches.
- **4‑event choreography without copy**: 176 µs each, 26 ms per chunk (`floor_bench (a)`), ~3 %.
- **Pageable H2D of ids/positions** (`copy_i32`): 25 µs each, negligible.
- **DFlash2 feature extraction**: 0.3 % (earlier session, `reqlog-c`).
- **Chunk 2048/4096 with transfers on**: +3‑4 %, costs ~0.7 GiB of context. With transfers off
  the 4096 chunk pushes GPU power 84 → 137 W: the GEMMs are fine, the serialized link idles them.
## 7. Recommendations, ranked by measured gain per unit of work

1. **Ship the reuse flags** in `fork/start-dflash2.ps1`:
   `--max-private-continuations 1 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0`.
   Zero code, zero VRAM, works at 110K context. Agent‑turn TTFT 5.3 s → 0.65 s (−88 %);
   cache‑miss fixed cost 0.80 → 0.28 s. Verified on 9‑request A→B→A and 6‑turn agent loops.
   The default's failure (pressure planner evicting the only private owner with
   `private 2 | shared 4 | 1 cached slot`) is a scheduler policy issue worth its own ticket.
2. **Intra‑chunk micro‑batch pipelining** (medium): run each 1024 chunk as two 512 halves
   staggered by one collective on two streams per rank, so half B's layer‑l GEMMs overlap half
   A's layer‑l reduce. §5 proves 92‑95 % of the copy hides; §4 caps the win at the compute‑only
   rate. Theoretical marginal ≈ max(compute, transfer) ≈ 0.43 ms/tok → **~2,300 tok/s (+60 %)**.
   Unlike the cross‑chunk wavefront it needs no KV/GDN‑state boundary handling: both halves are
   one chunk, half B's attention sees half A's K/V (appended at layer l before B's layer l runs).
   Enqueue compute before copies and use the pinned relay so the host thread is not blocked.
3. **Host‑issue reduction** (medium): ~130 ms/chunk of launch overhead is ~19 % of a 1024 chunk
   and ~65 % of a 128 chunk. Options: CUDA‑graph the eager per‑layer prefill body for the fixed
   1024 width, or issue rank 1's launches from a second host thread (concurrent *launches* were
   fine here; only concurrent *copies* serialized).
4. **Skip the TurnClosure chunk split** when the frontier falls inside the last few tokens
   (capture from the tail hidden instead of running a separate 7‑token chunk): ~0.13 s per
   request with reuse on. Touches `advance_prefill`.

## 8. Implemented: `--prefill-pipeline` and the pinned relay (2026‑09‑21)

Both are optional so before/after runs can be made on the same binary.

**`--prefill-pipeline`** (server flag → `EngineOptions::prefill_pipeline`, tp 2 only). Each
eager prefill chunk ≥ 256 tokens is split into a leading half A and a trailing half B. A runs on
the ordinary per‑rank streams; B runs on a second lane — a second `ExecutionContext` on the same
two devices (= second compute stream per rank), its own `PeerEvents`, and a per‑rank arena sized
for a half‑chunk layer body (`WorkspacePlan::prefill_pipeline_lane`, ~60 MiB on rank 0 at chunk
1024). Issue order `A(0) A(1)B(0) A(2)B(1) … A(63)B(62) B(63)`; B(l) waits on a per‑parity event
recorded after A(l) (it reads A's K/V rows and continues A's GDN state, so B reads/writes the
chunk's *destination* slot while A reads *source* → *destination*). Rank‑0 join before the final
norm. Multimodal chunks and tails < 256 tokens take the plain loop. Implementation:
`TextContext::run_layers_tp2_pipelined` (`text_context_impl.h`), lane storage
`ProgramImplCore::PrefillPipelineStorage` (`program_impl.h`), the layer body now takes its
streams/events/arenas from the *active lane* accessors (`ec()`, `peer_events()`, `workspaces()`).

**`NINFER_TP2_RELAY=1`** (env, opt‑in). Eager ≥ 1 MiB `allreduce_sum` payloads exchange through
pinned host bounce buffers (D2H on the owner stream, H2D pull on the peer) instead of the
driver‑staged UVA `cudaMemcpyAsync`. Same 4‑event protocol, same combine kernel; bounce pairs are
keyed by `PeerEvents` instance so the two lanes never share one. Main effect: the issuing host
thread no longer blocks ~45 % of each copy (§5), which is what lets the pipeline's second lane
actually get issued ahead.

Measured (`--no-prefix-reuse`, 5‑point sweep 545–16,844 tokens, linear fit; all servers same
binary, back to back):

| config | fixed | ms/tok | marginal tok/s | vs baseline | 16.8K TTFT |
|---|---:|---:|---:|---:|---:|
| baseline (staged, no pipeline) | 0.138 s | 0.693 | 1,444 | — | 11.90 s |
| relay only | 0.168 | 0.667 | 1,498 | +4 % | 11.37 |
| pipeline only | 0.154 | 0.624 | 1,602 | **+11 %** | 10.78 |
| **pipeline + relay** | 0.158 | 0.564 | **1,774** | **+23 %** | **9.75** |
| pipeline + relay, chunk 2048 (needs ≤ 64K ctx) | 0.213 | 0.553 | 1,809 | +25 % | — |
| pipeline, transfers disabled (ceiling) | 0.132 | 0.261 | 3,835 | — | — |

GPU during a 16K prefill: util 82–85 %, power 84 → 94 W (was 88 % / 84 W) — the SMs now do work
during the copies. The gain is below the 92–95 % overlap the microbench showed because the
pipeline only overlaps B(l−1)'s *compute* with A(l)'s *collective*; the two halves' own
collectives still serialize on the same PCIe links and each half's GEMMs run at half width.

**Numerics.** Pipeline runs are deterministic (two servers, four prompts, identical greedy
output). Relay vs staged is bit‑identical (same four prompts). Pipeline vs no‑pipeline is *not*
bit‑identical: the halves run the GEMMs at width 512 instead of 1024, and NVFP4 GEMM tile
schedules differ by width — the same effect as `--prefill-chunk 512` vs 1024 without the
pipeline (measured: 2 of 4 prompts differ between chunk 512 and 1024 on the unmodified path;
pipeline‑1024 matches chunk‑512 on 2 of 4). This is last‑bit BF16 drift amplified by greedy
decoding, of the class the tp2 path already carries.

**End‑to‑end** (`fork/start-dflash2-vision2.ps1` defaults: pipeline + relay + reuse flags,
DFlash2 K=4, 106K context): cold 6.2K‑token agent turn TTFT 4.7 → 3.8 s; cold 4.2K prompt
3.4 → 2.6 s; reuse hits unchanged at 0.46–0.62 s; vision requests run the plain loop and work.
The lane costs ~60 MiB on rank 0, so `--max-context` drops from 110,000 to ~106,000 on 16 GB (105,000 with the vision tower).

## Reproduce

```powershell
# scripts live in fork/research/prefill; outputs go to fork/research/prefill/out
.\start-server.ps1 -Tag base -Port 30006
.\start-server.ps1 -Tag noreuse -Port 30006 -NoPrefixReuse
.\start-server.ps1 -Tag nt -Port 30006 -NoPrefixReuse -Env @{ NINFER_TP2_DIAG_NO_TRANSFER='1' }
.\start-server.ps1 -Tag reuse -Port 30006 -MaxContext 110000 -Extra @('--max-private-continuations','1','--max-shared-prefixes','0','--max-long-anchors-per-continuation','0')
.\start-server.ps1 -Tag pipe -Port 30006 -NoPrefixReuse -Extra @('--prefill-pipeline') -Env @{ NINFER_TP2_RELAY='1' }   # §8
.\compare-outputs.ps1 -Port 30006 -Tag pipe        # greedy outputs for A/B numerics (out\outputs-<tag>.json)
.\sweep.ps1 -Port 30006 -Tag base -TargetTokens @(64,512,1024,2048,4096,8192,16384) -Distinct
.\report.ps1 -Tags base,noreuse,nt              # per-request table + linear fit from the JSONL
.\gpu-sample.ps1 -Port 30006 -Tokens 16384      # nvidia-smi util/power during one prefill
.\agent-sim.ps1 -Port 30006 -Turns 6; .\two-convs.ps1 -Port 30006; .\repeat-probe.ps1 -Port 30006
Stop-Process -Id (Get-Content .\out\serve-<tag>.pid)

# microbenches (self-contained; run inside VsDevCmd, nvcc 13.4)
nvcc -arch=sm_120a -O2 -std=c++20 overlap_bench.cu -o overlap_bench.exe ; .\overlap_bench.exe 0 1 10 3
nvcc -arch=sm_120a -O2 -std=c++20 issue_bench.cu   -o issue_bench.exe   ; .\issue_bench.exe 0 1 10
nvcc -arch=sm_120a -O2 -std=c++20 floor_bench.cu   -o floor_bench.exe   ; .\floor_bench.exe 0 1
nvcc -arch=sm_120a -O2 -std=c++20 launch_bench.cu  -o launch_bench.exe  ; .\launch_bench.exe 0
```
