# Prefill research #2 — new hypotheses, tested (2026-09-20)

Scope: this document contains only *new* work. The pinned CE relay from
`research-prefill-improvements.md` is now the production baseline (in
`src/ops/common/allreduce.cu`, `NINFER_TP2_RELAY=0` disables it). Hardware: 2× RTX 5060 Ti 16 GB,
Windows 11 WDDM, no P2P, both links PCIe 3.0 ×8 (~7.0 GB/s pinned each direction per device).

All numbers below are from `temp\prefill_sliced_relay_bench.exe` (new, self-contained,
source `temp\prefill_sliced_relay_bench.cu`, verifies the BF16 sum exactly against a CPU
reference before timing) and from server sweeps (`temp\sweep2_c1024.csv`, `temp\sweep2_c4096.csv`)
on the production server with the relay enabled.

---

## Hypothesis H1 — sliced full-duplex relay (intra-collective pipelining): REFUTED

**Reasoning.** PCIe is electrically dual-simplex (verified: each direction has dedicated lanes),
so D2H and H2D should each sustain ~7 GB/s *simultaneously*. The current relay serializes D2H
then H2D on one stream per rank — 2T for a T-sized one-way transfer. If I slice each 10 MiB
collective into S slices and pipeline them across two streams (D2H slice i+1 overlaps H2D
slice i), wall time should approach T(1+1/S) ≈ 1.9 ms vs the measured 3.3 ms serial. This is how
NCCL chunks channels internally, so it was my best idea.

**Measurement first — duplex ground truth.** Before building the pipeline I measured the
assumption: issue a full-payload D2H and a full-payload H2D *at the same time* on two streams:

| Payload | D2H alone | H2D alone | **both at once (d)** | serial relay (a) |
|---|---|---|---|---|
| 5 MiB | ~0.75 ms | ~0.78 ms | **1.58 ms** | 1.82 ms |
| 10 MiB | ~1.50 ms | ~1.54 ms | **3.22 ms** | 3.50 ms |
| 20 MiB | ~2.96 ms | ~3.08 ms | **6.49 ms** | 6.59 ms |

Simultaneous both-direction transfer takes the same wall time as the two directions serialized —
aggregate bandwidth stays ~6.5 GB/s. **The link does not duplex on this machine** (whether the
limit is the copy engines, WDDM packet scheduling, or the chipset/root-complex path, the measured
fact stands). The full-duplex assumption is false, so no scheduling trick can approach T; the
sliced relay confirmed it:

| 10 MiB variant | µs/call |
|---|---|
| serial relay (production) | 3,496 |
| sliced S=2 / S=4 / S=8, combine at end | 3,156 / 3,202 / 3,202 |
| sliced S=16 | 3,631 |
| sliced + per-slice fused combine (any S) | worse (3,350–6,232) |

Best case ~10% (S=2), within run-to-run noise of the serial relay on other runs; per-slice
combine is actively harmful (SM work interferes with copy-engine progress). **Not adopted.**
Consequence: the "full-duplex pipelined relay, +20–50%" estimate in research doc #1 is capped at
~10% by hardware; do not spend effort there.

## Corollary — the collective is already at the hardware floor

The serial relay moves 2×10 MiB per collective at the measured duplex-limited ~6.4 GB/s aggregate
= 3.28 ms theoretical; production measures 3.29 ms. **The TP2 allreduce is at ~100% of what this
machine's host link can physically do for this byte count.** Any further collective-time win must
move *fewer bytes* or *fewer collectives*, not schedule them better. (Mailbox/zero-copy SM
transport already measured 2.3× *slower* at these sizes — research doc #1.)


## Correction to research doc #1, lever 3 (sequence parallelism)

The claim "RS+AG halves bytes → +40–50% prefill" does not hold at **TP=2**: ring-allreduce between
2 ranks moves B/2+B/2 = B per direction; the current pull-based allreduce also moves exactly B per
direction. Reduce-scatter + all-gather moves B/2 + B/2 = B per direction. **Identical link bytes.**
SP would only halve *activation memory* and enable SP GEMM layouts — real work, no measured
prefill win to expect from the transport side at tp2. Downgraded.

## Hypothesis H4 — wider prefill chunk (4096): CONFIRMED, small win

Fewer, larger chunks = fewer collective *calls* (128/chunk regardless of width), fewer kernel
launches, better GEMM shapes. Measured (production server, relay on, max_tokens=1, dflash2):

| Prompt tokens | chunk 1024 | chunk 4096 | Δ |
|---|---|---|---|
| 1,088 | 1,160 ms | 1,052 ms | −9.3% |
| 4,144 | 3,094 ms | 2,972 ms | −3.9% |
| 17,071 | 11,730 ms | 11,263 ms | −4.0% |

Marginal rate: **0.661 → 0.639 ms/token (1,512 → 1,565 tok/s, +3.5%)**; fixed overhead ~0.35 s
unchanged. Cost: runtime reservation grows ~0.7 GiB, so at 16 GB cards `--max-context` must drop
from 110,000 to ≈ 65,536 (the startup preflight computes the exact ceiling; 110K fails with
"rank 0 is 656 MiB short"). Recommendation: use `--prefill-chunk 4096` when you don't need >64K
context; keep 1024 for the 110K configuration. Chunk 2048 sits strictly between (doc #1: +4%).

## What is NOT the bottleneck (measured, so nobody re-spends time)

- **DFlash2 feature extraction per chunk**: 0.3% (doc #1, session C).
- **Host submission at 1024-chunk**: if host-bound, doubling chunk width would nearly double
  throughput; it gains 4%. Engine JSONL shows `prefill_device_wait ≈ 0` only because submission
  is async, not because the device is idle.
- **Combine kernel**: ~0.2 ms of each 3.3 ms collective; fusing it into the H2D stream made
  things *worse* (H1 table). Moving it off the critical stream could save ≲6% of collective time
  (~+2% prefill) — small, noted for completeness.
- **The fixed ~0.35–0.41 s/request**: scales with nothing (tokenization, KV reset, final logits +
  sampling, HTTP). It dominates only sub-1K prompts (450 ms TTFT at 144 tokens). Worth ~1 day of
  profiling if short-prompt TTFT matters; untouched by any collective work.
## The one big structural idea left — cross-chunk wavefront pipelining (H18, analysis only)

The prefill dependency graph is *not* a pure chain. Chunk k+1s layer-l input needs only
(a) chunk ks layer-l output (residual stream, per layer) and (b) chunk k+1s own layer l-1
output; its layer-l attention needs chunk ks K/V *at layer l*, which exist as soon as chunk k
has passed layer l. So chunk k+1 may legally trail chunk k by one layer: while chunk k sits in
layer ls collective (links busy, SMs idle — GPU util was measured 82-83% at 74-78 W during a 16K
prefill), chunk k+1 could be running layer l-1s GEMMs (SMs busy, links idle). Perfect wavefront
overlap would hide most of the 65% collective share behind the 32% compute share of the *other*
chunk → prefill ceiling ≈ max(compute, collectives) instead of compute + collectives ≈ **+40-60%
on long prompts**. This is the prefill analogue of pipeline-parallel interleaving (Chimera-style)
and, notably, it needs no new transport — it reuses the existing collectives with two in-flight
chunk states (double the per-chunk activation workspaces, staggered KV appends, careful causal
masking at the boundary). It is a multi-day scheduler change in
`src/targets/qwen3_6/impl/runtime/`, not a transport tweak. Estimated effort 3-5 days; expected
ceiling +40-60% long-prompt prefill; bit-exactness preserved (same kernels, same order per token
within each chunks own layer sequence).

## Verified current state (all reproducible)

| Config | Marginal prefill | 17.1K-prompt TTFT |
|---|---|---|
| Original staged (research doc #1) | 0.691 ms/tok (1,446 tok/s) | 12.36 s |
| + pinned relay (now production) | 0.659 ms/tok (1,518 tok/s) | 11.81 s |
| + `--prefill-chunk 4096` | 0.639 ms/tok (1,565 tok/s) | 11.26 s |

Correctness: relay is bit-exact (identical SHA256 completion, 8,392-token prompt, relay on/off;
bench SHAs stable across rounds). The sliced-relay bench verifies every timed variant against a
CPU BF16-reference before reporting a number. Total measured gain so far: **+8.0% prefill
throughput, -1.1 s TTFT at 17K tokens**, with a documented hardware-floor argument for why the
collective itself cannot be scheduled any faster, and one structural idea (wavefront) with a
+40-60% ceiling left on the table.

## Reproduce

```powershell
# transport/hypothesis microbench (self-contained, no engine libs)
cmd /c temp\build_sliced.cmd; temp\prefill_sliced_relay_bench.exe 0 1   # log: temp\sliced.log
# server sweeps
.\temp\start-bench-server.ps1 -Port 30005 -Tag c1024                    # relay on by default
.\temp\bench-prefill-sweep.ps1 -Port 30005 -TargetTokens @(1024,4096,16384) -MaxTokens 1
.\temp\start-bench-server.ps1 -Port 30005 -Tag c4096 -PrefillChunk 4096 -MaxContext 65536
```
