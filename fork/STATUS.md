# DFlash2 TP2 Setup Status

## Result

`ninfer-dflash2-tp2-port` serves Qwen3.8-27B NVFP4 with `--spec dflash2 --tp 2` across two
RTX 5060 Ti 16 GB GPUs (no P2P; PeerMailbox collectives), CUDA graphs on by default.

The helper scripts and this document live in `fork/` inside the repository; the model artifact
lives in `model/` at the repository root (git-ignored). All paths below are relative to the
repository root unless stated otherwise.

- `.\fork\start-dflash2.ps1` -> `listening on http://127.0.0.1:30001` in ~65-80 s (weights ~55-73 s,
  graph capture ~4.5 s, warmup ~0.3 s). `-NoCudaGraph` is the eager escape hatch (warmup ~4.6 s).
- `.\fork\test-dflash2.ps1 -Port 30001` -> `/health` ok and chat reply `NInfer DFlash2 is ready.`
- `determinism.ps1 -Runs 4` (200-token Fibonacci prompt, temperature 0; kept outside the repo) -> 4/4 identical.
- `.\fork\bench-dflash2.ps1` (4 prompts x 256 tokens x 2 rounds) -> identical SHA per prompt across
  rounds.

## Fixes in this fork

- `src/artifact/materializer.cpp`: restored per-plane (`placement.copies`) shard copies. The merged
  upstream loop copied the first N bytes of the full tensor into every TP shard -> NaN at tp2.
- `src/core/paged_kv_cache.{h,cpp}`, `src/targets/qwen3_6/impl/state/state_image.cpp`,
  `.../export/.../state_image.h`, `.../runtime/program_impl.h`: rank-1 KV page tables and
  StateImage slots are now mirrored (`attach_mirror`) on zero/copy/publish, so both ranks see the
  same context state. Before this, DFlash2 output at tp2 was non-deterministic run to run.
- `src/serve/serve_options.cpp`: at `--tp 2` host state/KV tiers default to 0 unless given
  explicitly (host tiers crashed at startup).
- `src/core/arena.cu`: device arenas are zeroed once at allocation (defensive; no behavior change).
- `src/ops/softmax_attention/dense/causal_cache/small_t.cu`: the INT8 small-T decode kernel's
  `cudaFuncAttributeMaxDynamicSharedMemorySize` was set once per process (`static const`), i.e.
  only on whichever GPU ran first. Rank 1 then launched the 64 KiB dynamic-arena route (windows
  2054-8198 keys, TokenTile>=6) with the default 48 KiB limit -> `cudaErrorInvalidValue` on any
  prompt longer than ~2K tokens. Now set per device.
- `tests/CMakeLists.txt`: removed duplicate `ninfer_cli_options_test` target.
- `fork/start-dflash2.ps1`: CUDA graphs on by default, `[switch]$NoCudaGraph`; `--draft-tokens 4`.
- **Prefill attention route.**
  `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`
  (`causal_attention_resolve_route`): the TP2 port's rule for the sharded head count
  (`q_heads == 12`) sent every width > 6 to `ChunkedSmallT`, whose chunk size for 12 heads is 6
  tokens, so a 1,024-token prefill chunk ran as ~171 six-token decode-kernel launches per layer.
  The reference routes `q_heads == 12`, width > 6 to the dense prompt kernel. Now widths up to
  `kMaximumVerifyTokens` (decode/verify, captured in the graphs) keep the small-T kernels and
  larger widths take `Prompt`. `prompt.cu` had the same once-per-process
  `cudaFuncSetAttribute` as `small_t.cu`; replaced with `ensure_func_attr_per_device` so the
  prompt kernels launch on rank 1. Prefill 1.13K -> 1.35K tok/s, TTFT 5.8 -> 4.8-5.0 s at 6.6K
  prompt tokens (reference 1.41K / 4.7 s); outputs unchanged and deterministic.
- **DFlash2 weights on rank 0 only.** The `dflash2/*` objects (~2.07 GiB: 5 draft layers,
  feature projection, candidate-selector codebooks) were `Replicated` on both TP ranks, but the
  draft model only ever runs on rank 0 (`dflash_impl.h`; rank 1 receives drafts by memcpy). That
  cost 2 GiB of VRAM on rank 1 for nothing and, since KV capacity is sized by the rank with the
  least free memory, capped `--max-context` at ~6K on 16 GB cards. New
  `artifact::ShardAxis::PrimaryOnly` (`src/artifact/binder.{h,cpp}`) places a whole object on
  device 0 and nothing on the others; `qwen3_6_27b/impl/load/bindings.cpp` maps `dflash2/*` to it
  (rule must precede the text-family suffix rules, leaf names collide) and `build_device_view`
  only attaches `runtime.dflash` on device 0. Runtime weight total 25.0 -> ~23 GiB.
- **Per-rank VRAM ledger at startup.** `MaterializationStats` / `DeviceMemoryReport` now carry
  `weights_{sharded,replicated,local}_bytes`; `StartupLogRenderer::engine_ready` prints one line
  per rank (`weights X (sharded/replicated/rank-only) | runtime (KV, graphs) | free of total`)
  plus a note on which rank carries rank-only weights, so it is obvious why rank 0 uses more
  memory and which GPU bounds the KV pool. The `server_start` JSONL record gets a `devices` array
  with the same fields.
- `fork/start-dflash2.ps1`: `--kv-capacity` defaults to `-MaxContext` instead of `auto` (auto keeps a
  1 GiB headroom that blocked 16K on 16 GB cards; `-KvCapacity` still overrides).

## Benchmark vs reference (`C:\Work\Programming\ninfer_setup\start-ninfer.ps1`, MTP, graphs on)

Both servers started with the reference launcher's settings (`--tp 2 --devices 0,1 --kv-dtype int8
--max-concurrency 1 --draft-tokens 4 --lm-head-draft`) and benchmarked back to back with
`fork/bench-dflash2.ps1` (same 4 prompts, max 256 tokens, 2 rounds, single client). The reference runs
`--max-context 32768`; the 25.0 GiB DFlash2 artifact (vs 20.9 GiB MTP artifact) does not leave
room for that, so the fork ran `--max-context 8192` (startup fails at 32768: runtime reservation).

| prompt | reference MTP: TTFT / prefill / decode | fork DFlash2: TTFT / prefill / decode |
|---|---|---|
| 0 TP explanation (201-207 tok) | 245-280 ms / 125-143 tok/s / 59.9 tok/s (2.56 tok/round) | 460-480 ms / 74-76 tok/s / 53.1-53.5 tok/s (30-31% acc) |
| 1 Fibonacci code (256 tok) | 246-285 ms / 113-131 tok/s / 103.3-104.3 tok/s (4.40 tok/round) | 472-482 ms / 67-68 tok/s / 103.1-103.6 tok/s (85% acc) |
| 2 NaN causes list (256 tok) | 283 ms / 113 tok/s / 71.6-72.0 tok/s (3.04 tok/round) | 467-478 ms / 67-69 tok/s / 76.9-77.0 tok/s (56% acc) |
| 3 heist summary (256 tok) | 281-292 ms / 110-114 tok/s / 65.0-65.1 tok/s (2.74 tok/round) | 466-467 ms / 69 tok/s / 57.9 tok/s (36% acc) |
| aggregate client tok/s (incl. prefill) | **67.2** | **61.0** |

Both engines are deterministic across rounds (identical SHA per prompt). The fork ties or wins
on decode only when DFlash2 acceptance is high (prompts 1-2); it loses ~190 ms of TTFT on every
request (prefill ~68 vs ~115 tok/s), which dominates at these prompt/output sizes. Per-round cost
is ~40 ms in both engines. `--draft-tokens 3` gives the same aggregate (61.1 tok/s).
`--draft-tokens 7` (upstream's recommended value): short aggregate 63.6 tok/s
(code prompt 147 tok/s at 85% acceptance vs reference 104), still behind the
reference's 67.2 aggregate because of TTFT.

### Long context (~6.6K-token prompt in an 8K window)

`fork/bench-long.ps1`: both servers at `--max-context 8192`, one 26,000-char excerpt of
`eval/corpora/perplexity-1m/data/pg19/00.txt` (6,582 prompt tokens), two tasks, max 512 output
tokens, 2 rounds. Server-side numbers:

| task | reference MTP: TTFT / prefill / decode | fork DFlash2: TTFT / prefill / decode |
|---|---|---|
| 0 summarize (487 / 417 tok) | 4.7-4.9 s / 1343-1397 tok/s / 67.1-68.1 tok/s (2.92 tok/round, 48%) | 5.8 s / 1130-1140 tok/s / 56.0-56.4 tok/s (35% acc) |
| 1 continue story (512 tok) | 4.7 s / 1409-1411 tok/s / 94.8-94.9 tok/s (4.09 tok/round, 78%) | 5.8 s / 1140-1150 tok/s / 74.8-75.4 tok/s (56% acc) |
| aggregate client tok/s (incl. prefill) | **45.1** | **36.0** |

Both deterministic across rounds. At long context the fork does not close the gap: prefill was
~19% slower (1.14K vs 1.40K tok/s, +1.1 s TTFT) and DFlash2 accepts fewer tokens per round
(2.41 / 3.24 tok/round vs MTP 2.92 / 4.09), so decode is 17-21% slower. Per-round cost is
~43 ms in both engines at 6.6K context, unchanged from short context; the difference is purely
acceptance. Not a win for large tasks on this hardware and model pair.

**After the prefill attention-route fix** (same bench, K=7, 5 rounds): prefill 1.31-1.37K tok/s,
TTFT 4.8-5.1 s (was 5.8-5.9 s), decode 53-55 / 69-72 tok/s, aggregate client 37.3 tok/s (was
36.0). Prefill is now within ~3-5% of the reference; the remaining long-context gap is DFlash2
acceptance on prose. Short-prompt TTFT is unchanged (~450 ms vs reference ~280 ms; a fixed
~170 ms per-request cost, not per-token).

### 16K context (~14.6K-token prompt)

Same bench with `-Chars 57000` (14,632 prompt tokens). The fork cannot start with
`--max-context 16384 --kv-capacity auto` (needs 1.16 GB runtime + 1 GiB automatic headroom, only
2.11 GB free after the 25 GiB weights); `--kv-capacity 16384` (explicit, no headroom) works
(`fork/start-dflash2.ps1 -MaxContext 16384 -KvCapacity 16384`, 1.59 GiB free after startup). Reference
ran `start-ninfer.ps1 -MaxContext 16384`.

| task | reference MTP: TTFT / prefill / decode | fork DFlash2: TTFT / prefill / decode |
|---|---|---|
| 0 summarize (439 / 401 tok) | 10.5-10.7 s / 1372-1398 tok/s / 64.1 tok/s (2.81 tok/round, 45%) | 12.8-12.9 s / 1140 tok/s / 53.6 tok/s (2.33 tok/round, 33% acc) |
| 1 continue story (512 tok) | 10.5 s / 1398-1399 tok/s / 57.3-57.4 tok/s (2.50 tok/round, 38%) | 12.8 s / 1140 tok/s / 56.3-56.4 tok/s (2.46 tok/round, 36% acc) |
| aggregate client tok/s (incl. prefill) | **25.7** | **21.6** |

Deterministic across rounds in both. Per-round cost is still ~43.5 ms in both engines at 14.6K
context. Decode is a near tie on the continuation task (MTP acceptance drops to 2.50 tok/round),
16% behind on the summary; prefill remains ~18% slower, which at this prompt size costs 2.3 s of
TTFT per request and dominates the aggregate. DFlash2 does not overtake MTP as context grows.

## Why the fork is slower (root-cause investigation)

All runs at `--max-context 8192`, 6,582-token pg19 prompt, 512 output tokens, back to back.

**1. The PeerMailbox transport is not the cause.** `src/ops/common/allreduce.cu`,
`peer_mailbox.cu` and `wrapper/linear_add.cpp` are byte-identical between fork and reference
(`Compare-Object` diff = 0); both use the pinned-host mailbox only inside captured decode graphs
and host-staged `cudaMemcpyAsync` D2D pulls for eager prefill. Measured with speculation OFF
(`--spec` omitted, plain autoregressive decode) the two engines are identical:

| engine, no speculation | prefill | TTFT | decode |
|---|---|---|---|
| reference | 1411-1422 tok/s | 4.67-4.70 s | 34.6-34.8 tok/s (28.7 ms/token) |
| fork | 1120-1130 tok/s | 5.8-5.9 s | 34.8-34.9 tok/s (28.7 ms/token) |

Same decode cost per forward pass, so the TP2 layer schedule, collectives and CUDA-graph decode
loop are as fast as the reference. Build config is identical too (Release, `sm_120a`, same
nvcc/MSVC flags).

**2. GPU utilization is the same.** `fork/gpu-util.ps1` (nvidia-smi, 100 ms samples, DFlash2 K=7):
fork decode GPU0 90.4% / GPU1 79.4% (min 27/34%), prefill 73-75%; reference decode 90.9% /
91.5% (min 82/85%), prefill 72-74%. Rank 1 idles briefly each round in the fork (rank 0 alone
runs the DFlash2 draft head, rank 1 waits for the drafts), but the per-round wall time is the
same ~43-45 ms as the reference's MTP round, so this is not where the tokens are lost.

**3. Deficit A: prefill was ~20% slower, independent of DFlash2 - FIXED.** 1.12-1.16K vs
1.41K tok/s (+1.1 s TTFT at 6.6K tokens, +2.3 s at 14.6K). `--prefill-chunk 2048` (needs
explicit `--kv-capacity 8192` because of the 1 GiB auto headroom) gave 1.15-1.16K tok/s - no
change, so per-chunk host work was not the cause. Root cause: the TP2 port's attention route rule
for the sharded head count (`q_heads == 12` -> `ChunkedSmallT` for any width > 6) ran prefill
through the 6-token split-K decode kernel instead of the dense prompt kernel (~171 launches per
layer per 1,024-token chunk on 16 full-attention layers). The reference's
`gqa_attention_resolve_route` sends the same case to `Prompt`. Routing widths above
`kMaximumVerifyTokens` to `Prompt` (and setting the prompt kernels' shared-memory attribute per
device) brings prefill to 1.31-1.37K tok/s and TTFT to 4.8-5.0 s. Nsight Systems 2026.3.2 was
not usable (kills the server during warmup), so this was found by reading the route tables.

**4. Deficit B: DFlash2 acceptance on narrative prompts.** On the story-continuation task the
fork accepts 2.4-3.2 tok/round (20-56%) vs MTP's 4.09 tok/round (78%); on code it accepts 85%
and beats the reference (147 vs 104 tok/s). This matches upstream's own
`docs/performance/qwen3.8-27b.md` where Story is DFlash2's weakest category (2.17 tok/round vs
MTP3 2.12) while Code/Structured/Translation reach 4.6-6.5. The model-card averages
(GSM8K 5.46 vs 5.02, HumanEval 4.39 vs 3.91, ...) are on reasoning/code benchmarks, not prose.

**5. Second-order.** Decode host time is ~2.7 ms/round (host_exposed 0.53 s over 188 rounds)
against ~45 ms of device wait per round; not significant.

Conclusion: nothing from the reference's transport work is missing; the gap was (A) the TP2
prefill attention mis-route, now fixed (prefill within ~3-5% of the reference), and (B) DFlash2's
lower acceptance on free-form text with this model, which remains. The fork wins on
code/structured generation and ties on plain decode. Still open: ~170 ms of fixed per-request
TTFT on short prompts (450 vs 280 ms).

## Known issues

1. **Cold-start divergence with penalties.** With the default `presence_penalty` (1.5) and
   `--spec dflash2`, the first two identical-prompt requests after startup produce a slightly
   different (still valid) completion than every later request (pattern `A A' B B B ...`, fully
   repeatable, identical in graph and eager mode). Verified with a token_counts dump that penalty
   bookkeeping is correct (counts reset per request, identical up to the divergence round); the
   flip is a near-tie in the draft/target logits that the penalty creates. With
   `presence_penalty: 0` output is identical from the very first request, and steady state is
   deterministic in all cases.
2. `--spec mtp` crashes at startup in this fork (not needed for the DFlash2 target).
3. Long-prompt prefill is now within ~3-5% of the reference (was ~20% behind; see root-cause
   section). Short-prompt TTFT still carries ~170 ms of fixed per-request overhead (450 vs
   280 ms). Decode is ahead only when acceptance is high (code/structured), behind on prose.
4. `--kv-capacity auto` refuses to start when the runtime reservation plus the 1 GiB automatic
   headroom does not fit (e.g. `--max-context 16384` or `--prefill-chunk 2048`); pass an
   explicit `--kv-capacity N` to skip the headroom.
5. Nsight Systems 2026.3.2 cannot profile this server (dies during warmup under `nsys launch`).

## Housekeeping

- Model artifact stays in `model/` at the repository root (git-ignored via `*.ninfer`); helper
  scripts and docs stay in `fork/`; never delete `ninfer-upstream\.git`.

## Upstream notes

`ninfer-upstream` is a current upstream checkout at `d49296868dcc17bd478ec185f0d3a801bcc0bf56`.
It contains the Qwen3.8-27B DFlash2 artifact reader and supports `--spec dflash2 --draft-tokens 7`.

Run `./fork/build-dflash2.ps1` to provision the MSVC dependency set through vcpkg and build
`ninfer-serve.exe` for CUDA `sm_120a`.

This upstream supports one CUDA device only (no `--tp`/`--devices`) and cannot load the 23.7 GB
DFlash2 artifact on one 16 GB RTX 5060 Ti; the dual-GPU path lives in `ninfer-dflash2-tp2-port`.