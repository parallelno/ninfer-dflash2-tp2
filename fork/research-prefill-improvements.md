All tests are complete, servers stopped, GPUs clean. Here is the full report.

---

# Prefill measurement & stage decomposition (TP2, 2× RTX 5060 Ti, Windows 11 WDDM, no P2P)

## How I measured (all reproducible from `temp\`)

- **Transport microbench** `temp\prefill_transport_bench.cu` → `.exe` — links the *production* engine libs (`ninfer_ops` + `ninfer_core` + `ninfer_nvfp4_non_rdc`), so every number is the shipped code, not a lookalike. Measures, at 128/256/512/1024/2048-token chunk payloads: the production staged `allreduce_sum` (isolated + 128-chain), the **real PeerMailbox** (via actual `PeerMailbox` + CUDA-graph capture + engine flag-reset protocol — the same selection path decode uses), an explicit pinned CE relay, bare staged D2D copies per direction, and per-device pinned D2H/H2D.
- **Server sweeps** with your exact startup recipe (`fork\start-dflash2.ps1` args) + `--request-log-jsonl`: prompt lengths 144→17,071 tokens; three server sessions: (A) defaults (chunk 1024, dflash2), (B) `--prefill-chunk 2048`, (C) no `--spec` (isolates DFlash2's per-chunk feature extraction). Per-request `timings_seconds` come from the engine's own JSONL records.
- **GPU sampling** (nvidia-smi 100 ms) during a 16K prefill.

## 1. Your point about the links — measured ground truth

Your intuition was reasonable but the hardware negotiates differently than the docs claim: **both GPUs run PCIe 3.0 ×8** (`nvidia-smi` reports `gen.max=3, width.max=8` for *both* cards; under load both showed gen3x8; and measured pinned bandwidth is symmetric: dev0 D2H 7.03 / H2D 6.76 GB/s, dev1 6.97 / 6.71 GB/s). And the model-split doesn't bypass the slow side: **every allreduce's every byte crosses both links once** (rank r's partial goes GPU r → host, host → GPU 1−r), so the exchange is bounded by *each* link carrying 2×payload per collective. There is no "fast half" — both links carry the full 20 MiB per 1024-token-chunk collective.

## 2. Prefill speed (measured, default config = how you run it)

| Prompt tokens | TTFT (server) | Prefill | Marginal rate |
|---|---|---|---|
| 144 | 0.51 s | 0.51 s | — |
| 1,088 | 1.14 s | 1.14 s | — |
| 8,398 | 6.02 s | 6.01 s | — |
| 17,071 | 12.21 s | 12.19 s | — |

**Linear fit: 0.691 ms/token (1,446 tok/s) + 0.41 s fixed per request.** Chunk 2048: 1,502 tok/s (+4%); without DFlash2 features: identical (12.16 vs 12.19 s — feature extraction ≈ 0.3%). So neither chunk size nor DFlash is a lever.

## 3. Stage decomposition (17K prompt, TTFT 12.3 s)

| Stage | Time | Share | Evidence |
|---|---|---|---|
| Queue + prepare (tokenize/plan) | ~0.03 s | 0.2% | `prepare/queue` fields ≈ 0.01–0.02 s |
| Fixed per-request overhead (first-chunk effects, KV reset, final logits + sampling) | ~0.41 s | 3.3% | regression intercept; a 144-tok chunk's collectives are only ~0.11 s yet prefill = 0.51 s |
| **TP2 collectives** (128 staged allreduces per 1024-tok chunk) | **~7.9 s** | **~65%** | transport bench: 128 × 3.64 ms = 466 ms/chunk × 17 chunks |
| Local compute (embedding, 64 layers of half-width NVFP4 GEMMs, attention/GDN kernels, KV/GDN appends, DFlash features) | ~4.0 s | ~32% | remainder; GPU util 82–83% at only 74–78 W (long low-power CE-wait phases corroborate) |

Per 1024-token chunk: **708 ms total = 466 ms collectives (66%) + 242 ms everything else (34%)**.

## 4. Can the PeerMailbox make prefill faster? — **No. Measured: 2.3× SLOWER**

| Chunk payload | Staged (today) | PeerMailbox | Pinned CE relay |
|---|---|---|---|
| 1.25 MiB (128 tok) | 857 µs/call | 1,130 µs (−32%) | **565 µs (+34%)** |
| 2.5 MiB | 1,197 µs | 2,061 µs (−72%) | 967 µs (+19%) |
| 10 MiB (1024 tok) | 3,644 µs | **8,272 µs (−127%)** | 3,183 µs (+13%) |
| 20 MiB (2048 tok) | 7,628 µs | 16,424 µs (−115%) | 6,437 µs (+16%) |

The mailbox kernel's SM zero-copy reads over PCIe lose badly to the copy engine at MiB scale — exactly why the fork's designers gated it to captured 10–40 KiB payloads. The fork's selectivity is empirically vindicated on your hardware.

## 5. Real improvement levers (all bit-exact — none degrade model quality)

1. **Pinned relay transport** (drop-in change to `allreduce_sum`'s pull: D2H into pinned + H2D, same events/bytes/arithmetic): collectives 3.64→3.18 ms at 1024-chunk → **~+8% prefill** (more at short prompts: +34% at 128-tok tails — relevant to your 450 ms short-prompt TTFT).
2. **Full-duplex pipelined relay**: measured single-direction staged copy = 1.75 ms while the pair takes 3.64 ms — the two directions aren't overlapping. A double-buffered relay that pipelines collective i+1's publish under collective i's fetch could approach **~1.75–2.9 ms/call → +20–50% of collective time → up to ~+25% prefill**. Moderate complexity.
3. **Sequence parallelism** (reduce-scatter + allgather along tokens): halves bytes — collectives → ~½, potential **~+40–50% prefill** on long prompts. Bit-exact for the sums, but a deep schedule change.
4. **Not worth it**: chunk-size tuning (+3–4%), DFlash-off (0.3%), mailbox (negative), quantized exchange (excluded per your quality requirement).

**Bottom line:** prefill today runs at 1,446 tok/s and is **65% TP2-collective-bound** on symmetric PCIe 3.0 ×8 links; the mailbox transport would make it worse, but a quality-neutral transport relay + overlap work offers a realistic ~10–25%, and sequence parallelism the big ~1.5× if you ever want it. All data and tools are in `temp\` (`bench_run2.log`, `reqlog-{a,b,c}.jsonl`, `prefill_sweep.csv`, the bench `.cu/.exe`, and the three `.ps1` scripts) — rerun with `temp\prefill_transport_bench.exe 0 1` and the sweep scripts against a server started with `temp\start-bench-server.ps1`.