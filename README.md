# NInfer — DFlash2 on Windows 11 TP2 fork

> Qwen3.8-27B NVFP4 with DFlash2 speculative decoding, tensor-parallel across two consumer
> GeForce GPUs on native Windows 11 — measured on 2 × RTX 5060 Ti 16 GB, no CUDA P2P, no NCCL,
> no WSL2 or Docker.

> **This is the DFlash2-TP2 fork** ([parallelno/ninfer-dflash2-tp2](https://github.com/parallelno/ninfer-dflash2-tp2)).
> It descends from [ivanov84/ninfer-windows-tp2](https://github.com/ivanov84/ninfer-windows-tp2)
> (native Windows TP2 with the pinned-host peer-mailbox transport), which in turn builds on
> [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m),
> [natpate/ninfer-windows](https://github.com/natpate/ninfer-windows), and upstream
> [Neroued/ninfer](https://github.com/Neroued/ninfer). See [NOTICE](NOTICE) for attribution.
>
> What this fork adds:
>
> - **`--spec dflash2` under `--tp 2`.** The DFlash2 draft model runs on rank 0 only
>   (`artifact::ShardAxis::PrimaryOnly`); rank 1 receives drafts by memcpy, saving ~2 GiB on the
>   second GPU. CUDA graphs are on by default; `--no-cuda-graph` is the eager escape hatch.
> - **TP2 correctness fixes** that also affect MTP: per-plane shard copies in
>   `src/artifact/materializer.cpp` (the merged upstream loop gave every shard the first N bytes
>   of the full tensor → NaN), mirrored rank-1 KV page tables / StateImage slots (deterministic
>   output at tp2), host state/KV tiers default to 0 at `--tp 2`, and per-device
>   `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` in the attention kernels (was set once
>   per process → `cudaErrorInvalidValue` on rank 1 for prompts longer than ~2K tokens).
> - **Prefill routing fix**: 12-head prompts no longer run as 6-token decode launches
>   (`causal_softmax_attention.cpp`), prefill 1.13K → 1.35K tok/s on the test hardware.
> - **Per-rank startup memory report**: weights sharded/replicated/rank-only, KV, graphs, free;
>   also emitted as a `devices` array in the `server_start` JSONL record.
> - **Windows helper scripts and status docs in [`fork/`](fork/)**: `build-dflash2.ps1`,
>   `start-dflash2.ps1`, `test-dflash2.ps1`, `bench-dflash2.ps1`, `bench-long.ps1`,
>   `gpu-util.ps1`, plus [fork/STATUS.md](fork/STATUS.md) (fixes, known issues, measurements) and
>   [fork/BENCHMARKS.md](fork/BENCHMARKS.md) (agent-workflow comparison against the MTP reference).
>
> Everything below this fork section is the upstream README; the single-RTX-5090 / Linux
> statements describe upstream's product boundary, not this fork's tested configuration.

## Fork quick start (Windows 11, two GPUs)

Requirements: Windows 11 x64, Visual Studio 2022 (Desktop C++), CMake 3.28+, Ninja, CUDA
Toolkit 13.4 (`sm_120a`), two Blackwell GeForce GPUs (tested: 2 × RTX 5060 Ti 16 GB; the Windows
display GPU has ~1.2 GB less free memory and bounds KV capacity), and vcpkg for FFmpeg/libcurl.

Build (the CUDA compiler needs `cl.exe` on `PATH`; `VsDevCmd.bat` provides it):

```powershell
cmd /c """C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"" -arch=amd64 -host_arch=amd64 -no_logo && cmake -S . -B build-tp2-check -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && cmake --build build-tp2-check --target ninfer-serve --parallel 4"
```

`fork/build-dflash2.ps1` is the one-shot alternative that also bootstraps vcpkg and its
dependency set. Place the Qwen3.8-27B NVFP4 artifact with DFlash2 companion weights (see the
[model card](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md)) at
`model\qwen3_8_27b_nvfp4.dflash2.ninfer` (git-ignored), then:

```powershell
.\fork\start-dflash2.ps1                     # tp2, dflash2, K=4, graphs on, port 30000
.\fork\start-dflash2.ps1 -DraftTokens 7      # K=7 (higher acceptance on code/structured output)
.\fork\start-dflash2.ps1 -NoCudaGraph        # eager fallback
.\fork\test-dflash2.ps1 -Port 30000          # /health + deterministic chat smoke test
```

The launcher expands to:

```powershell
build-tp2-check\apps\ninfer-serve.exe model\qwen3_8_27b_nvfp4.dflash2.ninfer `
  --host 127.0.0.1 --port 30000 --model-id qwen-local `
  --tp 2 --devices 0,1 --max-context 32768 --kv-capacity 32768 --kv-dtype int8 `
  --max-concurrency 1 --spec dflash2 --draft-tokens 4 --lm-head-draft
```

`--kv-capacity` is passed explicitly because `auto` keeps 1 GiB of headroom that does not fit
alongside the 25 GiB DFlash2 artifact on two 16 GB cards. Startup takes ~65-80 s (weights
~55-73 s, graph capture ~4.5 s). Measured against the MTP reference fork on the same hardware
(`fork/bench-dflash2.ps1`, 4 prompts x 256 tokens): the reference averages ~67 tok/s decode; this
fork ranges 53-113 tok/s depending on DFlash2 draft acceptance (31%-96%) — ahead on code and
structured output, behind on free-form prose. Prefill is ~4% slower and short-prompt TTFT ~170 ms
higher. Details and the long-context run are in [fork/STATUS.md](fork/STATUS.md).

---

# NInfer (upstream README)

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen checkpoints on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

NInfer supports five artifact identities. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

The artifact identity fixes the exact model and weight profile. Every artifact also embeds the
tokenizer, chat template, and media frontend resources required by its registered target.

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, CUDA Toolkit 13.1 or newer, CMake 3.28 or
newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, and `libswscale >= 7`), and
`libcurl >= 7.85`. The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests, benchmarks, and maintainer tools are excluded from the default build. There is no install
target or packaged binary distribution; run NInfer from its source build tree.

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% | 2.15× |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

All registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
