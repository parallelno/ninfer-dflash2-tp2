param(
    [string]$modelId = 'qwen-dflash2-vision',
    [int]$Port = 30000,
    # measured: 105,000 fits with -VisionDevice 1 -MaxVisionTokens 1024 (200 MiB free on rank 0);
    # the old dual-replicated 16384-token layout capped out at ~51K
    [int]$MaxContext = 110000,
    [int]$DraftTokens = 4,
    [int]$Timeout = 600000,
    [int]$KvCapacity = 0,
    [int[]]$Devices = @(0, 1),
    # CUDA device id that holds the vision tower (~282 MiB weights + encode workspace). Rank 0
    # (device 0) already carries the ~2 GiB rank-only DFlash2 draft, so rank 1 has the headroom.
    [int]$VisionDevice = 1,
    # Per-image cap in merged vision tokens; one token covers 32x32 px, so
    #   256 -> ~512x512, 1024 -> ~1024x1024, 2048 -> ~1448x1448, 16384 -> ~4096x4096 (ceiling).
    # Larger images are downscaled to fit; the encode workspace scales with this value.
    [int]$MaxVisionTokens = 1024,
    # Prefill chunk width in tokens (multiple of 128). 0 = engine default (1024). 2048/4096 give
    # +3-4% marginal prefill but cost ~0.35/0.7 GiB of activation workspace on each rank, which
    # lowers the reachable -MaxContext (4096 needs -MaxContext <= ~65536 on 16 GB cards).
    [int]$PrefillChunk = 0,
    # Escape hatch: restore the upstream context-cache defaults (private 2, shared 4, anchors 2).
    # Measured on this hardware those defaults never produce a prefix-cache hit (the pressure
    # planner evicts the only retained checkpoint before the next turn) while still paying
    # ~0.65 s of checkpoint-capture work per request. See fork/research/prefill/README.md.
    [switch]$DefaultContextCache,
    [switch]$NoCudaGraph
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$server = Join-Path $repo 'build-tp2-check\apps\ninfer-serve.exe'
$model = Join-Path $repo 'model\qwen3_8_27b_nvfp4.dflash2.ninfer'

if ($Devices.Count -ne 2) {
    throw '-Devices must contain exactly two CUDA device ids.'
}

if ($Devices -notcontains $VisionDevice) {
    throw '-VisionDevice must be one of -Devices.'
}

if ($PrefillChunk -lt 0 -or ($PrefillChunk % 128) -ne 0) {
    throw '-PrefillChunk must be 0 (engine default) or a positive multiple of 128.'
}

if (-not (Test-Path -LiteralPath $server -PathType Leaf)) {
    throw "NInfer server was not found: $server. Build build-tp2-check in the repository root first."
}

if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
    throw "DFlash2 artifact was not found: $model"
}

$arguments = @(
    $model,
    '--host', '127.0.0.1',
    '--port', $Port,
    '--model-id', $modelId,
    '--tp', '2',
    '--devices', ($Devices -join ','),
    '--max-context', $MaxContext,
    # explicit capacity skips the 1 GiB headroom that 'auto' enforces (blocks 16K on 16 GB cards)
    '--kv-capacity', $(if ($KvCapacity -gt 0) { $KvCapacity } else { $MaxContext }),
    '--kv-dtype', 'int8',
    '--max-concurrency', '1',
    '--spec', 'dflash2',
    '--draft-tokens', $DraftTokens,
    '--lm-head-draft',
    '--pending-timeout-ms', $Timeout,
    '--vision',
    '--vision-device', $VisionDevice,
    '--max-vision-tokens', $MaxVisionTokens
)

if (-not $DefaultContextCache) {
    # Prefill speed-up for the single-user / agent workload (fork/research/prefill/README.md §3):
    # one private continuation, no shared-prefix catalog, no long anchors. With these the
    # turn-closure checkpoint of the previous request survives and every follow-up turn reuses
    # the whole conversation prefix (95%+ hit): a 6.5K-token agent turn measured 5.3 s -> 1.0 s
    # wall (TTFT 0.62 s), an identical repeated prompt 3.9 s -> 0.5-0.8 s. Cache misses are also
    # ~0.15 s cheaper because the shared-prefix capture chunks are gone. No extra VRAM; works at
    # the full 110K context. Note: a reuse hit is not bit-identical to a cold recompute (the known
    # TP2 last-bit residual drift); greedy continuations may diverge in wording after ~20 tokens.
    $arguments += @(
        '--max-private-continuations', '1',
        '--max-shared-prefixes', '0',
        '--max-long-anchors-per-continuation', '0'
    )
}

if ($PrefillChunk -gt 0) {
    $arguments += @('--prefill-chunk', $PrefillChunk)
}

if ($NoCudaGraph) {
    $arguments += '--no-cuda-graph'
}

& $server @arguments
