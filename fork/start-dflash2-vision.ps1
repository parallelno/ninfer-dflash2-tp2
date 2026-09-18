param(
    [string]$modelId = 'qwen-dflash2-vision',
    [int]$Port = 30000,
    # measured: 105,000 fits with -VisionDevice 1 -MaxVisionTokens 1024 (200 MiB free on rank 0);
    # the old dual-replicated 16384-token layout capped out at ~51K
    [int]$MaxContext = 110080,
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

if ($NoCudaGraph) {
    $arguments += '--no-cuda-graph'
}

& $server @arguments