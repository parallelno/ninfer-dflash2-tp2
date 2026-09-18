param(
    [string]$modelId = 'qwen-dflash2',
    [int]$Port = 30000,
    # preflight reports ~110.7K as the ceiling on rank 0 (16 GB) with the 24 MiB/class tp2 graph allowance
    [int]$MaxContext = 7000,
    [int]$DraftTokens = 4,
    [int]$Timeout = 600000,
    [int]$KvCapacity = 0,
    [int[]]$Devices = @(0, 1),
    [switch]$NoCudaGraph
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$server = Join-Path $repo 'build-tp2-check\apps\ninfer-serve.exe'
$model = Join-Path $repo 'model\qwen3_8_27b_nvfp4.dflash2.ninfer'

if ($Devices.Count -ne 2) {
    throw '-Devices must contain exactly two CUDA device ids.'
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
    '--vision'
)

if ($NoCudaGraph) {
    $arguments += '--no-cuda-graph'
}

& $server @arguments