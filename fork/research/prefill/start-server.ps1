param(
    [string]$Tag = 'x',
    [int]$Port = 30006,
    [int]$PrefillChunk = 0,
    [int]$MaxContext = 32768,
    [string]$Spec = 'dflash2',
    [switch]$NoPrefixReuse,
    [string[]]$Extra = @(),
    [switch]$Nsys,
    [int]$NsysDelaySec = 90,
    [int]$NsysDurationSec = 60,
    [hashtable]$Env = @{}
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$out  = (Join-Path $PSScriptRoot 'out')
$server = Join-Path $repo 'build-tp2-check\apps\ninfer-serve.exe'
$model  = Join-Path $repo 'model\qwen3_8_27b_nvfp4.dflash2.ninfer'
New-Item -ItemType Directory -Force -Path $out | Out-Null

$reqlog = Join-Path $out "reqlog-$Tag.jsonl"
foreach ($f in @($reqlog, (Join-Path $out "serve-$Tag.log"), (Join-Path $out "serve-$Tag.err.log"))) {
    if (Test-Path $f) { Remove-Item $f }
}

$arguments = @(
    $model, '--host', '127.0.0.1', '--port', "$Port", '--model-id', 'qwen-bench',
    '--tp', '2', '--devices', '0,1',
    '--max-context', "$MaxContext", '--kv-capacity', "$MaxContext", '--kv-dtype', 'int8',
    '--max-concurrency', '1', '--pending-timeout-ms', '600000',
    '--request-log-jsonl', $reqlog
)
if ($Spec -ne 'none') { $arguments += @('--spec', $Spec, '--draft-tokens', '4', '--lm-head-draft') }
if ($PrefillChunk -gt 0) { $arguments += @('--prefill-chunk', "$PrefillChunk") }
if ($NoPrefixReuse) { $arguments += '--no-prefix-reuse' }
if ($Extra.Count -gt 0) { $arguments += $Extra }

foreach ($k in $Env.Keys) { Set-Item -Path "Env:$k" -Value $Env[$k] }

if ($Nsys) {
    $nsysExe = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe'
    $rep = Join-Path $out "prof-$Tag.nsys-rep"
    if (Test-Path $rep) { Remove-Item $rep }
    $nsysArgs = @('profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none',
                  "--delay=$NsysDelaySec", "--duration=$NsysDurationSec", '--force-overwrite=true', "--output=$rep", $server) + $arguments
    $proc = Start-Process -FilePath $nsysExe -ArgumentList $nsysArgs `
        -RedirectStandardOutput (Join-Path $out "serve-$Tag.log") -RedirectStandardError (Join-Path $out "serve-$Tag.err.log") `
        -WindowStyle Hidden -PassThru
} else {
    $proc = Start-Process -FilePath $server -ArgumentList $arguments `
        -RedirectStandardOutput (Join-Path $out "serve-$Tag.log") -RedirectStandardError (Join-Path $out "serve-$Tag.err.log") `
        -WindowStyle Hidden -PassThru
}
$proc.Id | Out-File (Join-Path $out "serve-$Tag.pid") -Encoding ascii
foreach ($k in $Env.Keys) { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }

$deadline = (Get-Date).AddSeconds(300)
$ready = $false
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) { break }
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        if ($h.status -eq 'ok') { $ready = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) {
    $tail = Get-Content (Join-Path $out "serve-$Tag.err.log") -Tail 5 -ErrorAction SilentlyContinue
    throw "server not ready (pid $($proc.Id), exited=$($proc.HasExited)): $tail"
}
"ready tag=$Tag port=$Port pid=$($proc.Id) chunk=$PrefillChunk spec=$Spec noreuse=$NoPrefixReuse env=$($Env | ConvertTo-Json -Compress)"

