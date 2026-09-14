param(
    [int]$Port = 30001,
    [string]$ModelId = 'qwen-local-dflash2',
    [int]$MaxTokens = 512,
    [int]$Chars = 26000,
    [int]$SampleMs = 100,
    [string]$Corpus = (Join-Path (Split-Path -Parent $PSScriptRoot) 'eval\corpora\perplexity-1m\data\pg19\00.txt')
)

$ErrorActionPreference = 'Stop'

$doc = (Get-Content -Raw -Encoding UTF8 $Corpus).Substring(0, $Chars)
$prompt = "Below is an excerpt from a book. Continue writing the story from exactly where it stops, keeping the same style, tone and characters. Write at least 400 words.`n`n<document>`n$doc`n</document>"

$body = @{
    model            = $ModelId
    messages         = @(@{ role = 'user'; content = $prompt })
    max_tokens       = $MaxTokens
    temperature      = 0
    reasoning_effort = 'none'
} | ConvertTo-Json -Depth 5

# Sample both GPUs while the request runs; CSV rows: timestamp, index, util.gpu, util.mem, power, clocks.sm
$csv = Join-Path (Split-Path -Parent $PSScriptRoot) 'temp\gpu_util.csv'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $csv) | Out-Null
if (Test-Path $csv) { Remove-Item $csv }
$sampler = Start-Process -FilePath nvidia-smi -ArgumentList @(
    '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm,pcie.link.gen.current',
    '--format=csv,noheader,nounits', "-lms", $SampleMs) -RedirectStandardOutput $csv -NoNewWindow -PassThru

Start-Sleep -Milliseconds 500
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body
$sw.Stop()
Start-Sleep -Milliseconds 300
Stop-Process -Id $sampler.Id -Force

$rows = Import-Csv $csv -Header ts, idx, util, memutil, power, sm, gen | ForEach-Object {
    [pscustomobject]@{
        t     = [datetime]::ParseExact($_.ts.Trim(), 'yyyy/MM/dd HH:mm:ss.fff', $null)
        idx   = [int]$_.idx
        util  = [int]$_.util
        mem   = [int]$_.memutil
        power = [double]$_.power
        sm    = [int]$_.sm
    }
}
$t0 = ($rows | Measure-Object t -Minimum).Minimum
# Split into prefill (first ~TTFT) and decode phases using the server-reported TTFT if given, else 25%/75% by time.
$total = ($rows | Measure-Object t -Maximum).Maximum - $t0
$ttft = if ($env:TTFT_MS) { [timespan]::FromMilliseconds([int]$env:TTFT_MS) } else { [timespan]::FromTicks([long]($total.Ticks * 0.3)) }

"prompt_tok=$($resp.usage.prompt_tokens) output_tok=$($resp.usage.completion_tokens) wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s samples=$($rows.Count) split_at=$([math]::Round($ttft.TotalSeconds,1))s"
foreach ($g in 0, 1) {
    $pre = $rows | Where-Object { $_.idx -eq $g -and ($_.t - $t0) -lt $ttft }
    $dec = $rows | Where-Object { $_.idx -eq $g -and ($_.t - $t0) -ge $ttft }
    $fmt = { param($s) if ($s) { "util {0,5:N1}% (min {1,3} max {2,3})  mem {3,5:N1}%  power {4,6:N1}W  sm {5,5:N0}MHz" -f ($s | Measure-Object util -Average).Average, ($s | Measure-Object util -Minimum).Minimum, ($s | Measure-Object util -Maximum).Maximum, ($s | Measure-Object mem -Average).Average, ($s | Measure-Object power -Average).Average, ($s | Measure-Object sm -Average).Average } else { 'n/a' } }
    "GPU$g prefill: $(& $fmt $pre)"
    "GPU$g decode : $(& $fmt $dec)"
}
