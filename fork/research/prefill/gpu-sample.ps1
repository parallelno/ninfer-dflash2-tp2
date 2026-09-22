param([int]$Port = 30006, [int]$Tokens = 8192, [string]$Tag = 'x', [int]$SampleMs = 50)
# Samples nvidia-smi while one prefill-only request runs; prints mean util / power per GPU over the request window.
$ErrorActionPreference = 'Stop'
$out = (Join-Path $PSScriptRoot 'out')
$corpus = (Join-Path $PSScriptRoot '..\..\..\eval\corpora\perplexity-1m\data\pg19\00.txt')
$doc = Get-Content -Raw -Encoding UTF8 $corpus
$slice = $doc.Substring(120000, [int](($Tokens - 38) * 4.05))
$body = @{ model = 'qwen-bench'; messages = @(@{ role = 'user'; content = "Continue the story.`n`n<document>`n$slice`n</document>" })
           max_tokens = 1; temperature = 0; reasoning_effort = 'none' } | ConvertTo-Json -Depth 5
$csv = Join-Path $out "gpu-$Tag.csv"
if (Test-Path $csv) { Remove-Item $csv }
$sampler = Start-Process -FilePath nvidia-smi -ArgumentList @('--query-gpu=timestamp,index,utilization.gpu,power.draw,clocks.sm', '--format=csv,noheader,nounits', '-lms', "$SampleMs") -RedirectStandardOutput $csv -NoNewWindow -PassThru
Start-Sleep -Milliseconds 600
$t0 = Get-Date
$resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
$t1 = Get-Date
Start-Sleep -Milliseconds 300
Stop-Process -Id $sampler.Id -Force
$rows = Import-Csv $csv -Header ts, idx, util, power, sm | ForEach-Object {
    [pscustomobject]@{ t = [datetime]::ParseExact($_.ts.Trim(), 'yyyy/MM/dd HH:mm:ss.fff', $null); idx = [int]$_.idx; util = [int]$_.util; power = [double]$_.power; sm = [int]$_.sm }
}
# trim 300 ms at each edge of the request window to exclude ramp
$lo = $t0.AddMilliseconds(300); $hi = $t1.AddMilliseconds(-300)
"prompt_tok=$($resp.usage.prompt_tokens) wall=$([math]::Round(($t1 - $t0).TotalSeconds, 2))s"
foreach ($g in 0, 1) {
    $s = $rows | Where-Object { $_.idx -eq $g -and $_.t -ge $lo -and $_.t -le $hi }
    "GPU$g  n=$($s.Count)  util {0,5:N1}% (min {1} max {2})  power {3,6:N1} W  sm {4,5:N0} MHz" -f ($s | Measure-Object util -Average).Average, ($s | Measure-Object util -Minimum).Minimum, ($s | Measure-Object util -Maximum).Maximum, ($s | Measure-Object power -Average).Average, ($s | Measure-Object sm -Average).Average
}

