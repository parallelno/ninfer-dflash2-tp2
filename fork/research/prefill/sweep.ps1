param(
    [int]$Port = 30006,
    [int[]]$TargetTokens = @(64, 512, 1024, 4096, 16384),
    [int]$MaxTokens = 1,
    [int]$Repeats = 1,
    [string]$Tag = 'x',
    [switch]$NoWarmup,
    [switch]$Distinct   # use different doc offsets per request so prefix reuse cannot trigger
)
$ErrorActionPreference = 'Stop'
$out = (Join-Path $PSScriptRoot 'out')
$corpus = (Join-Path $PSScriptRoot '..\..\..\eval\corpora\perplexity-1m\data\pg19\00.txt')
$doc = Get-Content -Raw -Encoding UTF8 $corpus
$instruction = "Below is an excerpt from a book. Continue writing the story from exactly where it stops, keeping the same style, tone and characters. Write at least 400 words."
$instrTokens = 38

function Send-Prompt([string]$slice) {
    $body = @{ model = 'qwen-bench'; messages = @(@{ role = 'user'; content = "$instruction`n`n<document>`n$slice`n</document>" })
               max_tokens = $MaxTokens; temperature = 0; reasoning_effort = 'none'; presence_penalty = 0 } | ConvertTo-Json -Depth 5
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
    $sw.Stop()
    [pscustomobject]@{ prompt_tokens = [int]$resp.usage.prompt_tokens; wall_ms = [math]::Round($sw.Elapsed.TotalMilliseconds) }
}

if (-not $NoWarmup) { $w = Send-Prompt ($doc.Substring(0, 3000)); "warmup prompt=$($w.prompt_tokens) wall=$($w.wall_ms)ms" }

$offset = 0
$rows = @()
foreach ($t in $TargetTokens) {
    for ($r = 0; $r -lt $Repeats; ++$r) {
        $chars = [math]::Max(120, [int](($t - $instrTokens) * 4.05))
        if ($Distinct) { $offset = ($offset + 50000) % ($doc.Length - $chars - 1) }
        $res = Send-Prompt ($doc.Substring($offset, $chars))
        $rows += [pscustomobject]@{ target = $t; prompt_tok = $res.prompt_tokens; wall_ms = $res.wall_ms }
        "target=$t prompt_tok=$($res.prompt_tokens) wall=$($res.wall_ms)ms"
        Start-Sleep -Milliseconds 500
    }
}
$rows | Export-Csv -NoTypeInformation (Join-Path $out "sweep-$Tag.csv")

