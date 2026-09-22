param([int]$Port = 30006, [int]$Turns = 3, [int]$DocChars = 16000)
# Two independent multi-turn conversations, A then B, each extended turn by turn. Tests whether a
# second conversation can still get prefix-cache hits after the first one has been served.
$ErrorActionPreference = 'Stop'
$doc = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot '..\..\..\eval\corpora\perplexity-1m\data\pg19\00.txt')
function Send($messages, [string]$label) {
    $body = @{ model = 'qwen-bench'; messages = @($messages); max_tokens = 24; temperature = 0; reasoning_effort = 'none'; presence_penalty = 0 } | ConvertTo-Json -Depth 6
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
    $sw.Stop()
    Write-Host ("{0,-12} prompt={1,5} wall={2,6} ms" -f $label, $resp.usage.prompt_tokens, [math]::Round($sw.Elapsed.TotalMilliseconds))
    $c = [string]$resp.choices[0].message.content; if ([string]::IsNullOrWhiteSpace($c)) { $c = 'ok' }; return $c
}
foreach ($conv in @(@{ name = 'A'; off = 30000 }, @{ name = 'B'; off = 130000 }, @{ name = 'A2'; off = 30000 })) {
    $m = @(@{ role = 'user'; content = "Document:`n<document>`n$($doc.Substring($conv.off, $DocChars))`n</document>`nSummarize the first paragraph in one sentence." })
    for ($t = 0; $t -lt $Turns; ++$t) {
        $a = Send $m "$($conv.name) turn $t"
        $m += @{ role = 'assistant'; content = $a }
        $m += @{ role = 'user'; content = "Step $t done. Next paragraph, one sentence." }
    }
}

