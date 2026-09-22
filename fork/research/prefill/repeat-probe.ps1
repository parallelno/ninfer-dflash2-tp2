param([int]$Port = 30006, [int]$DocChars = 16000, [int]$Offset = 10000, [int]$MaxTokens = 24)
# Sends one prompt, then the identical prompt, then the conversation extended by one turn.
$ErrorActionPreference = 'Stop'
$doc = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot '..\..\..\eval\corpora\perplexity-1m\data\pg19\00.txt')
function Send($messages, [string]$label) {
    $body = @{ model = 'qwen-bench'; messages = @($messages); max_tokens = $MaxTokens; temperature = 0; reasoning_effort = 'none'; presence_penalty = 0 } | ConvertTo-Json -Depth 6
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
    $sw.Stop()
    Write-Host ("{0,-36} prompt={1,5} wall={2,6} ms" -f $label, $resp.usage.prompt_tokens, [math]::Round($sw.Elapsed.TotalMilliseconds))
    $c = [string]$resp.choices[0].message.content; if ([string]::IsNullOrWhiteSpace($c)) { $c = 'ok' }; return $c
}
$u = "Document:`n<document>`n$($doc.Substring($Offset, $DocChars))`n</document>`nSummarize the first paragraph in one sentence."
$a = Send @(@{ role = 'user'; content = $u }) 'first'
$null = Send @(@{ role = 'user'; content = $u }) 'identical repeat'
$b = Send @(@{ role = 'user'; content = $u }, @{ role = 'assistant'; content = $a }, @{ role = 'user'; content = 'Now the second paragraph.' }) 'extend +1 turn'
$null = Send @(@{ role = 'user'; content = $u }, @{ role = 'assistant'; content = $a }, @{ role = 'user'; content = 'Now the second paragraph.' }, @{ role = 'assistant'; content = $b }, @{ role = 'user'; content = 'And the third.' }) 'extend +2 turns'

