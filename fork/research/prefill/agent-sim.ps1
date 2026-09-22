param([int]$Port = 30006, [string]$Tag = 'x', [int]$Turns = 5, [int]$DocChars = 24000, [int]$ToolChars = 1200, [int]$MaxTokens = 48)
# Simulates an agent loop: long first prompt, then each turn appends the assistant reply plus a
# ~300-token "tool result" user message. Reports per-turn prompt tokens, computed tokens and TTFT.
$ErrorActionPreference = 'Stop'
$corpus = (Join-Path $PSScriptRoot '..\..\..\eval\corpora\perplexity-1m\data\pg19\00.txt')
$doc = Get-Content -Raw -Encoding UTF8 $corpus
$messages = @(@{ role = 'user'; content = "You are an assistant working through a document step by step. Document:`n<document>`n$($doc.Substring(100000, $DocChars))`n</document>`nSummarize the first paragraph in one sentence." })
$rows = @()
for ($t = 0; $t -lt $Turns; ++$t) {
    $body = @{ model = 'qwen-bench'; messages = $messages; max_tokens = $MaxTokens; temperature = 0; reasoning_effort = 'none'; presence_penalty = 0 } | ConvertTo-Json -Depth 6
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
    $sw.Stop()
    $reply = [string]$resp.choices[0].message.content
    $rows += [pscustomobject]@{ turn = $t; prompt_tok = $resp.usage.prompt_tokens; out = $resp.usage.completion_tokens; wall_ms = [math]::Round($sw.Elapsed.TotalMilliseconds) }
    $messages += @{ role = 'assistant'; content = $reply }
    $messages += @{ role = 'user'; content = "Tool result:`n$($doc.Substring(150000 + $t * 5000, $ToolChars))`nContinue with the next step in one sentence." }
}
$rows | Format-Table -AutoSize
$rows | Export-Csv -NoTypeInformation (Join-Path $PSScriptRoot "out\agent-$Tag.csv")


