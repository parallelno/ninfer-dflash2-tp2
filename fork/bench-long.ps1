param(
    [int]$Port = 30001,
    [string]$ModelId = 'qwen-local-dflash2',
    [int]$MaxTokens = 512,
    [int]$Repeats = 2,
    [int]$Chars = 26000,
    [string]$Corpus = 'ninfer-dflash2-tp2-port\eval\corpora\perplexity-1m\data\pg19\00.txt'
)

$ErrorActionPreference = 'Stop'

$doc = (Get-Content -Raw -Encoding UTF8 $Corpus).Substring(0, $Chars)

# Task 0: summarize (low draft acceptance expected). Task 1: continue text (high acceptance expected).
$prompts = @(
    "Below is an excerpt from a book. Read it carefully, then write a detailed summary of the events, characters and themes in about 300 words.`n`n<document>`n$doc`n</document>",
    "Below is an excerpt from a book. Continue writing the story from exactly where it stops, keeping the same style, tone and characters. Write at least 400 words.`n`n<document>`n$doc`n</document>"
)

$health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get
Write-Host "health: $($health | ConvertTo-Json -Compress)"

$rows = @()
for ($r = 0; $r -lt $Repeats; $r++) {
    $i = 0
    foreach ($p in $prompts) {
        $body = @{
            model            = $ModelId
            messages         = @(@{ role = 'user'; content = $p })
            max_tokens       = $MaxTokens
            temperature      = 0
            reasoning_effort = 'none'
        } | ConvertTo-Json -Depth 5

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body
        $sw.Stop()
        $out = $resp.usage.completion_tokens
        $msg = $resp.choices[0].message
        $text = [string]$msg.content + '|' + [string]$msg.reasoning_content + [string]$msg.reasoning
        $rows += [pscustomobject]@{
            round        = $r
            task         = $i
            prompt_tok   = $resp.usage.prompt_tokens
            output_tok   = $out
            latency_ms   = [math]::Round($sw.Elapsed.TotalMilliseconds)
            client_tok_s = [math]::Round($out / $sw.Elapsed.TotalSeconds, 1)
            sha          = (Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes($text))) -Algorithm SHA1).Hash.Substring(0, 8)
        }
        $i++
    }
}

$rows | Format-Table -AutoSize
$tot = ($rows | Measure-Object latency_ms -Sum).Sum / 1000
$tok = ($rows | Measure-Object output_tok -Sum).Sum
Write-Host ("aggregate: {0} output tok in {1:N1}s -> {2:N1} tok/s (client-side, incl. prefill)" -f $tok, $tot, ($tok / $tot))
