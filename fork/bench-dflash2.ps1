param(
    [int]$Port = 30001,
    [string]$ModelId = 'qwen-local-dflash2',
    [int]$MaxTokens = 256,
    [int]$Repeats = 2
)

$ErrorActionPreference = 'Stop'

$prompts = @(
    'Explain how tensor parallelism splits a transformer layer across two GPUs. Answer in about 150 words.',
    'Write a Python function that returns the n-th Fibonacci number iteratively, then explain its time complexity.',
    'List ten common causes of NaN values in mixed-precision neural network inference and how to detect each.',
    'Summarize the plot of a heist movie in which the crew are all retired mathematicians.'
)

$health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get
Write-Host "health: $($health | ConvertTo-Json -Compress)"

$rows = @()
for ($r = 0; $r -lt $Repeats; $r++) {
    $i = 0
    foreach ($p in $prompts) {
        $body = @{
            model       = $ModelId
            messages    = @(@{ role = 'user'; content = $p })
            max_tokens  = $MaxTokens
            temperature = 0
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
            prompt       = $i
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
