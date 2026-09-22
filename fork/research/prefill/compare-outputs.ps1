param([int]$Port = 30006, [string]$Tag = 'x', [int]$MaxTokens = 64)
# Greedy completions for a fixed prompt set; writes out\outputs-<Tag>.json for A/B comparison.
$ErrorActionPreference = 'Stop'
$doc = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot '..\..\..\eval\corpora\perplexity-1m\data\pg19\00.txt')
$cases = @(
    @{ name = 'short-300';  off = 5000;   chars = 1200 },
    @{ name = 'mid-1100';   off = 40000;  chars = 4400 },
    @{ name = 'long-4200';  off = 90000;  chars = 17000 },
    @{ name = 'long-9000';  off = 150000; chars = 36000 }
)
$out = @{}
foreach ($c in $cases) {
    $prompt = "Continue the story from exactly where it stops.`n`n<document>`n$($doc.Substring($c.off, $c.chars))`n</document>"
    $body = @{ model = 'qwen-bench'; messages = @(@{ role = 'user'; content = $prompt }); max_tokens = $MaxTokens; temperature = 0; reasoning_effort = 'none'; presence_penalty = 0 } | ConvertTo-Json -Depth 5
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 900
    $text = [string]$r.choices[0].message.content
    $out[$c.name] = @{ prompt_tokens = $r.usage.prompt_tokens; text = $text }
    Write-Host ("{0,-10} prompt={1,5}  {2}" -f $c.name, $r.usage.prompt_tokens, $text.Substring(0, [Math]::Min(80, $text.Length)).Replace("`n", ' '))
}
$out | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $PSScriptRoot "out\outputs-$Tag.json") -Encoding UTF8
