param([string[]]$Tags)
# Summarize per-request engine timings from the server's JSONL request log.
$out = (Join-Path $PSScriptRoot 'out')
foreach ($t in $Tags) {
    $f = Join-Path $out "reqlog-$t.jsonl"
    if (-not (Test-Path $f)) { "missing $f"; continue }
    "=== $t ==="
    $rows = Get-Content $f | ForEach-Object {
        $j = $_ | ConvertFrom-Json
        if ($j.event -eq 'request_done') {
            $e = $j.engine_timing; $ts = $j.timings_seconds; $r = $j.result
            [pscustomobject]@{
                prompt     = $r.prompt_tokens
                computed   = $r.computed_prefill_tokens
                cache_hit  = $r.prefix_cache_hit_tokens
                path       = $r.prefix_reuse_path
                out        = $r.completion_tokens
                prefill_s  = [math]::Round($ts.prefill, 3)
                ttft_s     = [math]::Round($ts.ttft, 3)
                units      = $e.units.prefill
                ms_per_tok = if ($r.computed_prefill_tokens -gt 0) { [math]::Round(1000 * $ts.prefill / $r.computed_prefill_tokens, 3) } else { 0 }
            }
        }
    }
    $rows | Format-Table -AutoSize | Out-String -Width 200
    # linear fit prefill_s = a + b*computed over rows with computed >= 256
    $fit = $rows | Where-Object { $_.computed -ge 256 }
    if ($fit.Count -ge 2) {
        $n = $fit.Count; $sx = ($fit | Measure-Object computed -Sum).Sum; $sy = ($fit | Measure-Object prefill_s -Sum).Sum
        $sxx = ($fit | ForEach-Object { [double]$_.computed * $_.computed } | Measure-Object -Sum).Sum
        $sxy = ($fit | ForEach-Object { [double]$_.computed * $_.prefill_s } | Measure-Object -Sum).Sum
        $b = ($n * $sxy - $sx * $sy) / ($n * $sxx - $sx * $sx); $a = ($sy - $b * $sx) / $n
        "fit: prefill_s = {0:N3} s + {1:N4} ms/tok  => marginal {2:N0} tok/s" -f $a, ($b * 1000), (1 / $b)
    }
}

