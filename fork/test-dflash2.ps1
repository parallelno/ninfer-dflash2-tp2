param(
    [int]$Port = 30001
)

$ErrorActionPreference = 'Stop'

$health = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:$Port/health"
if ($health.status -ne 'ok') {
    throw "Server health check failed: $($health | ConvertTo-Json -Compress)"
}

$body = @{
    model = 'qwen-local-dflash2'
    messages = @(@{ role = 'user'; content = 'Reply with exactly: NInfer DFlash2 is ready.' })
    max_tokens = 32
    temperature = 0
    reasoning_effort = 'none'
} | ConvertTo-Json -Depth 5

$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/v1/chat/completions" `
    -ContentType 'application/json' -Body $body

$response.choices[0].message | ConvertTo-Json -Depth 5