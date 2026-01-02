param(
  [string]$Base = "http://127.0.0.1:8000",
  [string]$Cid = ""
)

if (-not $Cid) {
  Write-Host "Usage: .\\tests\\test_gemini_stream.ps1 -Cid <character_id> [-Base http://127.0.0.1:8000]"
  exit 1
}

$ErrorActionPreference = "Stop"

function Invoke-Stream($payload) {
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
  $bodyPath = Join-Path $env:TEMP "chat_stream_body.json"
  [System.IO.File]::WriteAllBytes($bodyPath, $bytes)
  $ProgressPreference = 'SilentlyContinue'
  $raw = & curl.exe -N --no-buffer --http1.1 -X POST "$Base/chat/stream" `
    -H "Accept: text/event-stream" `
    -H "Content-Type: application/json; charset=utf-8" `
    --data-binary "@$bodyPath"
  return $raw
}

Write-Host "case1: normal streaming"
$payload1 = @{ character_id = $Cid; user_input = "你好" ; inject_memory = $true } | ConvertTo-Json -Depth 20
$out1 = Invoke-Stream $payload1
$deltaCount = ($out1 | Select-String -Pattern "^event:\s*delta").Count
Write-Host "delta_count=$deltaCount"
if ($deltaCount -lt 3) {
  Write-Host "case1 failed: delta count < 3"
  exit 1
}

Write-Host "case2: wrong key (requires server started with DMX_API_KEY invalid)"
if (-not $env:GEMINI_BAD_KEY_TEST) {
  Write-Host "Skip case2: set `$env:GEMINI_BAD_KEY_TEST=1 and restart server with bad DMX_API_KEY"
  exit 0
}

$payload2 = @{ character_id = $Cid; user_input = "回忆一下那次争吵"; inject_memory = $true } | ConvertTo-Json -Depth 20
$out2 = Invoke-Stream $payload2
$metaLine = ($out2 | Select-String -Pattern "^data:\s*\\{.*\\}$" | Select-Object -Last 1).Line
if (-not $metaLine) {
  Write-Host "case2 failed: meta not found"
  exit 1
}
$metaJson = $metaLine -replace "^data:\s*", ""
try {
  $meta = $metaJson | ConvertFrom-Json
} catch {
  Write-Host "case2 failed: meta json parse error"
  exit 1
}
$stageB = $meta.errors.stage_b
if (-not $stageB -or ($stageB -notmatch "403")) {
  Write-Host "case2 failed: meta.errors.stage_b missing 403/body"
  exit 1
}
Write-Host "case2 ok"
