$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8000"
$repoRoot = Split-Path $PSScriptRoot -Parent
$personaPath = Join-Path $repoRoot "artifacts\personas.jsonl"

if (-not (Test-Path $personaPath)) {
  Write-Error "personas.jsonl not found; run /persona/generate first."
  exit 1
}

$lastLine = Get-Content -Path $personaPath -Encoding UTF8 | Select-Object -Last 1
if (-not $lastLine) {
  Write-Error "personas.jsonl is empty."
  exit 1
}

$record = $lastLine | ConvertFrom-Json
$cid = $record.character_id
$persona = $record.persona

if (-not $cid -or -not $persona) {
  Write-Error "Failed to read character_id/persona from personas.jsonl."
  exit 1
}

$payload = @{
  character_id  = $cid
  persona       = $persona
  user_input    = "那次具体发生了什么？"
  inject_memory = $true
} | ConvertTo-Json -Depth 20

$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
$bodyPath = Join-Path $env:TEMP "chat_stream_test_body.json"
[System.IO.File]::WriteAllBytes($bodyPath, $bytes)

$outPath = Join-Path $env:TEMP "chat_stream_test.log"
if (Test-Path $outPath) { Remove-Item $outPath -Force }

curl.exe -N --no-buffer --http1.1 -X POST "$base/chat/stream" `
  -H "Accept: text/event-stream" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@$bodyPath" | Tee-Object -FilePath $outPath | Out-Host

$raw = Get-Content -Path $outPath -Raw
$deltaIndex = $raw.IndexOf("event: delta")
$suppIndex = $raw.IndexOf("event: supplement")
$metaIndex = $raw.IndexOf("event: meta")

if ($deltaIndex -lt 0) { Write-Error "Missing delta event." ; exit 1 }
if ($metaIndex -lt 0) { Write-Error "Missing meta event." ; exit 1 }
if ($suppIndex -ge 0 -and $suppIndex -lt $deltaIndex) { Write-Error "Supplement before delta." ; exit 1 }
if ($metaIndex -lt $deltaIndex) { Write-Error "Meta before delta." ; exit 1 }

$metaMatch = [regex]::Match($raw, "event: meta\s+data: (\{.*?\})\s+event:", "Singleline")
if (-not $metaMatch.Success) { Write-Error "Meta JSON not found." ; exit 1 }
$meta = $metaMatch.Groups[1].Value | ConvertFrom-Json

if (-not $meta.decision) { Write-Error "Meta missing decision." ; exit 1 }
if (-not $meta.timing) { Write-Error "Meta missing timing." ; exit 1 }
if ($null -eq $meta.facts_allowed) { Write-Error "Meta missing facts_allowed." ; exit 1 }
if (-not $meta.errors) { Write-Error "Meta missing errors." ; exit 1 }

Write-Host "OK: delta/meta present, order OK, meta schema OK."
