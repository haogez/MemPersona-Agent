# MemPersona-Agent 指令速查（UTF-8 发送）

## 0. 前置
- 环境： MemPersona-Agent 目录下，PowerShell 激活虚拟环境 
        `.\.venv\Scripts\Activate.ps1`
- 启动：`uvicorn mem_persona_agent.api.main:app --host 127.0.0.1 --port 8000`
- `.env` 配好 LLM/Neo4j，`LLM_TIMEOUT_SECONDS=0` 表示不限时。
- PowerShell 编码：`[Console]::InputEncoding=[System.Text.Encoding]::UTF8; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8`
- 发送 JSON 用 UTF-8 bytes；curl 用 `--data-binary`。

## 1) 顺序测试全流程（逐个接口，按顺序执行，可直接复制）
PowerShell（UTF-8 bytes）：
```powershell
$base = "http://127.0.0.1:8000"
$cid = "f01930d0-7ed0-45f6-bbb6-70f26a9a8883"

# 0) 健康检查
Invoke-WebRequest -UseBasicParsing -Uri "$base/health" -Method GET | Select-Object -ExpandProperty Content

# 1) 一键清空
$json = '{}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "$base/reset/all" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes

# 2) 生成 persona
$seed = "一个因为从小家中父母吵架不合而自卑敏感脆弱，但会和喜爱的小动物待在一起寻找慰藉的女高中生，叫白心怡，17岁"
$json = "{`"seed`":`"$seed`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
$resp = Invoke-RestMethod -Uri "$base/persona/generate" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes
$cid = $resp.character_id
$persona = $resp.persona

# 3) 列出角色
Invoke-WebRequest -UseBasicParsing -Uri "$base/persona/list?limit=20" -Method GET | Select-Object -ExpandProperty Content

# 4) 生成记忆（related_characters → worldrule → inspiration → scene_memories → detail_graphs）
$payload = @{ character_id = $cid; persona = $persona; seed = $seed } | ConvertTo-Json -Depth 20
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
Invoke-RestMethod -Uri "$base/memory/static/generate" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes

# 5) Scene 检索调试（检索统计）
$q = [System.Uri]::EscapeDataString("争吵")
Invoke-WebRequest -UseBasicParsing -Uri "$base/debug/jsonl/stats?owner_id=$cid&query=$q" -Method GET |
  Select-Object -ExpandProperty Content

# 6) 非流式 /chat（StageA 首次回应）
$payload = @{
  character_id = $cid
  persona = $persona
  place = "咖啡店"
  npc = "妈妈"
  mode = "static_only"
  dialogue_history = @()
  user_input = "你还记得高中那次误会吗？"
} | ConvertTo-Json -Depth 20
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
Invoke-RestMethod -Uri "$base/chat" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes

# 7) 非流式 /chat（追问细节）
$payload = @{
  character_id = $cid
  persona = $persona
  place = "咖啡店"
  npc = "妈妈"
  mode = "static_only"
  dialogue_history = @()
  user_input = "那次具体发生了什么？"
} | ConvertTo-Json -Depth 20
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
Invoke-RestMethod -Uri "$base/chat" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes

# 8) 流式 /chat/stream（实时输出，禁缓冲；persona 可省略）
$payload = @{ character_id = $cid; user_input = "你好，你叫什么名字呀？"; inject_memory = $true } | ConvertTo-Json -Depth 20
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
$bodyPath = Join-Path $env:TEMP "chat_stream_body.json"
[System.IO.File]::WriteAllBytes($bodyPath, $bytes)
$ProgressPreference = 'SilentlyContinue'
curl.exe -N --no-buffer --http1.1 -X POST "$base/chat/stream" `
  -H "Accept: text/event-stream" `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@$bodyPath"

# 9) 删除单个角色
$json = "{`"character_id`":`"$cid`"}"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "$base/persona/delete" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes

# 10) 删除全部角色
Invoke-WebRequest -UseBasicParsing -Uri "$base/persona/delete_all" -Method POST | Select-Object -ExpandProperty Content

# 11) 一键清空（再次清理）
$json = '{}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "$base/reset/all" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes
```
