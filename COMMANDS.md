# MemPersona-Agent 使用指令速查

## 0. 前置
- 启动服务：`uvicorn mem_persona_agent.api.main:app --host 127.0.0.1 --port 8000`
- 确保 `.env` 配好 LLM/Neo4j，默认模型：`gpt-4.1-mini`。
- PowerShell 避免乱码：`[Console]::InputEncoding=[System.Text.Encoding]::UTF8; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8`，发送 JSON 时用 UTF-8 bytes。

## 1) 生成角色（返回 character_id + persona，并写入 Neo4j 与 artifacts/personas.jsonl）
PowerShell（UTF-8 bytes）:
```powershell
$json = '{"seed":"一个表面阳光但内心黑暗的女高中生，叫周静宜，17岁"}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/persona/generate" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $bytes
```
Bash:
```bash
curl -s -X POST http://127.0.0.1:8000/persona/generate \
  -H "Content-Type: application/json" \
  --data-binary '{"seed":"一个表面阳光但内心黑暗的女高中生，叫周静宜，17岁"}'
```

## 2) 列出已生成角色（本地 JSONL）
PowerShell:
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/persona/list?limit=20" -Method GET |
Select-Object -ExpandProperty Content
```
Bash: `curl -s "http://127.0.0.1:8000/persona/list?limit=20"`

## 3) 删除单个角色（本地 JSONL + Neo4j 及静态记忆）
PowerShell（UTF-8 bytes）:
```powershell
$json = '{"character_id":"<要删除的ID>"}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/persona/delete" `
  -Method POST -ContentType "application/json; charset=utf-8" `
  -Body $bytes
```
Bash:
```bash
curl -s -X POST http://127.0.0.1:8000/persona/delete \
  -H "Content-Type: application/json" \
  --data-binary '{"character_id":"<要删除的ID>"}'
```

## 3b) 删除所有角色（清空 JSONL、缓存、Neo4j 角色及静态记忆）
PowerShell:
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/persona/delete_all" -Method POST |
Select-Object -ExpandProperty Content
```
Bash: `curl -s -X POST http://127.0.0.1:8000/persona/delete_all`

## 4) 生成静态记忆并写入 Neo4j
（需使用步骤 1 返回的 persona 对象与同一个 character_id）
PowerShell（UTF-8 bytes）:
```powershell
$json = '{"character_id":"<cid>","persona":<persona对象>}' 
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/memory/static/generate" `
  -Method POST -ContentType "application/json; charset=utf-8" `
  -Body $bytes
```
Bash:
```bash
curl -s -X POST http://127.0.0.1:8000/memory/static/generate \
  -H "Content-Type: application/json" \
  --data-binary '{"character_id":"<cid>","persona":<persona对象>}'
```

## 5) 带人设 + 记忆检索的对话
PowerShell（UTF-8 bytes）:
```powershell
$json = '{"character_id":"<cid>","persona":<persona对象>,"place":"咖啡店","npc":"妈妈","mode":"static_only","dialogue_history":[],"user_input":"你为什么喜欢咖啡？"}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" `
  -Method POST -ContentType "application/json; charset=utf-8" `
  -Body $bytes
```
Bash:
```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  --data-binary '{"character_id":"<cid>","persona":<persona对象>,"place":"咖啡店","npc":"妈妈","mode":"static_only","dialogue_history":[],"user_input":"你为什么喜欢咖啡？"}'
```

## 6) 一行跑全流程（生成人设+记忆+对话，落盘 JSON）
Python 同步示例（适合脚本批量）：
```python
from mem_persona_agent.flow import run_full_pipeline_sync

res = run_full_pipeline_sync(
    seed="爱街舞的深圳高中女生",
    user_input="你为什么喜欢咖啡？",
    place="咖啡店",
    npc="妈妈",
    timeline_mode="strict",
    save_dir="artifacts"
)
print(res["paths"])  # persona/episodes/chat JSON 路径
```

## 7) 查看/管理本地角色文件
- 路径：`artifacts/personas.jsonl`
- 查看：`Get-Content artifacts/personas.jsonl -Encoding utf8` 或用支持 UTF-8 的编辑器打开。
- 该文件每条记录 `{character_id, persona}`，可替换为大规模细粒度人设数据集做测试。
