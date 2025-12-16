# MemPersona-Agent 指令速查（UTF-8 发送）

## 0. 前置
- 启动：`uvicorn mem_persona_agent.api.main:app --host 127.0.0.1 --port 8000`
- `.env` 配好 LLM/Neo4j，`LLM_TIMEOUT_SECONDS=0` 表示不限时。
- PowerShell 编码：`[Console]::InputEncoding=[System.Text.Encoding]::UTF8; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8`
- 发送 JSON 用 UTF-8 bytes；curl 用 `--data-binary`。

## 1) 生成角色（写入 Neo4j + artifacts/personas.jsonl）
PowerShell：
```powershell
$json = '{"seed":"一个表面阳光但内心阴暗的女高中生，叫周静宜，17岁"}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/persona/generate" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes
```
curl：
```bash
curl -s -X POST http://127.0.0.1:8000/persona/generate \
  -H "Content-Type: application/json" \
  --data-binary '{"seed":"一个表面阳光但内心阴暗的女高中生，叫周静宜，17岁"}'
```

## 2) 列出角色（本地 JSONL）
PowerShell：`Invoke-WebRequest -Uri "http://127.0.0.1:8000/persona/list?limit=20" -Method GET | Select-Object -ExpandProperty Content`
curl：`curl -s "http://127.0.0.1:8000/persona/list?limit=20"`

## 3) 删除角色 / 删除全部 / 重置
- 单个（PowerShell UTF-8 bytes）：
```powershell
$json = '{"character_id":"<ID>"}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/persona/delete" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes
```
- 删除全部：`Invoke-WebRequest -Uri "http://127.0.0.1:8000/persona/delete_all" -Method POST | Select-Object -ExpandProperty Content`
- 重置全部（角色+记忆+关联文件）：`Invoke-WebRequest -Uri "http://127.0.0.1:8000/reset/all" -Method POST | Select-Object -ExpandProperty Content`

## 4) 生成静态记忆（写入 Neo4j + artifacts/memories.jsonl + related_characters.jsonl）
需要 persona、character_id，可带 seed：
```powershell
$json = '{"character_id":"<cid>","persona":<persona对象>,"seed":"<seed>"}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/memory/static/generate" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes
```
curl：
```bash
curl -s -X POST http://127.0.0.1:8000/memory/static/generate \
  -H "Content-Type: application/json" \
  --data-binary '{"character_id":"<cid>","persona":<persona对象>,"seed":"<seed>"}'
```

## 5) 对话（带记忆检索）
```powershell
$json = '{"character_id":"<cid>","persona":<persona对象>,"place":"咖啡店","npc":"妈妈","mode":"static_only","dialogue_history":[],"user_input":"你为什么喜欢咖啡？"}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method POST `
  -ContentType "application/json; charset=utf-8" -Body $bytes
```
curl 同上，替换为 `--data-binary`。

## 6) 脚本一键生成（角色+静态记忆）
本地已启动 API 后：
```bash
python scripts/generate_with_memory.py "你的seed" --base-url http://127.0.0.1:8000
```

## 7) 一行跑全流程（代码）
```python
from mem_persona_agent.flow import run_full_pipeline_sync
res = run_full_pipeline_sync(seed="爱街舞的深圳高中女生", user_input="你为什么喜欢咖啡？", place="咖啡店", npc="妈妈")
print(res["paths"])  # persona/episodes/chat 路径
```

## 8) 本地文件
- 人设：`artifacts/personas.jsonl`
- 关联角色：`artifacts/related_characters.jsonl`
- 记忆：`artifacts/memories.jsonl`
- 查看：`Get-Content <file> -Encoding utf8`
