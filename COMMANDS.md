# MemPersona-Agent 全流程测试指令（Bash，一段段复制运行）

## 0) 前置
- 位置：项目根目录。
- 依赖：`pip install -r requirements.txt`；`apt-get update && apt-get install -y jq`（如无 jq 可用 Python 解析）。
- 启动 vLLM：确保 `http://127.0.0.1:18000/v1` 可用（model: qwen3-14b，api_key 任意非空）。
- LLM 环境变量（必设，启动前执行）：
```bash
export LLM_API_BASE_URL=http://127.0.0.1:18000/v1
export LLM_API_KEY=dummy
export LLM_MODEL_NAME=qwen3-14b
export CHAT_MODEL_NAME=qwen3-14b
```
- 可选 Neo4j（如有容器）：`export NEO4J_URI=bolt://127.0.0.1:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=zyh123456 NEO4J_DB=neo4j`
- 启动应用：`python -m uvicorn mem_persona_agent.api.main:app --host 127.0.0.1 --port 8000`
- 统一变量：`base="http://127.0.0.1:8000"`

## 1) 顺序测试
```bash
base="http://127.0.0.1:8000"

# 0. 健康检查
curl -s "$base/health"

# 1. 一键清空
curl -s -X POST "$base/reset/all" -H "content-type: application/json"

# 2. 生成 persona
seed="一个因为从小家中父母吵架不合而自卑敏感脆弱，但会和喜爱的小动物待在一起寻找慰藉的女高中生，叫白心怡，17岁"
resp=$(curl -s -X POST "$base/persona/generate" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"seed\":\"$seed\"}")
CHAR_ID=$(echo "$resp" | jq -r '.character_id')
PERSONA=$(echo "$resp" | jq -c '.persona')
echo "CHAR_ID=$CHAR_ID"

# 3. 列出角色
curl -s "$base/persona/list?limit=20"

# 4. 生成记忆
curl -s -X POST "$base/memory/static/generate" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"character_id\":\"$CHAR_ID\",\"persona\":$PERSONA,\"seed\":\"$seed\"}"

# 5. Scene 检索调试
curl -s "$base/debug/jsonl/stats?owner_id=$CHAR_ID&query=$(python - <<'PY'\nimport urllib.parse\nprint(urllib.parse.quote('争吵'))\nPY)"

# 6. 非流式 /chat（首次回应）
curl -s -X POST "$base/chat" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"character_id\":\"$CHAR_ID\",\"persona\":$PERSONA,\"place\":\"咖啡店\",\"npc\":\"妈妈\",\"mode\":\"static_only\",\"dialogue_history\":[],\"user_input\":\"你还记得高中那次误会吗？\"}"

# 7. 非流式 /chat（追问）
curl -s -X POST "$base/chat" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"character_id\":\"$CHAR_ID\",\"persona\":$PERSONA,\"place\":\"咖啡店\",\"npc\":\"妈妈\",\"mode\":\"static_only\",\"dialogue_history\":[],\"user_input\":\"那次具体发生了什么？\"}"

# 8. 流式 /chat/stream（SSE: stage_a / stage_b / meta）

curl -N --http1.1 -X POST "$base/chat/stream" \
  -H "accept: text/event-stream" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"character_id\":\"$CHAR_ID\",\"user_input\":\"你还记得在宠物医院的事情吗\",\"inject_memory\":true}"

curl -N --http1.1 -X POST "$base/chat/stream" \
  -H "accept: text/event-stream" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"character_id\":\"$CHAR_ID\",\"user_input\":\"陆医生还说了什么？\",\"inject_memory\":true}"

curl -N --http1.1 -X POST "$base/chat/stream" \
  -H "accept: text/event-stream" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"character_id\":\"$CHAR_ID\",\"user_input\":\"就说了这一句话吗？\",\"inject_memory\":true}"

curl -N --http1.1 -X POST "$base/chat/stream" \
  -H "accept: text/event-stream" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"character_id\":\"$CHAR_ID\",\"user_input\":\"你仔细想想他还说了什么？\",\"inject_memory\":true}"

# 9. 删除单个角色
curl -s -X POST "$base/persona/delete" \
  -H "content-type: application/json; charset=utf-8" \
  -d "{\"character_id\":\"$CHAR_ID\"}"

# 10. 删除全部角色
curl -s -X POST "$base/persona/delete_all"

# 11. 一键清空（收尾）
curl -s -X POST "$base/reset/all" -H "content-type: application/json"
```
