# MemPersona-Agent

MemPersona-Agent（忆格人格体框架）是一套以“真实人格 + 真实人生记忆 + 类人类认知结构”为目标的角色智能体框架。用户一句话生成细粒度 20 维角色档案，并衍生 3-5 条静态人生记忆 Episode 存入 Neo4j 及向量索引，支持“边说话边回忆”的单次 LLM 对话体验。

## 目录结构
```
mem-persona-agent/
  README.md
  requirements.txt
  Dockerfile
  .env.example
  src/
    mem_persona_agent/
      config.py
      message.py
      llm/
      persona/
      memory/
      agent/
      flow/
      api/
  tests/
```

## 环境准备
1. 复制 `.env.example` 为 `.env` 并填写 LLM 与 Neo4j 连接信息。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 运行服务
本地运行：
```bash
export PYTHONPATH=./src
uvicorn mem_persona_agent.api.main:app --host 0.0.0.0 --port 8000
```

### Docker 部署
```
docker build -t mem-persona-agent .
docker run -it --env-file .env -p 8000:8000 mem-persona-agent
```

## Neo4j 配置与向量索引
- 确保 Neo4j 5.11+ 开启向量索引功能。
- GraphStore 启动时执行以下 schema：
  - `CREATE CONSTRAINT static_episode_id_unique IF NOT EXISTS FOR (e:StaticEpisode) REQUIRE e.id IS UNIQUE;`
  - `CREATE VECTOR INDEX episode_embedding_index IF NOT EXISTS FOR (e:StaticEpisode) ON (e.embedding) OPTIONS { indexConfig: { 'vector.dim': 1536, 'vector.similarity_function': 'cosine' }};`
- 所有节点都携带 `owner_id`，防止不同角色记忆混用。

## API 示例
### 生成人设
`POST /persona/generate`
```json
{"seed": "爱玩街舞的深圳高中女生"}
```
返回：`character_id` + 20 维 persona。

### 生成静态记忆
`POST /memory/static/generate`
```json
{"character_id": "<uuid>", "persona": { ...20维... }}
```
返回：生成并写入的 episodes。

### 聊天
`POST /chat`
```json
{
  "character_id": "<uuid>",
  "persona": { ...20维... },
  "place": "咖啡馆",
  "npc": "妈妈",
  "mode": "static_only",
  "dialogue_history": [],
  "user_input": "你为什么喜欢咖啡？"
}
```
返回：角色回复与检索到的记忆片段。

## 对话示例
```
用户：妈妈今天早上又给你打电话了吗？
助手：我刚在厨房煮咖啡时接到了她的电话，她叮嘱我别忘了穿外套，我答应晚点回拨给她。
```

## 测试
```bash
PYTHONPATH=./src pytest
```
