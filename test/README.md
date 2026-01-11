# TIME-Lite 时序推理评测

## 前置
- 启动 vLLM（例）：  
  `vllm serve /data_zhouyuhao/models/Qwen/Qwen3/Qwen3-14B --host 127.0.0.1 --port 18000 --tensor-parallel-size 2 --max-model-len 8192 --served-model-name qwen3-14b`
- 环境变量（示例）：  
  ```
  export LLM_API_BASE_URL=http://127.0.0.1:18000/v1
  export LLM_API_KEY=dummy
  export LLM_MODEL_NAME=qwen3-14b
  export CHAT_MODEL_NAME=qwen3-14b
  export NEO4J_URI=bolt://127.0.0.1:7687
  export NEO4J_USER=neo4j
  export NEO4J_PASSWORD=your_password
  export NEO4J_DB=neo4j
  ```
- 依赖：`pip install -r requirements.txt`（新增 datasets、tqdm）

## 目录
```
test/
  data/
    time_lite/      # HF 数据缓存
    processed/
  outputs/
    logs/
    predictions/
    metrics/
  scripts/
    download_timelite.py
    build_graph.py
    run_closedbook.py
    run_memrag.py
    run_all.py
```

## 快速开始
```bash
cd projects/MemPersona-Agent
# 下载数据
python -m test.scripts.download_timelite

# 构建 Neo4j 图（会写入 tag=timelite，--wipe 清理旧数据）
python -m test.scripts.build_graph --wipe

# 仅闭卷
python -m test.scripts.run_closedbook --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1

# 仅 Full-context
python -m test.scripts.fullcontext --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1

# 仅 Graph-RAG (MemRAG)
python -m test.scripts.run_memrag --k 3 --neighbor 2 --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1

# Regular RAG
python -m test.scripts.run_regular_rag --k 3 --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1

# GraphRAG-Local
python -m test.scripts.run_graphrag_local --k 3 --neighbor 1 --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1

# GraphRAG-Global
python -m test.scripts.run_graphrag_global --k 3 --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1

# HippoRAG2
python -m test.scripts.run_hipporag2 --k 3 --path_k 8 --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1

# CausalRAG (可选摘要 LLM)
python -m test.scripts.run_causalrag --k 3 --steps 2 --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1

# 一键（构图+闭卷+Graph-RAG）
python -m test.scripts.run_all --mode all --limit 200 --model_card qwen3-14b --base_url http://127.0.0.1:18000/v1
```

## 输出
- 预测：`test/outputs/predictions/<method>_<model_tag>_<split>.jsonl`
- 指标：`test/outputs/metrics/<method>_<model_tag>_<split>.json`

## 参数
- `--limit N`：只跑前 N 条
- `--k`：Graph-RAG 检索 top-k 事件
- `--neighbor`：时间线邻居窗口
- `--resume`：断点续跑（跳过已有预测）
- `--skip_build_graph`（run_all）：跳过重建图
