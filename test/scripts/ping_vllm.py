from __future__ import annotations

import os

from .utils import call_vllm, strip_think


def main() -> None:
    messages = [
        {"role": "system", "content": "You are a friendly assistant. Keep replies short."},
        {"role": "user", "content": "你好，测试一下 vLLM 服务是否正常？"},
    ]
    try:
        resp = call_vllm(messages, max_tokens=64, temperature=0.7, top_p=0.9)
        resp_clean = strip_think(resp)
        print("vLLM response:\n", resp_clean)
    except Exception as exc:
        print("vLLM call failed:", exc)


if __name__ == "__main__":
    main()
