import asyncio
import json
import sys

import httpx


async def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/demo_stream_chat.py <base_url> <character_id> '<persona_json>'")
        sys.exit(1)
    base_url = sys.argv[1]
    cid = sys.argv[2]
    persona = json.loads(sys.argv[3])

    payload = {
        "character_id": cid,
        "persona": persona,
        "user_input": "你好，今天心情如何？",
        "history": [],
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{base_url}/chat/stream", json=payload) as resp:
            resp.raise_for_status()
            current_event = None
            buffer = ""
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("data:"):
                    data = line.split(":", 1)[1].strip()
                    if current_event == "delta" or current_event is None:
                        buffer += data
                        sys.stdout.write(data)
                        sys.stdout.flush()
                    elif current_event == "meta":
                        sys.stdout.write("\n\n[META]\n")
                        sys.stdout.write(json.dumps(json.loads(data), ensure_ascii=False, indent=2))
                        sys.stdout.write("\n")
                    elif current_event == "end":
                        break


if __name__ == "__main__":
    asyncio.run(main())
