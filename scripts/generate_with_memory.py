from __future__ import annotations

import argparse
import json
import sys

import httpx


def main():
    parser = argparse.ArgumentParser(description="Generate persona and static memories via API.")
    parser.add_argument("seed", help="角色 seed 描述")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base url")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    persona_resp = _post(f"{base}/persona/generate", {"seed": args.seed})
    character_id = persona_resp.get("character_id")
    persona = persona_resp.get("persona")
    if not character_id or not persona:
        sys.stderr.write(f"Persona generation failed: {persona_resp}\n")
        sys.exit(1)

    memory_resp = _post(f"{base}/memory/static/generate", {"character_id": character_id, "persona": persona, "seed": args.seed})

    print(json.dumps({"character_id": character_id, "persona": persona, "memories": memory_resp}, ensure_ascii=False, indent=2))


def _post(url: str, payload: dict) -> dict:
    try:
        res = httpx.post(url, json=payload, timeout=120.0)
        res.raise_for_status()
        return res.json()
    except Exception as exc:  # pragma: no cover - CLI helper
        sys.stderr.write(f"Request failed {url}: {exc}\n")
        return {}


if __name__ == "__main__":
    main()
