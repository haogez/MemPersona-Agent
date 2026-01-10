from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import build_graph, run_closedbook, run_graphrag
from .utils import OUTPUT_DIR


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TIME-Lite evaluations")
    parser.add_argument("--mode", choices=["closedbook", "graphrag", "all"], default="all")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--neighbor", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_build_graph", action="store_true")
    args = parser.parse_args(argv)

    metrics = {}

    if args.mode in ("all", "graphrag") and not args.skip_build_graph:
        build_graph.main(
            [
                *(["--split", args.split] if args.split else []),
                *(["--limit", str(args.limit)] if args.limit else []),
                "--wipe",
            ]
        )

    if args.mode in ("all", "closedbook"):
        run_closedbook.main(
            [
                *(["--split", args.split] if args.split else []),
                *(["--limit", str(args.limit)] if args.limit else []),
                *(["--resume"] if args.resume else []),
            ]
        )
        metrics["closedbook"] = json.loads((OUTPUT_DIR / "metrics" / "closedbook.json").read_text(encoding="utf-8"))

    if args.mode in ("all", "graphrag"):
        run_graphrag.main(
            [
                *(["--split", args.split] if args.split else []),
                *(["--limit", str(args.limit)] if args.limit else []),
                "--k",
                str(args.k),
                "--neighbor",
                str(args.neighbor),
                *(["--resume"] if args.resume else []),
            ]
        )
        metrics["graphrag"] = json.loads((OUTPUT_DIR / "metrics" / "graphrag.json").read_text(encoding="utf-8"))

    if metrics:
        summary = {
            "closedbook_acc": metrics.get("closedbook", {}).get("accuracy"),
            "graphrag_acc": metrics.get("graphrag", {}).get("accuracy"),
            "total": metrics.get("closedbook", {}).get("total") or metrics.get("graphrag", {}).get("total"),
        }
        summary_path = OUTPUT_DIR / "metrics" / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
