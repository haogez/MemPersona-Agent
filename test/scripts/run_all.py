from __future__ import annotations

import argparse
import json
from typing import Callable, Dict, List

from . import (
    build_graph,
    fullcontext,
    run_causalrag,
    run_closedbook,
    run_graphrag_global,
    run_graphrag_local,
    run_hipporag2,
    run_memrag,
    run_regular_rag,
)
from .eval_utils import build_output_paths, model_tag_from_card


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TIME-Lite evaluations")
    parser.add_argument("--methods", type=str, default=None, help="comma-separated method names")
    parser.add_argument("--mode", choices=["closedbook", "memrag", "all"], default="all")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--neighbor", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_build_graph", action="store_true")
    parser.add_argument("--model_card", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--enable_summary_llm", action="store_true")
    args = parser.parse_args(argv)

    method_map: Dict[str, Callable[[List[str]], None]] = {
        "closedbook": run_closedbook.main,
        "fullcontext": fullcontext.main,
        "memrag": run_memrag.main,
        "regular_rag": run_regular_rag.main,
        "graphrag_local": run_graphrag_local.main,
        "graphrag_global": run_graphrag_global.main,
        "hipporag2": run_hipporag2.main,
        "causalrag": run_causalrag.main,
    }

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        if args.mode == "closedbook":
            methods = ["closedbook"]
        elif args.mode == "memrag":
            methods = ["memrag"]
        else:
            methods = ["closedbook", "fullcontext", "memrag"]

    if any(m in {"memrag", "graphrag_local", "graphrag_global", "hipporag2", "causalrag"} for m in methods) and not args.skip_build_graph:
        build_graph.main(
            [
                *(["--split", args.split] if args.split else []),
                *(["--limit", str(args.limit)] if args.limit else []),
                "--wipe",
            ]
        )

    metrics: Dict[str, Dict[str, float]] = {}
    for method in methods:
        runner = method_map.get(method)
        if not runner:
            raise SystemExit(f"Unknown method: {method}")
        base_args = [
            *(["--split", args.split] if args.split else []),
            *(["--limit", str(args.limit)] if args.limit else []),
            *(["--max_samples", str(args.max_samples)] if args.max_samples else []),
            *(["--model_card", args.model_card] if args.model_card else []),
            *(["--base_url", args.base_url] if args.base_url else []),
            *(["--api_key", args.api_key] if args.api_key else []),
            *(["--out_dir", args.out_dir] if args.out_dir else []),
            *(["--resume"] if args.resume else []),
        ]
        method_args = list(base_args)
        if method in {"memrag", "regular_rag", "graphrag_local", "graphrag_global", "hipporag2", "causalrag"}:
            method_args += ["--k", str(args.k)]
        if method in {"memrag", "graphrag_local"}:
            method_args += ["--neighbor", str(args.neighbor)]
        if method in {"graphrag_global", "regular_rag"} and args.rebuild_index:
            method_args.append("--rebuild_index")
        if method in {"causalrag"}:
            method_args += ["--steps", str(args.steps)]
            if args.enable_summary_llm:
                method_args.append("--enable_summary_llm")
        runner(method_args)
        model_tag = model_tag_from_card(args.model_card)
        _, metrics_path = build_output_paths(method, model_tag, args.split, args.out_dir)
        if metrics_path.exists():
            metrics[method] = json.loads(metrics_path.read_text(encoding="utf-8"))

    if metrics:
        summary = {f"{name}_acc": info.get("accuracy") for name, info in metrics.items()}
        summary["total"] = next(iter(metrics.values())).get("total")
        summary_path = build_output_paths("summary", model_tag_from_card(args.model_card), args.split, args.out_dir)[1]
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
