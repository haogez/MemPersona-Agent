from __future__ import annotations

import argparse
from typing import List

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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: closedbook -> build graph -> selected RAG methods"
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="limit samples for quick run")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--neighbor", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--methods", type=str, default="memrag")
    parser.add_argument("--model_card", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--enable_summary_llm", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume graphrag/closedbook if outputs exist")
    parser.add_argument("--no_wipe", action="store_true", help="skip wiping Neo4j before build")
    args = parser.parse_args(argv)

    # 1) Closed-book
    run_closedbook.main(
        [
            *(["--split", args.split] if args.split else []),
            *(["--limit", str(args.limit)] if args.limit else []),
            *(["--max_samples", str(args.max_samples)] if args.max_samples else []),
            *(["--model_card", args.model_card] if args.model_card else []),
            *(["--base_url", args.base_url] if args.base_url else []),
            *(["--api_key", args.api_key] if args.api_key else []),
            *(["--out_dir", args.out_dir] if args.out_dir else []),
            *(["--resume"] if args.resume else []),
        ]
    )

    # 2) Build graph (default wipe unless no_wipe)
    build_args = []
    if args.split:
        build_args += ["--split", args.split]
    if args.limit:
        build_args += ["--limit", str(args.limit)]
    if not args.no_wipe:
        build_args.append("--wipe")
    build_graph.main(build_args)

    # 3) Selected methods
    method_map = {
        "memrag": run_memrag.main,
        "fullcontext": fullcontext.main,
        "regular_rag": run_regular_rag.main,
        "graphrag_local": run_graphrag_local.main,
        "graphrag_global": run_graphrag_global.main,
        "hipporag2": run_hipporag2.main,
        "causalrag": run_causalrag.main,
    }
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        runner = method_map.get(method)
        if not runner:
            raise SystemExit(f"Unknown method: {method}")
        runner_args: List[str] = [
            *(["--split", args.split] if args.split else []),
            *(["--limit", str(args.limit)] if args.limit else []),
            *(["--max_samples", str(args.max_samples)] if args.max_samples else []),
            *(["--model_card", args.model_card] if args.model_card else []),
            *(["--base_url", args.base_url] if args.base_url else []),
            *(["--api_key", args.api_key] if args.api_key else []),
            *(["--out_dir", args.out_dir] if args.out_dir else []),
            *(["--resume"] if args.resume else []),
        ]
        if method in {"memrag", "regular_rag", "graphrag_local", "graphrag_global", "hipporag2", "causalrag"}:
            runner_args += ["--k", str(args.k)]
        if method in {"memrag", "graphrag_local"}:
            runner_args += ["--neighbor", str(args.neighbor)]
        if method in {"graphrag_global", "regular_rag"} and args.rebuild_index:
            runner_args.append("--rebuild_index")
        if method == "causalrag":
            runner_args += ["--steps", str(args.steps)]
            if args.enable_summary_llm:
                runner_args.append("--enable_summary_llm")
        runner(runner_args)


if __name__ == "__main__":
    main()
