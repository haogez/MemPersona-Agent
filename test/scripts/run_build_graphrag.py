from __future__ import annotations

import argparse
from typing import List

from . import build_graph, run_memrag


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build TIME-Lite graph then run Graph-RAG eval")
    parser.add_argument("--split", type=str, default=None, help="dataset split")
    parser.add_argument("--limit", type=int, default=None, help="limit samples for quick run")
    parser.add_argument("--k", type=int, default=3, help="top-k events to start timeline")
    parser.add_argument("--neighbor", type=int, default=20, help="neighbors before/after hit event")
    parser.add_argument("--resume", action="store_true", help="resume graphrag from existing predictions")
    parser.add_argument("--no_wipe", action="store_true", help="skip wiping old timelite data in Neo4j")
    args = parser.parse_args(argv)

    # Step 1: build graph (with wipe by default)
    build_args: List[str] = []
    if args.split:
        build_args += ["--split", args.split]
    if args.limit:
        build_args += ["--limit", str(args.limit)]
    if not args.no_wipe:
        build_args.append("--wipe")
    build_graph.main(build_args)

    # Step 2: run Graph-RAG eval
    rag_args: List[str] = []
    if args.split:
        rag_args += ["--split", args.split]
    if args.limit:
        rag_args += ["--limit", str(args.limit)]
    rag_args += ["--k", str(args.k), "--neighbor", str(args.neighbor)]
    if args.resume:
        rag_args.append("--resume")
    run_memrag.main(rag_args)


if __name__ == "__main__":
    main()
