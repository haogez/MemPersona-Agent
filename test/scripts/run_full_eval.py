from __future__ import annotations

import argparse

from . import build_graph, run_closedbook, run_memrag


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: closedbook -> build graph -> graphrag"
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="limit samples for quick run")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--neighbor", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="resume graphrag/closedbook if outputs exist")
    parser.add_argument("--no_wipe", action="store_true", help="skip wiping Neo4j before build")
    args = parser.parse_args(argv)

    # 1) Closed-book
    run_closedbook.main(
        [
            *(["--split", args.split] if args.split else []),
            *(["--limit", str(args.limit)] if args.limit else []),
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

    # 3) Graph-RAG
    run_memrag.main(
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


if __name__ == "__main__":
    main()
