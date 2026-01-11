from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import os
import re

from .utils import load_timelite, pick_fields

METRIC_FILES = {
    "closedbook": "test/outputs/metrics/closedbook.json",
    "fullcontext": "test/outputs/metrics/fullcontext.json",
    "memrag": "test/outputs/metrics/memrag.json",
}

# TIME-Lite 固定任务集合（按论文定义顺序）
TASK_ORDER = [
    "Extraction",
    "Localization",
    "Computation",
    "Duration_Compare",
    "Order_Compare",
    "Explicit_Reasoning",
    "Order_Reasoning",
    "Relative_Reasoning",
    "Co_temporality",
    "Timeline",
    "Counterfactual",
    "unknown",
]


def scene_id_from_sample(sample: Dict[str, Any], idx: int) -> str:
    return str(sample.get("id") or sample.get("sample_id") or sample.get("idx") or f"S{idx:06d}")


def load_metrics(path: str) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        return {}
    samples = data.get("samples", [])
    out = {}
    for rec in samples:
        sid = str(rec.get("id"))
        out[sid] = rec
    return out


def compute_accuracy(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    hits = sum(1 for rec in records if rec.get("correct"))
    return hits / len(records)


def model_tag_from_env() -> str:
    raw = (os.getenv("LLM_MODEL_NAME") or "model").strip()
    tag = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return tag or "model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-task accuracy for existing metrics files.")
    parser.add_argument("--split", type=str, default=None, help="dataset split")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="comma-separated method names (use new naming rule if provided)",
    )
    parser.add_argument("--model_tag", type=str, default=None, help="model tag for new naming rule")
    parser.add_argument("--metrics", type=str, default=None, help="comma-separated metrics file paths")
    args = parser.parse_args()

    dataset = load_timelite(args.split)
    sample0 = dataset[0]
    q_key, a_key, c_key = pick_fields(sample0)

    # map scene_id -> task，以及 index -> task 方便纯数字 id 对齐
    id2task: Dict[str, str] = {}
    idx2task: Dict[int, str] = {}
    for idx, sample in enumerate(dataset):
        sid = scene_id_from_sample(sample, idx)
        task = ""
        if isinstance(sample, dict):
            task = str(sample.get("task") or sample.get("Task") or "")
        task_val = task or "unknown"
        id2task[sid] = task_val
        idx2task[idx] = task_val

    metric_sources: Dict[str, str] = dict(METRIC_FILES)
    if args.metrics:
        metric_sources = {
            Path(path).stem: path for path in [p.strip() for p in args.metrics.split(",") if p.strip()]
        }
    elif args.methods:
        model_tag = args.model_tag or model_tag_from_env()
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        metric_sources = {
            method: f"test/outputs/metrics/{method}_{model_tag}_{args.split or 'default'}.json"
            for method in methods
        }
    else:
        if not any(Path(path).exists() for path in metric_sources.values()):
            model_tag = args.model_tag or model_tag_from_env()
            metric_sources = {
                method: f"test/outputs/metrics/{method}_{model_tag}_{args.split or 'default'}.json"
                for method in metric_sources.keys()
            }

    metric_data = {name: load_metrics(path) for name, path in metric_sources.items()}

    # 聚合所有 task
    all_tasks = set(TASK_ORDER)
    per_metric_stats = {}
    for name, records in metric_data.items():
        task_hits = defaultdict(int)
        task_total = defaultdict(int)
        for sid, rec in records.items():
            task = id2task.get(sid, None)
            if task is None:
                try:
                    idx = int(sid)
                    task = idx2task.get(idx, "unknown")
                except Exception:
                    task = "unknown"
            task_total[task] += 1
            if rec.get("correct"):
                task_hits[task] += 1
        per_metric_stats[name] = (task_hits, task_total)
        all_tasks.update(task_total.keys())

    # 打印对比表：每行一个 task，每列一个 metric，显示 acc 和 hits/total
    metric_names = list(metric_sources.keys())
    header = ["task"] + [f"{m}_acc" for m in metric_names] + [f"{m}_h/t" for m in metric_names]

    rows = []
    ordered_tasks = [t for t in TASK_ORDER if t in all_tasks] + [t for t in sorted(all_tasks) if t not in TASK_ORDER]
    for task in ordered_tasks:
        row = [task]
        # accuracy
        for m in metric_names:
            hits, total = per_metric_stats[m][0].get(task, 0), per_metric_stats[m][1].get(task, 0)
            acc = hits / total if total else 0.0
            row.append(f"{acc:.4f}" if total else "-")
        # hits/total
        for m in metric_names:
            hits, total = per_metric_stats[m][0].get(task, 0), per_metric_stats[m][1].get(task, 0)
            row.append(f"{hits}/{total}" if total else "-")
        rows.append(row)

    # 计算列宽
    col_widths = [len(col) for col in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    align_left = {0}  # task 列左对齐

    def fmt_row(row):
        cells = []
        for i, cell in enumerate(row):
            cell_str = str(cell)
            if i in align_left:
                cells.append(cell_str.ljust(col_widths[i]))
            else:
                cells.append(cell_str.rjust(col_widths[i]))
        return "  ".join(cells)

    print(fmt_row(header))
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    main()
