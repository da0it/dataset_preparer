#!/usr/bin/env python3
"""Merge multiple train_advanced comparison.csv / run_log.json outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple comparison.csv files into one master comparison."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to comparison.csv files or experiment target directories containing comparison.csv.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="merged_comparison",
        help="Directory for merged comparison.csv and run_log.json (default: merged_comparison).",
    )
    parser.add_argument(
        "--sort-by",
        default="f1_weighted",
        help="Column used to sort merged comparison.csv (default: f1_weighted).",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order (default: enabled for metric-like columns).",
    )
    parser.add_argument(
        "--dedupe-models",
        action="store_true",
        help="Keep only the best row per model according to --sort-by.",
    )
    return parser.parse_args()


def resolve_target_dir(value: str) -> Path:
    path = Path(value).resolve()
    if path.is_file():
        if path.name != "comparison.csv":
            raise FileNotFoundError(f"Expected comparison.csv, got file: {path}")
        return path.parent

    comparison_path = path / "comparison.csv"
    if comparison_path.exists():
        return path

    raise FileNotFoundError(
        f"Could not find comparison.csv in {path}. "
        "Pass either comparison.csv itself or a target directory containing it."
    )


def load_run_log(target_dir: Path) -> list[dict]:
    run_log = target_dir / "run_log.json"
    if not run_log.exists():
        return []
    with open(run_log, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {run_log}, got {type(payload).__name__}")
    return payload


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_dirs = [resolve_target_dir(value) for value in args.inputs]
    comparison_frames: list[pd.DataFrame] = []
    merged_run_log: list[dict] = []

    for target_dir in target_dirs:
        comparison_path = target_dir / "comparison.csv"
        df = pd.read_csv(comparison_path, dtype={"notes": str}).fillna("")
        df["source_dir"] = str(target_dir)
        comparison_frames.append(df)

        for row in load_run_log(target_dir):
            row = dict(row)
            row["source_dir"] = str(target_dir)
            merged_run_log.append(row)

    if not comparison_frames:
        raise ValueError("No comparison.csv files loaded.")

    merged_df = pd.concat(comparison_frames, ignore_index=True)

    sort_by = args.sort_by
    descending = args.descending or sort_by.lower() in {
        "f1",
        "f1_weighted",
        "accuracy",
        "precision",
        "precision_weighted",
        "recall",
        "recall_weighted",
    }
    if sort_by in merged_df.columns:
        merged_df = merged_df.sort_values(sort_by, ascending=not descending)

    if args.dedupe_models and "model" in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=["model"], keep="first")

    comparison_out = output_dir / "comparison.csv"
    merged_df.to_csv(comparison_out, index=False, encoding="utf-8")

    if merged_run_log:
        if sort_by:
            def _sort_key(item: dict):
                value = item.get(sort_by)
                if value is None:
                    return float("-inf") if descending else float("inf")
                return value

            merged_run_log.sort(key=_sort_key, reverse=descending)
        run_log_out = output_dir / "run_log.json"
        with open(run_log_out, "w", encoding="utf-8") as f:
            json.dump(merged_run_log, f, ensure_ascii=False, indent=2)

    print(f"Merged comparison: {comparison_out}")
    if merged_run_log:
        print(f"Merged run log   : {output_dir / 'run_log.json'}")
    print(f"Rows             : {len(merged_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
