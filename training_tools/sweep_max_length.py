#!/usr/bin/env python3
"""Run repeated transformer training with different max_length values."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train_advanced.py multiple times with different --max-length values "
            "and collect a single comparison table."
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Training CSV path.")
    parser.add_argument(
        "-o",
        "--output-root",
        required=True,
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--max-lengths",
        default="64,128,192,256",
        help="Comma-separated max_length values. Default: 64,128,192,256",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="call_purpose",
        help="Target column. Default: call_purpose",
    )
    parser.add_argument(
        "-g",
        "--groups",
        default="transformers",
        help="Model groups passed to train_advanced.py. Default: transformers",
    )
    parser.add_argument("--sep", default=";", help="CSV separator. Default: ';'")
    parser.add_argument(
        "--eval-input",
        default=None,
        help="Optional fixed eval CSV passed through to train_advanced.py",
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=["head", "head_tail", "middle_cut"],
        default="head",
        help="Truncation strategy. Default: head",
    )
    parser.add_argument(
        "--rubert-model",
        default="cointegrated/rubert-tiny2",
        help="RuBERT model id passed to train_advanced.py",
    )
    parser.add_argument(
        "--xlmr-model",
        default="xlm-roberta-base",
        help="XLM-R model id passed to train_advanced.py",
    )
    parser.add_argument(
        "--extra-models",
        default=None,
        help="Extra HF models, same format as train_advanced.py",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Epochs. Default: 3")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size. Default: 16"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="Grad accum. Default: 1"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--freeze-layers", type=int, default=0, help="Freeze first N layers."
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="Label smoothing."
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        help="Early stopping patience. Default: 0",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=["f1", "loss"],
        default="f1",
        help="Early stopping metric. Default: f1",
    )
    parser.add_argument("--cv", type=int, default=0, help="CV folds. Default: 0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def parse_lengths(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("At least one max_length must be provided.")
    return values


def build_command(args: argparse.Namespace, train_script: Path, length: int, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(train_script),
        "-i",
        str(Path(args.input).resolve()),
        "-o",
        str(run_dir),
        "-g",
        args.groups,
        "-t",
        args.target,
        "--sep",
        args.sep,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--freeze-layers",
        str(args.freeze_layers),
        "--label-smoothing",
        str(args.label_smoothing),
        "--early-stopping",
        str(args.early_stopping),
        "--early-stopping-metric",
        args.early_stopping_metric,
        "--max-length",
        str(length),
        "--truncation-strategy",
        args.truncation_strategy,
        "--cv",
        str(args.cv),
        "--seed",
        str(args.seed),
        "--rubert-model",
        args.rubert_model,
        "--xlmr-model",
        args.xlmr_model,
    ]
    if args.eval_input:
        cmd.extend(["--eval-input", str(Path(args.eval_input).resolve())])
    if args.extra_models:
        cmd.extend(["--extra-models", args.extra_models])
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")
    if args.compile:
        cmd.append("--compile")
    return cmd


def load_run_comparison(run_dir: Path, target: str) -> pd.DataFrame:
    comparison_path = run_dir / target / "comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"comparison.csv not found: {comparison_path}")
    df = pd.read_csv(comparison_path)
    df["run_dir"] = str(run_dir)
    return df


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.max_lengths)

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "train_advanced.py"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []

    print("Max-length sweep")
    print(f"Train script : {train_script}")
    print(f"Input CSV    : {Path(args.input).resolve()}")
    print(f"Output root  : {output_root}")
    print(f"Lengths      : {lengths}")
    print(f"Target       : {args.target}")
    print(f"Groups       : {args.groups}")

    for length in lengths:
        run_dir = output_root / f"len_{length}"
        cmd = build_command(args, train_script, length, run_dir)
        print("\n" + "=" * 80)
        print(f"Running max_length={length}")
        print(" ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)
            run_df = load_run_comparison(run_dir, args.target)
            run_df["max_length"] = length
            all_rows.append(run_df)

    if args.dry_run:
        return 0

    if not all_rows:
        raise RuntimeError("No runs were collected.")

    combined = pd.concat(all_rows, ignore_index=True)
    comparison_out = output_root / "sweep_comparison.csv"
    combined.to_csv(comparison_out, index=False, encoding="utf-8")

    sort_cols = [c for c in ("f1_weighted", "accuracy", "precision_weighted", "recall_weighted") if c in combined.columns]
    ascending = [False] * len(sort_cols)
    best = (
        combined.sort_values(sort_cols + ["max_length"], ascending=ascending + [True])
        .groupby("model", as_index=False)
        .first()
    )
    best_out = output_root / "best_by_model.csv"
    best.to_csv(best_out, index=False, encoding="utf-8")

    print("\nSaved:")
    print(f"  sweep comparison : {comparison_out}")
    print(f"  best by model    : {best_out}")

    display_cols = [c for c in [
        "model",
        "max_length",
        "accuracy",
        "f1_weighted",
        "infer_ms_per_sample",
        "run_dir",
    ] if c in best.columns]
    print("\nBest by model:")
    print(best[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
