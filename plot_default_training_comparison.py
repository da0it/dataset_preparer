#!/usr/bin/env python3
"""Plot the same comparison chart layout used by train_advanced.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


COLORS = {
    "baseline": "#4C72B0",
    "embeddings": "#55A868",
    "transformers": "#C44E52",
    "llm": "#8172B2",
    "legacy_baseline": "#DD8452",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate the default comparison chart from train_advanced.py."
    )
    parser.add_argument("-i", "--input", required=True, help="Input CSV/TSV/XLSX file.")
    parser.add_argument("-o", "--output", default="comparison_default.png", help="Output PNG path.")
    parser.add_argument("--sep", default=None, help="CSV separator. Auto-detect if omitted.")
    parser.add_argument("--sheet", default=None, help="Sheet name for XLSX.")
    parser.add_argument("--title", default="Сравнение моделей", help="Figure title.")
    parser.add_argument("--target", default="", help="Optional target suffix for title.")
    parser.add_argument("--model-col", default="model", help="Column with model names.")
    parser.add_argument("--group-col", default="group", help="Column with model groups.")
    parser.add_argument("--f1-col", default="f1_weighted", help="Column with weighted F1.")
    parser.add_argument(
        "--speed-col",
        default="infer_ms_per_sample",
        help="Column with inference time in ms per sample.",
    )
    parser.add_argument(
        "--sort-by",
        default="f1_weighted",
        help="Sort rows descending by this column. Default: f1_weighted",
    )
    return parser.parse_args()


def read_table(path: Path, sep: str | None, sheet: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        if sep:
            return pd.read_csv(path, sep=sep)
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(
        f"Unsupported input type: {suffix}. "
        "Export the Numbers table to CSV/XLSX first."
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    df = read_table(input_path, args.sep, args.sheet).copy()
    required = [args.model_col, args.group_col, args.f1_col, args.speed_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    df = df[[args.model_col, args.group_col, args.f1_col, args.speed_col]].copy()
    df[args.model_col] = df[args.model_col].astype(str)
    df[args.group_col] = df[args.group_col].astype(str)
    df[args.f1_col] = pd.to_numeric(df[args.f1_col], errors="coerce")
    df[args.speed_col] = pd.to_numeric(df[args.speed_col], errors="coerce")
    df = df.dropna(subset=[args.model_col, args.group_col, args.f1_col, args.speed_col])

    sort_col = args.sort_by
    if sort_col not in df.columns:
        raise ValueError(f"sort-by column '{sort_col}' not found in {list(df.columns)}")
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(df) * 0.5 + 2)))
    title = args.title if not args.target else f"{args.title} — {args.target}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    bar_colors = [COLORS.get(group, "#777777") for group in df[args.group_col]]
    bars = ax.barh(df[args.model_col], df[args.f1_col], color=bar_colors, height=0.6)
    ax.set_xlabel("F1 (weighted)")
    ax.set_title("F1-score по моделям")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, df[args.f1_col]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.invert_yaxis()

    ax2 = axes[1]
    for _, row in df.iterrows():
        color = COLORS.get(row[args.group_col], "#777777")
        ax2.scatter(row[args.speed_col], row[args.f1_col], c=color, s=90, zorder=3)
        ax2.annotate(
            row[args.model_col],
            (row[args.speed_col], row[args.f1_col]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax2.set_xlabel("Время инференса (мс / образец)")
    ax2.set_ylabel("F1 (weighted)")
    ax2.set_title("Качество vs Скорость")
    ax2.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor=color, label=group)
        for group, color in COLORS.items()
        if group in set(df[args.group_col])
    ]
    if legend_elements:
        ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=110)
    plt.close()

    print(f"Input   : {input_path}")
    print(f"Rows    : {len(df)}")
    print(f"Output  : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
