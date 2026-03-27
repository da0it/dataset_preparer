#!/usr/bin/env python3
"""Build a comparison chart from CSV/XLSX/Numbers tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a comparison chart from a table file."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input table: .csv, .tsv, .xlsx, .xls, or .numbers",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="comparison_plot.png",
        help="Output image path. Default: comparison_plot.png",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional sheet name for .xlsx/.numbers",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Optional table name for .numbers",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="CSV separator. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Column to use as model label. Default: first non-numeric column.",
    )
    parser.add_argument(
        "--value-cols",
        default=None,
        help="Comma-separated metric columns. Default: all numeric columns except label-col.",
    )
    parser.add_argument(
        "--title",
        default="Model Comparison",
        help="Chart title.",
    )
    parser.add_argument(
        "--sort-by",
        default=None,
        help="Optional metric column for sorting rows descending.",
    )
    parser.add_argument(
        "--kind",
        choices=["bar", "barh"],
        default="barh",
        help="Chart type. Default: barh",
    )
    parser.add_argument(
        "--figsize",
        default=None,
        help="Optional figure size as WIDTH,HEIGHT. Example: 14,8",
    )
    return parser.parse_args()


def read_numbers_table(path: Path, sheet_name: str | None, table_name: str | None) -> pd.DataFrame:
    try:
        from numbers_parser import Document
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "To read .numbers files install numbers-parser: "
            "pip install numbers-parser"
        ) from exc

    doc = Document(str(path))
    sheets = doc.sheets
    if not sheets:
        raise ValueError(f"No sheets found in {path}")

    if sheet_name:
        matches = [sheet for sheet in sheets if sheet.name == sheet_name]
        if not matches:
            raise ValueError(f"Sheet '{sheet_name}' not found in {path}")
        sheet = matches[0]
    else:
        sheet = sheets[0]

    tables = sheet.tables
    if not tables:
        raise ValueError(f"No tables found in sheet '{sheet.name}'")

    if table_name:
        matches = [table for table in tables if table.name == table_name]
        if not matches:
            raise ValueError(
                f"Table '{table_name}' not found in sheet '{sheet.name}'"
            )
        table = matches[0]
    else:
        table = tables[0]

    rows = table.rows(values_only=True)
    rows = [list(row) for row in rows]
    if not rows:
        raise ValueError(f"Table '{table.name}' is empty")

    header = [str(col).strip() if col is not None else f"col_{idx}" for idx, col in enumerate(rows[0])]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


def read_table(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".numbers":
        return read_numbers_table(path, args.sheet, args.table)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=args.sheet or 0)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        if args.sep:
            return pd.read_csv(path, sep=args.sep)
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(f"Unsupported file type: {suffix}")


def parse_figsize(raw: str | None, rows: int, cols: int) -> tuple[float, float]:
    if raw:
        width, height = raw.split(",", 1)
        return float(width), float(height)
    return max(10, 2.5 * cols + 4), max(6, 0.45 * rows + 2)


def choose_label_col(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Label column '{explicit}' not found. Available: {list(df.columns)}")
        return explicit
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return col
    return df.columns[0]


def choose_value_cols(df: pd.DataFrame, label_col: str, explicit: str | None) -> list[str]:
    if explicit:
        cols = [col.strip() for col in explicit.split(",") if col.strip()]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Value columns not found: {missing}. Available: {list(df.columns)}")
        return cols

    value_cols = []
    for col in df.columns:
        if col == label_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            value_cols.append(col)
    if not value_cols:
        raise ValueError("No numeric columns found for plotting.")
    return value_cols


def prepare_frame(df: pd.DataFrame, label_col: str, value_cols: list[str], sort_by: str | None) -> pd.DataFrame:
    out = df[[label_col] + value_cols].copy()
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[label_col])
    out = out.dropna(how="all", subset=value_cols)
    out[label_col] = out[label_col].astype(str)
    if sort_by:
        if sort_by not in value_cols:
            raise ValueError(f"sort-by column '{sort_by}' must be one of {value_cols}")
        out = out.sort_values(sort_by, ascending=False)
    return out.reset_index(drop=True)


def plot_grouped_bars(df: pd.DataFrame, label_col: str, value_cols: list[str], output: Path, title: str, kind: str, figsize: tuple[float, float]) -> None:
    labels = df[label_col].tolist()
    values = df[value_cols].to_numpy(dtype=float)
    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_rows)
    total_width = 0.8
    bar_width = total_width / max(n_cols, 1)

    for idx, col in enumerate(value_cols):
        offset = (idx - (n_cols - 1) / 2.0) * bar_width
        if kind == "bar":
            bars = ax.bar(x + offset, values[:, idx], width=bar_width, label=col)
            for bar, val in zip(bars, values[:, idx]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8)
        else:
            bars = ax.barh(x + offset, values[:, idx], height=bar_width, label=col)
            for bar, val in zip(bars, values[:, idx]):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                        ha="left", va="center", fontsize=8)

    if kind == "bar":
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Value")
    else:
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Value")
        ax.invert_yaxis()

    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x" if kind == "barh" else "y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    df = read_table(input_path, args)
    label_col = choose_label_col(df, args.label_col)
    value_cols = choose_value_cols(df, label_col, args.value_cols)
    df = prepare_frame(df, label_col, value_cols, args.sort_by)

    figsize = parse_figsize(args.figsize, len(df), len(value_cols))
    plot_grouped_bars(df, label_col, value_cols, output_path, args.title, args.kind, figsize)

    print(f"Input   : {input_path}")
    print(f"Rows    : {len(df)}")
    print(f"Label   : {label_col}")
    print(f"Metrics : {', '.join(value_cols)}")
    print(f"Output  : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
