#!/usr/bin/env python3
"""Remap obsolete eval labels in a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MAPPING = {
    "escalation": "consulting",
    "marketing": "cooperation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap obsolete labels in eval CSV."
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV path.")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path.")
    parser.add_argument(
        "--column",
        default="call_purpose",
        help="Target column to remap (default: call_purpose).",
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="CSV separator (default: ;).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise SystemExit(f"Error: input file not found: {input_path}")

    df = pd.read_csv(input_path, sep=args.sep, dtype=str).fillna("")
    if args.column not in df.columns:
        raise SystemExit(
            f"Error: column '{args.column}' not found. Available: {df.columns.tolist()}"
        )

    before = df[args.column].value_counts().sort_index()
    df[args.column] = df[args.column].replace(DEFAULT_MAPPING)
    after = df[args.column].value_counts().sort_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=args.sep, index=False, encoding="utf-8")

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Column: {args.column}")
    print("\nMapping:")
    for src, dst in DEFAULT_MAPPING.items():
        replaced = int((before.get(src, 0)))
        print(f"  {src} -> {dst}  |  replaced: {replaced}")

    print("\nLabel counts before:")
    print(before.to_string())
    print("\nLabel counts after:")
    print(after.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
