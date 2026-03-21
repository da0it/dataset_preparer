#!/usr/bin/env python3
"""
Unified CLI for dataset preparation and cleanup steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dataset_tools.dataset_ops import (
    run_clean_noise,
    run_filter_and_clean,
    run_normalize_labels,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dataset operations CLI for preparation and cleanup."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser(
        "normalize-labels",
        help="Normalize label columns (call_purpose, priority, assigned_group).",
    )
    normalize.add_argument("--input", "-i", required=True, help="Input CSV")
    normalize.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV (default: <input>_norm.csv)",
    )
    normalize.add_argument(
        "--sep",
        default=",",
        help="CSV separator (default: comma). Use ';' if needed.",
    )

    filter_cmd = subparsers.add_parser(
        "filter-ready",
        help="Drop rows without call_purpose and normalize filename.",
    )
    filter_cmd.add_argument("--input", "-i", required=True, help="Input CSV")
    filter_cmd.add_argument("--output", "-o", required=True, help="Output CSV")
    filter_cmd.add_argument(
        "--sep",
        default=";",
        help="CSV separator (default: ;)",
    )
    filter_cmd.add_argument(
        "--keep-all-cols",
        action="store_true",
        help="Keep all columns instead of reducing to training columns.",
    )

    noise = subparsers.add_parser(
        "clean-noise",
        help="Remove call-center noise phrases from text column.",
    )
    noise.add_argument("--input", "-i", required=True, help="Input CSV")
    noise.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV (default: <input>_no_noise.csv)",
    )
    noise.add_argument(
        "--min-tokens",
        type=int,
        default=3,
        help="Minimum token count after cleaning (default: 3)",
    )
    noise.add_argument(
        "--source-column",
        default="text",
        help="Column to clean (default: text)",
    )
    noise.add_argument(
        "--sep",
        default=",",
        help="CSV field separator (default: comma)",
    )
    noise.add_argument(
        "--dry-run",
        action="store_true",
        help="Print 5 random before/after examples and exit",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "normalize-labels":
            input_path = Path(args.input).resolve()
            output_path = (
                Path(args.output).resolve()
                if args.output
                else input_path.with_stem(input_path.stem + "_norm")
            )
            run_normalize_labels(input_path, output_path, args.sep)
            return 0

        if args.command == "filter-ready":
            run_filter_and_clean(
                Path(args.input).resolve(),
                Path(args.output).resolve(),
                args.sep,
                args.keep_all_cols,
            )
            return 0

        if args.command == "clean-noise":
            input_path = Path(args.input).resolve()
            output_path = (
                Path(args.output).resolve()
                if args.output
                else input_path.with_stem(input_path.stem + "_no_noise")
            )
            run_clean_noise(
                input_path,
                output_path,
                args.min_tokens,
                args.source_column,
                args.sep,
                args.dry_run,
            )
            return 0

    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
