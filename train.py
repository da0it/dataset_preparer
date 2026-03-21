#!/usr/bin/env python3
"""
Compatibility wrapper for the legacy classical training entrypoint.

Prefer:
    python train_advanced.py -g legacy-baseline ...
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Legacy wrapper around train_advanced.py for classical models."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Cleaned CSV file")
    parser.add_argument("--target", "-t", default="call_purpose",
                        choices=["call_purpose", "priority", "assig_group", "all"],
                        help="Column to predict (default: call_purpose)")
    parser.add_argument("--output", "-o", default="models",
                        help="Output directory (default: ./models)")
    parser.add_argument("--sep", default=";",
                        help="CSV separator (default: ;)")
    parser.add_argument("--dataset-variant", choices=["multiclass", "binary_spam"],
                        default="multiclass",
                        help="How to interpret the main --input (default: multiclass)")
    parser.add_argument("--binary-input", default=None,
                        help="Optional second CSV for binary spam/non-spam training")
    args = parser.parse_args(argv)

    forwarded = [
        "--input", args.input,
        "--target", args.target,
        "--output", args.output,
        "--sep", args.sep,
        "--groups", "legacy-baseline",
        "--dataset-variant", args.dataset_variant,
    ]
    if args.binary_input:
        forwarded.extend(["--binary-input", args.binary_input])
    from training_tools.train_advanced import main as train_advanced_main
    return train_advanced_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
