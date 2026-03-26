#!/usr/bin/env python3
"""Analyze token length distribution for text datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token length distribution for a CSV text dataset."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="CSV separator. Default: ';'",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Text column name. Default: text",
    )
    parser.add_argument(
        "--model",
        default="DeepPavlov/rubert-base-cased",
        help="Tokenizer model name or local path. Default: DeepPavlov/rubert-base-cased",
    )
    parser.add_argument(
        "--thresholds",
        default="128,256,512",
        help="Comma-separated token thresholds. Default: 128,256,512",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to save per-row token lengths CSV.",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[int]:
    thresholds = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(int(item))
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")
    return thresholds


def load_tokenizer(model_ref: str):
    try:
        return AutoTokenizer.from_pretrained(model_ref)
    except TypeError as exc:
        # Some transformers/tokenizers combinations fail on fast BERT tokenizers.
        if "BertPreTokenizer" not in str(exc) and "pre_tokenizer" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(model_ref, use_fast=False)


def summarize_lengths(lengths: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)

    df = pd.read_csv(args.input, sep=args.sep, dtype=str).fillna("")
    if args.text_col not in df.columns:
        raise ValueError(
            f"Column '{args.text_col}' not found. Available columns: {list(df.columns)}"
        )

    tokenizer = load_tokenizer(args.model)
    texts = df[args.text_col].astype(str).tolist()
    lengths: list[int] = []

    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths.append(len(encoded["input_ids"]))

    length_array = np.array(lengths, dtype=np.int32)
    summary = summarize_lengths(length_array)

    print("\nToken length summary")
    print(f"Input file      : {args.input}")
    print(f"Tokenizer       : {args.model}")
    print(f"Text column     : {args.text_col}")
    print(f"Rows            : {len(length_array)}")
    print(f"Mean            : {summary['mean']:.2f}")
    print(f"Median          : {summary['median']:.2f}")
    print(f"P90             : {summary['p90']:.2f}")
    print(f"P95             : {summary['p95']:.2f}")
    print(f"P99             : {summary['p99']:.2f}")
    print(f"Max             : {summary['max']}")

    print("\nThreshold coverage")
    total = len(length_array)
    for threshold in thresholds:
        count = int((length_array > threshold).sum())
        share = count / total if total else 0.0
        print(f">{threshold:4d} tokens : {count:6d} rows ({share:.2%})")

    if args.output:
        out_df = df.copy()
        out_df["token_length"] = lengths
        output_path = Path(args.output)
        out_df.to_csv(output_path, sep=args.sep, index=False, encoding="utf-8")
        print(f"\nSaved per-row lengths to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
