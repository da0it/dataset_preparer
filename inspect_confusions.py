#!/usr/bin/env python3
"""Inspect model misclassifications on the reconstructed hold-out split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


MIN_SAMPLES_PER_CLASS = 5
TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export examples where a saved model confused classes."
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to .joblib model or transformer model directory.")
    parser.add_argument("--input", "-i", required=True,
                        help="Dataset CSV used for training/evaluation.")
    parser.add_argument("--target", required=True,
                        help="Target column name.")
    parser.add_argument("--sep", default=";",
                        help="CSV separator (default: ;)")
    parser.add_argument("--output", "-o", default="confusions.csv",
                        help="Output CSV path.")
    parser.add_argument("--true-label", default=None,
                        help="Optional filter: only rows with this true label.")
    parser.add_argument("--pred-label", default=None,
                        help="Optional filter: only rows with this predicted label.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Fallback max_length for transformer inference.")
    parser.add_argument("--truncation-strategy", default="head",
                        choices=["head", "head_tail", "middle_cut"],
                        help="Fallback truncation strategy for transformer inference.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for transformer inference.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reconstructing the 80/20 split.")
    return parser.parse_args()


def load_eval_frame(csv_path: Path, target: str, sep: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path, sep=sep, dtype=str).fillna("")
    if "text" not in df.columns or target not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and '{target}'. Available: {list(df.columns)}"
        )

    df = df[df["text"].astype(str).str.strip() != ""].copy()
    df[target] = df[target].astype(str).str.strip()
    df = df[df[target] != ""].copy()

    counts = df[target].value_counts()
    valid = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    df = df[df[target].isin(valid)].copy()
    if df.empty:
        raise ValueError("No rows left after filtering empty/rare classes.")

    X = df["text"].reset_index(drop=True)
    y = df[target].reset_index(drop=True)
    return df.reset_index(drop=True), X, y


def reconstruct_test_split(df: pd.DataFrame, X: pd.Series, y: pd.Series, seed: int) -> pd.DataFrame:
    idx = np.arange(len(df))
    _, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    return df.iloc[test_idx].reset_index(drop=True)


def load_transformer_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(str(model_dir))
    except TypeError as exc:
        if "BertPreTokenizer" not in str(exc) and "pre_tokenizer" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)


def predict_transformer(model_dir: Path, texts: list[str], batch_size: int,
                        fallback_max_length: int, fallback_truncation_strategy: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoModelForSequenceClassification
    from training_tools.tokenization_utils import encode_text_batch, resolve_inference_config

    le = joblib.load(model_dir / "label_encoder.joblib")
    tokenizer = load_transformer_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    max_length, truncation_strategy = resolve_inference_config(
        model_dir, fallback_max_length, fallback_truncation_strategy
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = encode_text_batch(
            tokenizer,
            batch,
            max_length=max_length,
            truncation_strategy=truncation_strategy,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    proba = np.vstack(all_probs)
    pred_idx = proba.argmax(axis=1)
    y_pred = le.inverse_transform(pred_idx)
    confidence = proba.max(axis=1)
    return y_pred, confidence


def predict_joblib(model_path: Path, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    model = joblib.load(model_path)
    y_pred = model.predict(texts)

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        confidence = proba.max(axis=1)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        scores = np.asarray(scores)
        if scores.ndim == 1:
            confidence = np.abs(scores)
        else:
            confidence = scores.max(axis=1)
    else:
        confidence = np.full(len(texts), np.nan)

    return np.asarray(y_pred), np.asarray(confidence)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    df, X, y = load_eval_frame(input_path, args.target, args.sep)
    test_df = reconstruct_test_split(df, X, y, args.seed)

    texts = test_df["text"].astype(str).tolist()
    y_true = test_df[args.target].astype(str).to_numpy()

    if model_path.is_dir():
        if not (model_path / "label_encoder.joblib").exists():
            raise ValueError(f"{model_path} looks like a directory, but label_encoder.joblib is missing.")
        y_pred, confidence = predict_transformer(
            model_path,
            texts,
            batch_size=args.batch_size,
            fallback_max_length=args.max_length,
            fallback_truncation_strategy=args.truncation_strategy,
        )
    else:
        y_pred, confidence = predict_joblib(model_path, texts)

    out = test_df.copy()
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["confidence"] = confidence
    out["is_error"] = out["y_true"] != out["y_pred"]

    errors = out[out["is_error"]].copy()
    if args.true_label is not None:
        errors = errors[errors["y_true"] == args.true_label].copy()
    if args.pred_label is not None:
        errors = errors[errors["y_pred"] == args.pred_label].copy()

    priority_cols = ["filename", "text", args.target, "y_true", "y_pred", "confidence", "is_error"]
    ordered_cols = [c for c in priority_cols if c in errors.columns] + [c for c in errors.columns if c not in priority_cols]
    errors = errors[ordered_cols]
    errors.to_csv(output_path, sep=args.sep, index=False, encoding="utf-8")

    print(f"Model      : {model_path}")
    print(f"Input      : {input_path}")
    print(f"Test rows   : {len(test_df)}")
    print(f"Errors      : {len(out[out['is_error']])}")
    if args.true_label or args.pred_label:
        print(f"Filtered    : {len(errors)}")
    print(f"Output      : {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
