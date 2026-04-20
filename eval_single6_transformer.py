#!/usr/bin/env python3
"""Evaluate one saved 6-class transformer model on an external eval CSV."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification

from training_tools.tokenization_utils import (
    encode_text_batch,
    load_hf_tokenizer,
    resolve_inference_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved 6-class transformer on an external eval CSV."
    )
    parser.add_argument("--model-path", required=True,
                        help="Saved HuggingFace sequence-classification directory.")
    parser.add_argument("--eval-input", required=True,
                        help="External labeled eval CSV.")
    parser.add_argument("--target", default="call_purpose",
                        help="Target column name. Default: call_purpose")
    parser.add_argument("--sep", default=";",
                        help="CSV separator. Default: ;")
    parser.add_argument("--text-col", default="text",
                        help="Text column name. Default: text")
    parser.add_argument("--spam-label", default="spam",
                        help="Spam class label. Default: spam")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size. Default: 16")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Fallback max sequence length. Default: 512")
    parser.add_argument("--truncation-strategy",
                        choices=["head", "head_tail", "middle_cut"],
                        default="head",
                        help="Fallback truncation strategy.")
    parser.add_argument("-o", "--output", default=None,
                        help="Optional CSV path for per-row predictions.")
    return parser.parse_args()


def load_eval(path: Path, sep: str, text_col: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
    missing = [col for col in (text_col, target) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available columns: {list(df.columns)}")
    df[text_col] = df[text_col].astype(str).str.strip()
    df[target] = df[target].astype(str).str.strip()
    df = df[(df[text_col] != "") & (df[target] != "")].copy()
    if df.empty:
        raise ValueError("No usable eval rows after filtering empty text/target.")
    return df.reset_index(drop=True)


def predict_transformer(
    model_dir: Path,
    texts: list[str],
    batch_size: int,
    fallback_max_length: int,
    fallback_truncation_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    le_path = model_dir / "label_encoder.joblib"
    if not le_path.exists():
        raise FileNotFoundError(f"label_encoder.joblib not found in {model_dir}")

    label_encoder = joblib.load(le_path)
    labels = np.asarray(label_encoder.classes_)
    tokenizer = load_hf_tokenizer(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    max_length, truncation_strategy = resolve_inference_config(
        model_dir,
        fallback_max_length,
        fallback_truncation_strategy,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    probas = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = encode_text_batch(
            tokenizer,
            batch,
            max_length=max_length,
            truncation_strategy=truncation_strategy,
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        probas.append(probs)

    proba = np.vstack(probas)
    pred_idx = proba.argmax(axis=1)
    pred = labels[pred_idx]
    confidence = proba.max(axis=1)
    return pred, confidence, labels


def metric_line(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return (
        f"{name}: samples={len(y_true)} "
        f"accuracy={accuracy_score(y_true, y_pred):.4f} "
        f"macro_f1={f1_score(y_true, y_pred, average='macro', zero_division=0):.4f} "
        f"weighted_f1={f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}"
    )


def collapse_spam(labels: np.ndarray, spam_label: str) -> np.ndarray:
    return np.where(labels == spam_label, spam_label, "non_spam")


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_path).resolve()
    eval_path = Path(args.eval_input).resolve()

    df = load_eval(eval_path, args.sep, args.text_col, args.target)

    started = time.perf_counter()
    y_pred, confidence, model_labels = predict_transformer(
        model_dir,
        df[args.text_col].astype(str).tolist(),
        batch_size=args.batch_size,
        fallback_max_length=args.max_length,
        fallback_truncation_strategy=args.truncation_strategy,
    )
    duration = time.perf_counter() - started

    y_true = df[args.target].astype(str).to_numpy()
    y_true_binary = collapse_spam(y_true, args.spam_label)
    y_pred_binary = collapse_spam(y_pred, args.spam_label)
    nonspam_mask = y_true != args.spam_label

    print(f"model={model_dir}")
    print(f"eval_input={eval_path}")
    print(f"model_labels={list(model_labels)}")
    print(f"rows_total={len(df)}, duration_sec={duration:.2f}")
    print("== Single 6-class model / all ==")
    print(metric_line("multiclass", y_true, y_pred))
    print(metric_line("binary spam", y_true_binary, y_pred_binary))
    if int(nonspam_mask.sum()) > 0:
        print(metric_line("nonspam intents", y_true[nonspam_mask], y_pred[nonspam_mask]))
    else:
        print("nonspam intents: samples=0")

    if args.output:
        out = df.copy()
        out["y_true"] = y_true
        out["y_pred"] = y_pred
        out["confidence"] = confidence
        out["y_true_binary"] = y_true_binary
        out["y_pred_binary"] = y_pred_binary
        out["is_error"] = out["y_true"] != out["y_pred"]
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, sep=args.sep, index=False, encoding="utf-8")
        print(f"saved_predictions={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
