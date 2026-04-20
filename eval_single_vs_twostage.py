#!/usr/bin/env python3
"""Compare one 6-class transformer and a two-stage transformer pipeline on eval CSV."""

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
        description="Evaluate a single 6-class model and a two-stage pipeline on external eval."
    )
    parser.add_argument("--eval-input", required=True, help="External labeled eval CSV.")
    parser.add_argument("--sep", default=";", help="CSV separator. Default: ;")
    parser.add_argument("--target", default="call_purpose", help="Target column. Default: call_purpose")
    parser.add_argument("--text-col", default="text", help="Text column. Default: text")
    parser.add_argument("--single-model-path", required=True,
                        help="Saved 6-class transformer directory.")
    parser.add_argument("--spam-model-path", required=True,
                        help="Saved binary spam/non_spam transformer directory.")
    parser.add_argument("--intent-model-path", required=True,
                        help="Saved non-spam intent transformer directory.")
    parser.add_argument("--spam-label", default="spam", help="Spam label. Default: spam")
    parser.add_argument("--non-spam-label", default="non_spam",
                        help="Non-spam label used by binary model. Default: non_spam")
    parser.add_argument("--auto-threshold", type=float, default=0.80,
                        help="Confidence threshold for auto ticket mode. Default: 0.80")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size. Default: 16")
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


def build_two_stage_predictions(
    spam_pred: np.ndarray,
    spam_conf: np.ndarray,
    intent_pred: np.ndarray,
    intent_conf: np.ndarray,
    spam_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    final_pred = np.where(spam_pred == spam_label, spam_label, intent_pred)
    final_conf = np.where(spam_pred == spam_label, spam_conf, np.minimum(spam_conf, intent_conf))
    return final_pred, final_conf


def print_blocks(title: str, y_true: np.ndarray, y_pred: np.ndarray, spam_label: str) -> None:
    print(f"== {title} ==")
    print(metric_line("multiclass", y_true, y_pred))
    print(metric_line("binary spam", collapse_spam(y_true, spam_label), collapse_spam(y_pred, spam_label)))
    nonspam_mask = y_true != spam_label
    if int(nonspam_mask.sum()) > 0:
        print(metric_line("nonspam intents", y_true[nonspam_mask], y_pred[nonspam_mask]))
    else:
        print("nonspam intents: samples=0")


def main() -> int:
    args = parse_args()
    df = load_eval(Path(args.eval_input).resolve(), args.sep, args.text_col, args.target)
    texts = df[args.text_col].astype(str).tolist()
    y_true = df[args.target].astype(str).to_numpy()

    started = time.perf_counter()
    single_pred, single_conf, single_labels = predict_transformer(
        Path(args.single_model_path).resolve(),
        texts,
        args.batch_size,
        args.max_length,
        args.truncation_strategy,
    )
    spam_pred, spam_conf, spam_labels = predict_transformer(
        Path(args.spam_model_path).resolve(),
        texts,
        args.batch_size,
        args.max_length,
        args.truncation_strategy,
    )
    intent_pred, intent_conf, intent_labels = predict_transformer(
        Path(args.intent_model_path).resolve(),
        texts,
        args.batch_size,
        args.max_length,
        args.truncation_strategy,
    )
    two_stage_pred, two_stage_conf = build_two_stage_predictions(
        spam_pred,
        spam_conf,
        intent_pred,
        intent_conf,
        args.spam_label,
    )
    duration = time.perf_counter() - started

    print(f"rows_total={len(df)}, duration_sec={duration:.2f}")
    print(f"single_labels={list(single_labels)}")
    print(f"spam_labels={list(spam_labels)}")
    print(f"intent_labels={list(intent_labels)}")
    print_blocks("Single 6-class model / all", y_true, single_pred, args.spam_label)
    print_blocks("Two-stage pipeline / all", y_true, two_stage_pred, args.spam_label)

    auto_mask = (two_stage_pred != args.spam_label) & (two_stage_conf >= args.auto_threshold)
    print(
        "Two-stage operational: "
        f"auto_ticket_allowed={int(auto_mask.sum())}, "
        f"review_required={int((~auto_mask).sum())}"
    )

    if args.output:
        out = df.copy()
        out["y_true"] = y_true
        out["single_pred"] = single_pred
        out["single_conf"] = single_conf
        out["spam_stage_pred"] = spam_pred
        out["spam_stage_conf"] = spam_conf
        out["intent_stage_pred"] = intent_pred
        out["intent_stage_conf"] = intent_conf
        out["two_stage_pred"] = two_stage_pred
        out["two_stage_conf"] = two_stage_conf
        out["two_stage_auto_allowed"] = auto_mask
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, sep=args.sep, index=False, encoding="utf-8")
        print(f"saved_predictions={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
