#!/usr/bin/env python3
"""Compare saved sklearn, transformer, and ensemble models on a labeled eval CSV."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from training_tools.ensemble_experiments import hard_vote, max_vote, soft_vote
from training_tools.tokenization_utils import encode_text_batch, resolve_inference_config


@dataclass
class ModelSpec:
    name: str
    path: Path
    kind: str  # sklearn | transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved models on an external eval CSV with production-oriented metrics."
    )
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Root directory containing saved model artifacts.",
    )
    parser.add_argument(
        "--eval-input",
        required=True,
        help="Labeled eval CSV.",
    )
    parser.add_argument(
        "--target",
        default="call_purpose",
        choices=["call_purpose", "priority", "assig_group"],
        help="Target column name (default: call_purpose).",
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="CSV separator (default: ;).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="compare_saved_models",
        help="Output directory for comparison artifacts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for transformer inference (default: 32).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Fallback max sequence length for transformer inference (default: 512).",
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=["head", "head_tail", "middle_cut"],
        default="head",
        help="Fallback truncation strategy for transformer inference.",
    )
    parser.add_argument(
        "--ensemble-models",
        default="ruRoBERTa-large,RuBERT,RuBERT-tiny",
        help="Comma-separated model names for the top ensemble (default: ruRoBERTa-large,RuBERT,RuBERT-tiny).",
    )
    parser.add_argument(
        "--ensemble-methods",
        default="soft,max,hard",
        help="Comma-separated ensemble methods (default: soft,max,hard).",
    )
    parser.add_argument(
        "--no-transformers",
        action="store_true",
        help="Skip transformer model discovery.",
    )
    parser.add_argument(
        "--no-sklearn",
        action="store_true",
        help="Skip sklearn/joblib model discovery.",
    )
    parser.add_argument(
        "--high-conf-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for production-oriented coverage metrics (default: 0.8).",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def file_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024 * 1024)


def load_eval_dataset(csv_path: Path, target: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=sep, dtype=str).fillna("")
    required = {"text", target}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")
    df = df[df["text"].astype(str).str.strip() != ""].copy()
    df[target] = df[target].astype(str).str.strip()
    df = df[df[target] != ""].copy()
    if df.empty:
        raise ValueError(f"No usable rows left in {csv_path}")
    return df.reset_index(drop=True)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _softmax_nd(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp / exp.sum(axis=1, keepdims=True)


def _load_tokenizer(model_ref: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_ref, fix_mistral_regex=True)
    except TypeError as exc:
        msg = str(exc)
        if "fix_mistral_regex" not in msg:
            if "BertPreTokenizer" in msg or "pre_tokenizer" in msg:
                return AutoTokenizer.from_pretrained(model_ref, use_fast=False)
            raise

    try:
        return AutoTokenizer.from_pretrained(model_ref)
    except TypeError as exc:
        msg = str(exc)
        if "BertPreTokenizer" in msg or "pre_tokenizer" in msg:
            return AutoTokenizer.from_pretrained(model_ref, use_fast=False)
        raise


def discover_transformers(root: Path, target: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    seen: set[Path] = set()
    for le_path in sorted(root.rglob("label_encoder.joblib")):
        model_dir = le_path.parent
        path_str = model_dir.as_posix()
        if f"/{target}/" not in path_str:
            continue
        if "/transformers/" not in path_str:
            continue
        if model_dir in seen:
            continue
        seen.add(model_dir)
        specs.append(ModelSpec(name=model_dir.name, path=model_dir, kind="transformer"))
    return specs


def discover_sklearn(root: Path, target: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for model_path in sorted(root.rglob("*.joblib")):
        path_str = model_path.as_posix()
        if f"/{target}/" not in path_str:
            continue
        if "/transformers/" in path_str:
            continue
        if model_path.name == "label_encoder.joblib":
            continue
        if model_path.name.startswith("stacker_"):
            continue
        rel_name = model_path.relative_to(root).with_suffix("").as_posix()
        specs.append(ModelSpec(name=rel_name, path=model_path, kind="sklearn"))
    return specs


def align_proba(proba: np.ndarray, classes: list[str], labels: list[str], n_rows: int) -> np.ndarray:
    aligned = np.zeros((n_rows, len(labels)), dtype=float)
    idx_map = {label: i for i, label in enumerate(labels)}
    for src_idx, cls in enumerate(classes):
        if cls in idx_map:
            aligned[:, idx_map[cls]] = proba[:, src_idx]
    return aligned


def evaluate_sklearn(spec: ModelSpec, texts: pd.Series, labels: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    model = joblib.load(spec.path)

    if hasattr(model, "predict_proba"):
        t1 = time.perf_counter()
        proba = model.predict_proba(texts)
        infer_ms_total = (time.perf_counter() - t1) * 1000
        classes = list(model.classes_) if hasattr(model, "classes_") else None
        if classes is None and hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            classes = list(getattr(clf, "classes_", []))
    elif hasattr(model, "decision_function"):
        t1 = time.perf_counter()
        scores = model.decision_function(texts)
        infer_ms_total = (time.perf_counter() - t1) * 1000
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        proba = _softmax_nd(scores)
        classes = list(model.classes_) if hasattr(model, "classes_") else None
        if classes is None and hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            classes = list(getattr(clf, "classes_", []))
    else:
        raise ValueError(f"Model {spec.path} has neither predict_proba nor decision_function")

    if not classes:
        raise ValueError(f"Could not infer class order for {spec.path}")

    aligned = align_proba(proba, classes, labels, len(texts))
    pred = np.array(labels)[aligned.argmax(axis=1)]
    return aligned, pred, {
        "infer_ms_total": infer_ms_total,
        "model_size_mb": file_size_mb(spec.path),
        "n_params": np.nan,
        "component_count": 1,
    }


def evaluate_transformer(
    spec: ModelSpec,
    texts: pd.Series,
    labels: list[str],
    batch_size: int,
    max_length: int,
    truncation_strategy: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from transformers import AutoModelForSequenceClassification

    le_path = spec.path / "label_encoder.joblib"
    if not le_path.exists():
        raise FileNotFoundError(f"label_encoder.joblib not found in {spec.path}")

    le = joblib.load(le_path)
    tokenizer = _load_tokenizer(str(spec.path))
    model = AutoModelForSequenceClassification.from_pretrained(str(spec.path))
    n_params = int(sum(p.numel() for p in model.parameters()))
    effective_max_length, effective_truncation_strategy = resolve_inference_config(
        spec.path, max_length, truncation_strategy
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs = []
    text_list = list(texts)
    t1 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            batch = text_list[start:start + batch_size]
            enc = encode_text_batch(
                tokenizer,
                batch,
                max_length=effective_max_length,
                truncation_strategy=effective_truncation_strategy,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    infer_ms_total = (time.perf_counter() - t1) * 1000

    proba = np.vstack(all_probs)
    aligned = align_proba(proba, list(le.classes_), labels, len(texts))
    pred = np.array(labels)[aligned.argmax(axis=1)]
    return aligned, pred, {
        "infer_ms_total": infer_ms_total,
        "model_size_mb": file_size_mb(spec.path),
        "n_params": n_params,
        "component_count": 1,
    }


def summarize_result(
    *,
    model_name: str,
    family: str,
    method: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    infer_ms_total: float,
    model_size_mb: float,
    n_params: float,
    component_count: int,
    high_conf_threshold: float,
) -> dict:
    metrics = metric_bundle(y_true, pred)
    conf = proba.max(axis=1)
    correct = pred == y_true
    high_mask = conf >= high_conf_threshold
    high_share = float(high_mask.mean()) if len(high_mask) else 0.0
    high_acc = float(correct[high_mask].mean()) if high_mask.any() else np.nan
    high_errors = int((~correct & high_mask).sum())

    return {
        "model": model_name,
        "family": family,
        "method": method,
        "n_samples": int(len(y_true)),
        "n_errors": int((pred != y_true).sum()),
        **metrics,
        "infer_ms_total": round(float(infer_ms_total), 3),
        "infer_ms_per_sample": round(float(infer_ms_total) / max(len(y_true), 1), 3),
        "model_size_mb": round(float(model_size_mb), 3),
        "n_params": int(n_params) if not pd.isna(n_params) else np.nan,
        "component_count": int(component_count),
        "confidence_mean": round(float(conf.mean()), 4),
        "confidence_p90": round(float(np.quantile(conf, 0.9)), 4),
        f"high_conf_share_{str(high_conf_threshold).replace('.', '_')}": round(high_share, 4),
        f"high_conf_accuracy_{str(high_conf_threshold).replace('.', '_')}": (
            round(high_acc, 4) if not pd.isna(high_acc) else np.nan
        ),
        f"high_conf_errors_{str(high_conf_threshold).replace('.', '_')}": high_errors,
    }


def main() -> int:
    args = parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    eval_path = Path(args.eval_input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not experiment_root.exists():
        raise SystemExit(f"Experiment root not found: {experiment_root}")
    if not eval_path.exists():
        raise SystemExit(f"Eval CSV not found: {eval_path}")

    eval_df = load_eval_dataset(eval_path, args.target, args.sep)
    texts = eval_df["text"].astype(str)
    y_true = eval_df[args.target].astype(str).to_numpy()
    labels = sorted(eval_df[args.target].astype(str).unique())

    transformer_specs = [] if args.no_transformers else discover_transformers(experiment_root, args.target)
    sklearn_specs = [] if args.no_sklearn else discover_sklearn(experiment_root, args.target)
    specs = transformer_specs + sklearn_specs
    if not specs:
        raise SystemExit(f"No matching models found under: {experiment_root}")

    print("Discovered models:")
    for spec in specs:
        print(f"  - [{spec.kind}] {spec.name} -> {spec.path}")

    rows: list[dict] = []
    predictions = eval_df.copy()
    proba_map: dict[str, np.ndarray] = {}
    meta_map: dict[str, dict] = {}

    for spec in specs:
        print(f"\nEvaluating: {spec.name}")
        try:
            if spec.kind == "transformer":
                proba, pred, meta = evaluate_transformer(
                    spec,
                    texts,
                    labels,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    truncation_strategy=args.truncation_strategy,
                )
            else:
                proba, pred, meta = evaluate_sklearn(spec, texts, labels)
        except Exception as exc:
            print(f"  SKIP: {exc}")
            continue

        proba_map[spec.name] = proba
        meta_map[spec.name] = meta
        rows.append(
            summarize_result(
                model_name=spec.name,
                family=spec.kind,
                method="base",
                y_true=y_true,
                pred=pred,
                proba=proba,
                infer_ms_total=meta["infer_ms_total"],
                model_size_mb=meta["model_size_mb"],
                n_params=meta["n_params"],
                component_count=meta["component_count"],
                high_conf_threshold=args.high_conf_threshold,
            )
        )

        col_key = safe_name(spec.name)
        predictions[f"pred_{col_key}"] = pred
        predictions[f"conf_{col_key}"] = proba.max(axis=1)

    ensemble_names = [name.strip() for name in args.ensemble_models.split(",") if name.strip()]
    ensemble_methods = [name.strip() for name in args.ensemble_methods.split(",") if name.strip()]
    vote_handlers = {"soft": soft_vote, "max": max_vote, "hard": hard_vote}
    missing_ensemble_models = [name for name in ensemble_names if name not in proba_map]

    if ensemble_names and not missing_ensemble_models:
        prob_stack = np.stack([proba_map[name] for name in ensemble_names], axis=0)
        infer_ms_total = float(sum(meta_map[name]["infer_ms_total"] for name in ensemble_names))
        model_size_mb = float(sum(meta_map[name]["model_size_mb"] for name in ensemble_names))
        n_params_values = [meta_map[name]["n_params"] for name in ensemble_names if not pd.isna(meta_map[name]["n_params"])]
        n_params = float(sum(n_params_values)) if len(n_params_values) == len(ensemble_names) else np.nan

        for method in ensemble_methods:
            if method not in vote_handlers:
                print(f"  SKIP ensemble method '{method}': unsupported")
                continue
            proba, pred = vote_handlers[method](prob_stack, labels)
            model_name = f"Ensemble({'+'.join(ensemble_names)}:{method})"
            rows.append(
                summarize_result(
                    model_name=model_name,
                    family="ensemble",
                    method=method,
                    y_true=y_true,
                    pred=pred,
                    proba=proba,
                    infer_ms_total=infer_ms_total,
                    model_size_mb=model_size_mb,
                    n_params=n_params,
                    component_count=len(ensemble_names),
                    high_conf_threshold=args.high_conf_threshold,
                )
            )
            col_key = safe_name(model_name)
            predictions[f"pred_{col_key}"] = pred
            predictions[f"conf_{col_key}"] = proba.max(axis=1)
    elif ensemble_names:
        print(f"\nTop ensemble skipped, missing models: {missing_ensemble_models}")

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["f1_weighted", "accuracy", "infer_ms_per_sample"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    comparison_path = output_dir / "comparison.csv"
    predictions_path = output_dir / "predictions.csv"
    summary_path = output_dir / "summary.json"

    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_root": str(experiment_root),
                "eval_input": str(eval_path),
                "target": args.target,
                "sep": args.sep,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "truncation_strategy": args.truncation_strategy,
                "ensemble_models": ensemble_names,
                "ensemble_methods": ensemble_methods,
                "high_conf_threshold": args.high_conf_threshold,
                "rows": comparison.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.set_option("display.max_columns", None)
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison : {comparison_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved summary    : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
