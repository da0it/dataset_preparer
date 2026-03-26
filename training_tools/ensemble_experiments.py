#!/usr/bin/env python3
"""
ensemble_experiments.py — экспериментальный пайплайн ансамблей по уже обученным моделям.

Поддерживаемые режимы:
  - soft voting   : среднее вероятностей
  - max voting    : максимум вероятностей по каждому классу
  - hard voting   : голосование большинством
  - stacking      : meta-model по вероятностям базовых моделей

Поддерживаемые базовые модели:
  - sklearn .joblib
  - HuggingFace sequence classification dirs + label_encoder.joblib

Ограничения:
  - Все модели должны относиться к одному target и одной задаче.
  - Для stacking нужен отдельный meta-input.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

try:
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.neural_network import MLPClassifier
except ModuleNotFoundError:
    joblib = None
    plt = None
    np = None
    pd = None
    sns = None
    LogisticRegression = None
    MLPClassifier = None


@dataclass
class ModelSpec:
    name: str
    path: Path
    kind: str  # sklearn | transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble experiments over saved sklearn and transformer models."
    )
    parser.add_argument("--meta-input", default=None,
                        help="Optional labeled CSV for training stacking meta-model.")
    parser.add_argument("--test-input", required=True,
                        help="Labeled CSV for final evaluation.")
    parser.add_argument("--target", required=True,
                        help="Target column name.")
    parser.add_argument("--sep", default=";",
                        help="CSV separator (default: ;)")
    parser.add_argument("--output", "-o", default="ensemble_experiments",
                        help="Output directory.")
    parser.add_argument("--methods", default="soft,max,hard,stacking",
                        help="Comma-separated methods: soft,max,hard,stacking")
    parser.add_argument("--stacker", choices=["logreg", "mlp"], default="logreg",
                        help="Meta-model for stacking (default: logreg)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for transformer inference.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length for transformer inference.")
    parser.add_argument("--sklearn-model", action="append", default=[],
                        help="Saved sklearn model spec: Name=/abs/path/model.joblib")
    parser.add_argument("--transformer-model", action="append", default=[],
                        help="Saved transformer dir spec: Name=/abs/path/model_dir")
    return parser.parse_args()


def parse_model_spec(raw: str, kind: str) -> ModelSpec:
    if "=" in raw:
        name, path = raw.split("=", 1)
    else:
        path = raw
        name = Path(raw).stem if kind == "sklearn" else Path(raw).name
    return ModelSpec(name=name.strip(), path=Path(path).resolve(), kind=kind)


def load_eval_frame(csv_path: Path, target: str, sep: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path, sep=sep, dtype=str).fillna("")
    required = {"text", target}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")
    df = df[df["text"].astype(str).str.strip() != ""].copy()
    df = df[df[target].astype(str).str.strip() != ""].copy()
    if df.empty:
        raise ValueError(f"No usable rows left in {csv_path}")
    return df["text"].astype(str), df[target].astype(str)


def _load_tokenizer(model_ref: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_ref)
    except TypeError as exc:
        msg = str(exc)
        if "BertPreTokenizer" in msg or "pre_tokenizer" in msg:
            return AutoTokenizer.from_pretrained(model_ref, use_fast=False)
        raise


def _softmax_nd(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_sklearn_proba(model_path: Path, texts: pd.Series, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    model = joblib.load(model_path)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
        classes = list(model.classes_) if hasattr(model, "classes_") else None
        if classes is None and hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            classes = list(getattr(clf, "classes_", []))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        proba = _softmax_nd(np.asarray(scores))
        classes = list(model.classes_) if hasattr(model, "classes_") else None
        if classes is None and hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            classes = list(getattr(clf, "classes_", []))
    else:
        raise ValueError(f"Model {model_path} has neither predict_proba nor decision_function")

    if not classes:
        raise ValueError(f"Could not infer class order for {model_path}")

    aligned = np.zeros((len(texts), len(labels)), dtype=float)
    idx_map = {label: i for i, label in enumerate(labels)}
    for src_idx, cls in enumerate(classes):
        if cls in idx_map:
            aligned[:, idx_map[cls]] = proba[:, src_idx]
    pred = np.array(labels)[aligned.argmax(axis=1)]
    return aligned, pred


def predict_transformer_proba(
    model_dir: Path,
    texts: pd.Series,
    labels: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from transformers import AutoModelForSequenceClassification

    le_path = model_dir / "label_encoder.joblib"
    if not le_path.exists():
        raise FileNotFoundError(f"label_encoder.joblib not found in {model_dir}")
    le = joblib.load(le_path)
    tokenizer = _load_tokenizer(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs = []
    text_list = list(texts)
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start:start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    proba = np.vstack(all_probs)
    classes = list(le.classes_)
    aligned = np.zeros((len(texts), len(labels)), dtype=float)
    idx_map = {label: i for i, label in enumerate(labels)}
    for src_idx, cls in enumerate(classes):
        if cls in idx_map:
            aligned[:, idx_map[cls]] = proba[:, src_idx]
    pred = np.array(labels)[aligned.argmax(axis=1)]
    return aligned, pred


def collect_model_outputs(
    specs: list[ModelSpec],
    texts: pd.Series,
    labels: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    names = []
    probas = []
    pred_map: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.kind == "sklearn":
            proba, pred = predict_sklearn_proba(spec.path, texts, labels)
        else:
            proba, pred = predict_transformer_proba(spec.path, texts, labels, batch_size, max_length)
        names.append(spec.name)
        probas.append(proba)
        pred_map[spec.name] = pred
    return names, np.stack(probas, axis=0), pred_map


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def hard_vote(prob_stack: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    preds_idx = prob_stack.argmax(axis=2)
    n_models, n_samples, n_classes = prob_stack.shape
    final_idx = np.zeros(n_samples, dtype=int)
    avg_probs = prob_stack.mean(axis=0)
    for i in range(n_samples):
        counts = np.bincount(preds_idx[:, i], minlength=n_classes)
        winners = np.flatnonzero(counts == counts.max())
        if len(winners) == 1:
            final_idx[i] = winners[0]
        else:
            tie_scores = avg_probs[i, winners]
            final_idx[i] = winners[int(tie_scores.argmax())]
    final_proba = avg_probs
    final_pred = np.array(labels)[final_idx]
    return final_proba, final_pred


def soft_vote(prob_stack: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    proba = prob_stack.mean(axis=0)
    pred = np.array(labels)[proba.argmax(axis=1)]
    return proba, pred


def max_vote(prob_stack: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    proba = prob_stack.max(axis=0)
    pred = np.array(labels)[proba.argmax(axis=1)]
    return proba, pred


def fit_stacker(
    meta_features: np.ndarray,
    y_meta: np.ndarray,
    stacker_name: str,
) -> object:
    if stacker_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=42,
        )
    else:
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    model.fit(meta_features, y_meta)
    return model


def flatten_meta_features(prob_stack: np.ndarray) -> np.ndarray:
    # [n_models, n_samples, n_classes] -> [n_samples, n_models*n_classes]
    return np.transpose(prob_stack, (1, 0, 2)).reshape(prob_stack.shape[1], -1)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args()

    if any(dep is None for dep in (joblib, np, pd, plt, sns, LogisticRegression, MLPClassifier)):
        raise SystemExit(
            "Missing dependencies: install joblib, numpy, pandas, matplotlib, seaborn, scikit-learn."
        )

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"soft", "max", "hard", "stacking"}
    bad_methods = sorted(set(methods) - allowed)
    if bad_methods:
        raise ValueError(f"Unsupported methods: {bad_methods}")

    specs = [parse_model_spec(raw, "sklearn") for raw in args.sklearn_model]
    specs += [parse_model_spec(raw, "transformer") for raw in args.transformer_model]
    if len(specs) < 2:
        raise ValueError("Provide at least two base models.")

    test_texts, test_y = load_eval_frame(Path(args.test_input), args.target, args.sep)
    labels = sorted(test_y.unique())
    test_names, test_prob_stack, test_pred_map = collect_model_outputs(
        specs, test_texts, labels, args.batch_size, args.max_length
    )

    results = []
    per_sample = pd.DataFrame({
        "text": test_texts.values,
        "y_true": test_y.values,
    })

    for name, pred in test_pred_map.items():
        metrics = metric_bundle(test_y.to_numpy(), pred)
        results.append({"model": name, "method": "base", **metrics})
        per_sample[f"pred_{name}"] = pred
        save_confusion_matrix(
            test_y.to_numpy(),
            pred,
            labels,
            f"Base: {name}",
            output_dir / f"cm_base_{name.replace(' ', '_')}.png",
        )

    vote_handlers = {
        "soft": soft_vote,
        "max": max_vote,
        "hard": hard_vote,
    }
    for method in [m for m in methods if m in vote_handlers]:
        ensemble_proba, ensemble_pred = vote_handlers[method](test_prob_stack, labels)
        metrics = metric_bundle(test_y.to_numpy(), ensemble_pred)
        results.append({"model": f"Ensemble ({method})", "method": method, **metrics})
        per_sample[f"pred_ensemble_{method}"] = ensemble_pred
        per_sample[f"conf_ensemble_{method}"] = ensemble_proba.max(axis=1)
        save_confusion_matrix(
            test_y.to_numpy(),
            ensemble_pred,
            labels,
            f"Ensemble: {method}",
            output_dir / f"cm_ensemble_{method}.png",
        )

    if "stacking" in methods:
        if not args.meta_input:
            raise ValueError("Stacking requires --meta-input.")
        meta_texts, meta_y = load_eval_frame(Path(args.meta_input), args.target, args.sep)
        _, meta_prob_stack, _ = collect_model_outputs(
            specs, meta_texts, labels, args.batch_size, args.max_length
        )
        X_meta_train = flatten_meta_features(meta_prob_stack)
        X_meta_test = flatten_meta_features(test_prob_stack)
        stacker = fit_stacker(X_meta_train, meta_y.to_numpy(), args.stacker)
        stack_pred = stacker.predict(X_meta_test)
        if hasattr(stacker, "predict_proba"):
            stack_conf = stacker.predict_proba(X_meta_test).max(axis=1)
        else:
            stack_conf = np.full(len(stack_pred), np.nan)
        metrics = metric_bundle(test_y.to_numpy(), stack_pred)
        results.append({"model": f"Stacking ({args.stacker})", "method": "stacking", **metrics})
        per_sample[f"pred_stacking_{args.stacker}"] = stack_pred
        per_sample[f"conf_stacking_{args.stacker}"] = stack_conf
        save_confusion_matrix(
            test_y.to_numpy(),
            stack_pred,
            labels,
            f"Stacking: {args.stacker}",
            output_dir / f"cm_stacking_{args.stacker}.png",
        )
        joblib.dump(stacker, output_dir / f"stacker_{args.stacker}.joblib")

    summary_df = pd.DataFrame(results).sort_values(
        ["f1_weighted", "accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_df.to_csv(output_dir / "ensemble_comparison.csv", index=False, encoding="utf-8")
    per_sample.to_csv(output_dir / "ensemble_predictions.csv", index=False, encoding="utf-8")

    summary_json = {
        "meta_input": args.meta_input,
        "test_input": args.test_input,
        "target": args.target,
        "methods": methods,
        "stacker": args.stacker if "stacking" in methods else None,
        "models": [
            {"name": spec.name, "path": str(spec.path), "kind": spec.kind}
            for spec in specs
        ],
        "results": summary_df.to_dict(orient="records"),
    }
    with open(output_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("\nEnsemble comparison:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
