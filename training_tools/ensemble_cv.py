#!/usr/bin/env python3
"""Cross-validated transformer ensemble experiments with fold-wise retraining."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

from dataset_tools.dataset_variants import load_training_frame, save_prepared_dataset
from training_tools.train_advanced import (
    _finetune_transformer,
    prepare_dataset,
    resolve_cv_folds,
    save_confusion_matrix,
)


@dataclass
class ModelSpec:
    name: str
    model_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated ensemble experiments for transformer models."
    )
    parser.add_argument("-i", "--input", required=True, help="Training CSV path.")
    parser.add_argument("-o", "--output", default="ensemble_cv", help="Output directory.")
    parser.add_argument("-t", "--target", default="call_purpose",
                        choices=["call_purpose", "priority", "assig_group"],
                        help="Target column.")
    parser.add_argument("--sep", default=";", help="CSV separator.")
    parser.add_argument("--dataset-variant", choices=["multiclass", "multiclass_with_spam", "binary_spam"],
                        default="multiclass", help="Dataset variant.")
    parser.add_argument("--cv", type=int, default=5, help="Number of folds (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fold-manifest", default=None,
                        help="Optional JSON with fold indices to reuse across runs.")
    parser.add_argument("--methods", default="soft,max,hard",
                        help="Comma-separated methods: soft,max,hard,stacking")
    parser.add_argument("--stacker", choices=["logreg", "mlp"], default="logreg",
                        help="Meta-model for stacking (default: logreg).")
    parser.add_argument("--transformer-model", action="append", default=[],
                        help="Base model spec: Name=model_id_or_path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--freeze-layers", type=int, default=6, help="Freeze first N layers.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--early-stopping", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--early-stopping-metric", choices=["f1", "loss"], default="f1",
                        help="Early stopping metric.")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--truncation-strategy", choices=["head", "head_tail", "middle_cut"],
                        default="head", help="Truncation strategy.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    return parser.parse_args()


def parse_model_spec(raw: str) -> ModelSpec:
    if "=" in raw:
        name, model_id = raw.split("=", 1)
    else:
        model_id = raw
        name = Path(raw).name
    return ModelSpec(name=name.strip(), model_id=model_id.strip())


def align_proba(
    proba: np.ndarray,
    classes: list[str],
    labels: list[str],
) -> np.ndarray:
    aligned = np.zeros((len(proba), len(labels)), dtype=float)
    idx_map = {label: i for i, label in enumerate(labels)}
    for src_idx, cls in enumerate(classes):
        if cls in idx_map:
            aligned[:, idx_map[cls]] = proba[:, src_idx]
    return aligned


def soft_vote(prob_stack: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    proba = prob_stack.mean(axis=0)
    pred = np.array(labels)[proba.argmax(axis=1)]
    return proba, pred


def max_vote(prob_stack: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    proba = prob_stack.max(axis=0)
    pred = np.array(labels)[proba.argmax(axis=1)]
    return proba, pred


def hard_vote(prob_stack: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    preds_idx = prob_stack.argmax(axis=2)
    _, n_samples, n_classes = prob_stack.shape
    avg_probs = prob_stack.mean(axis=0)
    final_idx = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        counts = np.bincount(preds_idx[:, i], minlength=n_classes)
        winners = np.flatnonzero(counts == counts.max())
        if len(winners) == 1:
            final_idx[i] = winners[0]
        else:
            final_idx[i] = winners[int(avg_probs[i, winners].argmax())]
    return avg_probs, np.array(labels)[final_idx]


def flatten_meta_features(prob_maps: dict[str, np.ndarray], specs: list[ModelSpec]) -> tuple[np.ndarray, list[str]]:
    blocks = []
    columns = []
    for spec in specs:
        proba = prob_maps[spec.name]
        blocks.append(proba)
        for cls_idx in range(proba.shape[1]):
            columns.append(f"prob_{spec.name}_class_{cls_idx}")
    return np.concatenate(blocks, axis=1), columns


def fit_stacker(X_train: np.ndarray, y_train: np.ndarray, stacker_name: str, seed: int) -> object:
    if stacker_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=seed,
        )
    else:
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    model.fit(X_train, y_train)
    return model


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "errors": int((y_true != y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"soft", "max", "hard", "stacking"}
    bad_methods = sorted(set(methods) - allowed)
    if bad_methods:
        raise ValueError(
            f"Unsupported methods for ensemble CV: {bad_methods}. "
            "Supported: soft,max,hard,stacking."
        )

    specs = [parse_model_spec(raw) for raw in args.transformer_model]
    if len(specs) < 2:
        raise ValueError("Provide at least two --transformer-model specs.")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_training_frame(Path(args.input).resolve(), sep=args.sep)
    prepared_df = prepare_dataset(raw_df, args.target, args.dataset_variant)
    snapshot_path = output_dir / "dataset_prepared.csv"
    save_prepared_dataset(prepared_df, snapshot_path, sep=args.sep)

    texts = prepared_df["text"].astype(str).reset_index(drop=True)
    labels_series = prepared_df[args.target].astype(str).reset_index(drop=True)
    labels = sorted(labels_series.unique())

    print(f"Prepared dataset: {snapshot_path}")
    print(f"Samples         : {len(prepared_df)}")
    print(f"Classes         : {labels}")
    print(f"Models          : {[spec.name for spec in specs]}")
    print(f"Methods         : {methods}")

    n_samples = len(prepared_df)
    n_classes = len(labels)
    folds, manifest_path, manifest_loaded = resolve_cv_folds(
        texts,
        labels_series,
        cv=args.cv,
        seed=args.seed,
        output_dir=output_dir,
        manifest_path=Path(args.fold_manifest).resolve() if args.fold_manifest else None,
    )
    print(f"Fold manifest   : {manifest_path} ({'loaded' if manifest_loaded else 'saved'})")

    base_pred_map = {spec.name: np.empty(n_samples, dtype=object) for spec in specs}
    base_proba_map = {spec.name: np.zeros((n_samples, n_classes), dtype=float) for spec in specs}
    base_train_sec = {spec.name: 0.0 for spec in specs}
    base_infer_ms = {spec.name: 0.0 for spec in specs}
    base_fold_scores = {spec.name: [] for spec in specs}

    vote_methods = [method for method in methods if method in {"soft", "max", "hard"}]
    run_stacking = "stacking" in methods
    ens_pred_map = {method: np.empty(n_samples, dtype=object) for method in vote_methods}
    ens_conf_map = {method: np.zeros(n_samples, dtype=float) for method in vote_methods}
    ens_infer_ms = {method: 0.0 for method in vote_methods}
    ens_fold_scores = {method: [] for method in vote_methods}

    fold_dir = output_dir / "_fold_metrics"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
        print(f"\nFold {fold_idx}/{args.cv} (train={len(tr_idx)}, val={len(val_idx)})")
        X_train = texts.iloc[tr_idx].to_numpy(dtype=object)
        y_train = labels_series.iloc[tr_idx].to_numpy(dtype=object)
        X_val = texts.iloc[val_idx].to_numpy(dtype=object)
        y_val = labels_series.iloc[val_idx].to_numpy(dtype=object)

        fold_prob_list = []
        fold_infer_sum = 0.0

        for spec in specs:
            print(f"  [{spec.name}] {spec.model_id}")
            result = _finetune_transformer(
                model_name=spec.model_id,
                friendly_name=f"{spec.name} fold{fold_idx}",
                X_train=X_train,
                y_train=y_train,
                X_test=X_val,
                y_test=y_val,
                store=None,
                output_dir=fold_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                fp16=args.fp16,
                bf16=args.bf16,
                grad_accum=args.grad_accum,
                compile_model=args.compile,
                lr=args.lr,
                freeze_layers=args.freeze_layers,
                label_smoothing=args.label_smoothing,
                early_stopping=args.early_stopping,
                early_stopping_metric=args.early_stopping_metric,
                max_length=args.max_length,
                truncation_strategy=args.truncation_strategy,
                save_model=False,
                silent=True,
                return_details=True,
                seed=args.seed,
            )
            if result is None:
                raise RuntimeError(f"Fold {fold_idx}: model {spec.name} failed.")

            aligned_proba = align_proba(
                np.asarray(result["proba"], dtype=float),
                list(result["classes"]),
                labels,
            )
            fold_pred = np.asarray(result["y_pred"], dtype=object)
            fold_true = np.asarray(result["y_true"], dtype=object)
            fold_f1 = f1_score(fold_true, fold_pred, average="weighted", zero_division=0)

            base_pred_map[spec.name][val_idx] = fold_pred
            base_proba_map[spec.name][val_idx] = aligned_proba
            base_train_sec[spec.name] += float(result["train_sec"])
            base_infer_ms[spec.name] += float(result["infer_ms"])
            base_fold_scores[spec.name].append(float(fold_f1))
            fold_prob_list.append(aligned_proba)
            fold_infer_sum += float(result["infer_ms"])

            print(f"    F1 = {fold_f1:.3f}")

        fold_prob_stack = np.stack(fold_prob_list, axis=0)
        for method in vote_methods:
            t0 = time.perf_counter()
            if method == "soft":
                ensemble_proba, ensemble_pred = soft_vote(fold_prob_stack, labels)
            elif method == "max":
                ensemble_proba, ensemble_pred = max_vote(fold_prob_stack, labels)
            else:
                ensemble_proba, ensemble_pred = hard_vote(fold_prob_stack, labels)
            combine_ms = (time.perf_counter() - t0) * 1000
            fold_f1 = f1_score(y_val, ensemble_pred, average="weighted", zero_division=0)
            ens_pred_map[method][val_idx] = ensemble_pred
            ens_conf_map[method][val_idx] = ensemble_proba.max(axis=1)
            ens_infer_ms[method] += fold_infer_sum + combine_ms
            ens_fold_scores[method].append(float(fold_f1))
            print(f"  [Ensemble {method}] F1 = {fold_f1:.3f}")

    y_true_all = labels_series.to_numpy(dtype=object)
    results = []
    per_sample = prepared_df.copy()
    per_sample["y_true"] = y_true_all

    for spec in specs:
        pred = base_pred_map[spec.name]
        per_sample[f"pred_{spec.name}"] = pred
        metrics = metric_bundle(y_true_all, pred)
        metrics["train_sec"] = round(base_train_sec[spec.name], 1)
        metrics["infer_ms_per_sample"] = round(base_infer_ms[spec.name] / max(n_samples, 1), 3)
        results.append({
            "model": spec.name,
            "method": "base",
            "group": "transformers",
            "notes": f"cv={args.cv}; folds={','.join(f'{s:.4f}' for s in base_fold_scores[spec.name])}",
            **metrics,
        })
        save_confusion_matrix(y_true_all.tolist(), pred.tolist(), f"{spec.name}_CV{args.cv}", output_dir)

    for method in vote_methods:
        pred = ens_pred_map[method]
        per_sample[f"pred_ensemble_{method}"] = pred
        per_sample[f"conf_ensemble_{method}"] = ens_conf_map[method]
        metrics = metric_bundle(y_true_all, pred)
        metrics["train_sec"] = np.nan
        metrics["infer_ms_per_sample"] = round(ens_infer_ms[method] / max(n_samples, 1), 3)
        results.append({
            "model": f"Ensemble ({method})",
            "method": method,
            "group": "ensemble",
            "notes": f"cv={args.cv}; folds={','.join(f'{s:.4f}' for s in ens_fold_scores[method])}",
            **metrics,
        })
        save_confusion_matrix(y_true_all.tolist(), pred.tolist(), f"Ensemble_{method}_CV{args.cv}", output_dir)

    if run_stacking:
        X_stack, stack_feature_cols = flatten_meta_features(base_proba_map, specs)
        stack_features_df = pd.DataFrame(X_stack, columns=stack_feature_cols)
        if "filename" in prepared_df.columns:
            stack_features_df.insert(0, "filename", prepared_df["filename"].astype(str).values)
        stack_features_df.insert(0, args.target, y_true_all)
        stack_features_path = output_dir / "ensemble_cv_meta_features.csv"
        stack_features_df.to_csv(stack_features_path, index=False, encoding="utf-8", sep=args.sep)

        final_stacker = fit_stacker(
            X_stack,
            y_true_all,
            stacker_name=args.stacker,
            seed=args.seed,
        )
        stacker_path = output_dir / f"stacker_{args.stacker}.joblib"
        stacker_meta_path = output_dir / f"stacker_{args.stacker}_metadata.json"
        joblib.dump(final_stacker, stacker_path)
        stacker_metadata = {
            "target": args.target,
            "dataset_variant": args.dataset_variant,
            "cv": args.cv,
            "seed": args.seed,
            "stacker": args.stacker,
            "labels": labels,
            "feature_columns": stack_feature_cols,
            "models": [{"name": spec.name, "model_id": spec.model_id} for spec in specs],
            "fold_manifest": str(manifest_path),
            "training_input": str(Path(args.input).resolve()),
            "meta_features": str(stack_features_path),
            "note": (
                "Final stacker is trained on out-of-fold base-model probabilities "
                "from the full CV dataset. Use it only on an external eval/test set "
                "that was not used to train base models or the stacker."
            ),
        }
        with open(stacker_meta_path, "w", encoding="utf-8") as f:
            json.dump(stacker_metadata, f, ensure_ascii=False, indent=2)

        stack_pred = np.empty(n_samples, dtype=object)
        stack_conf = np.zeros(n_samples, dtype=float)
        stack_fold_scores = []
        stack_infer_ms = 0.0
        for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
            stacker = fit_stacker(
                X_stack[tr_idx],
                y_true_all[tr_idx],
                stacker_name=args.stacker,
                seed=args.seed,
            )
            t0 = time.perf_counter()
            fold_pred = stacker.predict(X_stack[val_idx])
            if hasattr(stacker, "predict_proba"):
                fold_conf = stacker.predict_proba(X_stack[val_idx]).max(axis=1)
            else:
                fold_conf = np.full(len(val_idx), np.nan)
            stack_infer_ms += (time.perf_counter() - t0) * 1000

            stack_pred[val_idx] = fold_pred
            stack_conf[val_idx] = fold_conf
            fold_f1 = f1_score(
                y_true_all[val_idx],
                fold_pred,
                average="weighted",
                zero_division=0,
            )
            stack_fold_scores.append(float(fold_f1))
            print(f"  [Stacking {args.stacker}] fold {fold_idx} F1 = {fold_f1:.3f}")

        per_sample[f"pred_stacking_{args.stacker}"] = stack_pred
        per_sample[f"conf_stacking_{args.stacker}"] = stack_conf
        metrics = metric_bundle(y_true_all, stack_pred)
        metrics["train_sec"] = np.nan
        metrics["infer_ms_per_sample"] = round(stack_infer_ms / max(n_samples, 1), 3)
        results.append({
            "model": f"Stacking ({args.stacker})",
            "method": "stacking",
            "group": "ensemble",
            "notes": (
                f"meta_cv={args.cv}; stacker={args.stacker}; "
                f"folds={','.join(f'{s:.4f}' for s in stack_fold_scores)}"
            ),
            **metrics,
        })
        save_confusion_matrix(
            y_true_all.tolist(),
            stack_pred.tolist(),
            f"Stacking_{args.stacker}_CV{args.cv}",
            output_dir,
        )
        print(f"Saved stacking meta-features: {stack_features_path}")
        print(f"Saved final stacker       : {stacker_path}")
        print(f"Saved stacker metadata    : {stacker_meta_path}")

    summary_df = pd.DataFrame(results).sort_values(
        ["f1_weighted", "accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_path = output_dir / "ensemble_cv_comparison.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    per_sample_path = output_dir / "ensemble_cv_predictions.csv"
    per_sample.to_csv(per_sample_path, index=False, encoding="utf-8", sep=args.sep)

    payload = {
        "input": str(Path(args.input).resolve()),
        "target": args.target,
        "dataset_variant": args.dataset_variant,
        "cv": args.cv,
        "seed": args.seed,
        "methods": methods,
        "models": [{"name": spec.name, "model_id": spec.model_id} for spec in specs],
        "results": summary_df.to_dict(orient="records"),
    }
    json_path = output_dir / "ensemble_cv_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nEnsemble CV comparison:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved comparison : {summary_path}")
    print(f"Saved predictions: {per_sample_path}")
    print(f"Saved summary    : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
