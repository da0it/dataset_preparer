#!/usr/bin/env python3
"""Bootstrap bagging experiments for transformer classifiers."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataset_tools.dataset_variants import load_training_frame, save_prepared_dataset
from training_tools.ensemble_cv import hard_vote, max_vote, soft_vote
from training_tools.train_advanced import (
    ResultStore,
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
        description=(
            "Train bootstrap bagging ensembles for one transformer architecture. "
            "Mode 'cv' evaluates bagging with out-of-fold validation. "
            "Mode 'train-final' trains bootstrap members on the full prepared dataset "
            "with out-of-bag validation and saves them for external evaluation."
        )
    )
    parser.add_argument("--mode", choices=["cv", "train-final"], default="cv",
                        help="Experiment mode (default: cv).")
    parser.add_argument("-i", "--input", required=True, help="Training CSV path.")
    parser.add_argument("-o", "--output", default="transformer_bagging",
                        help="Output directory.")
    parser.add_argument("-t", "--target", default="call_purpose",
                        choices=["call_purpose", "priority", "assig_group"],
                        help="Target column.")
    parser.add_argument("--sep", default=";", help="CSV separator.")
    parser.add_argument("--dataset-variant",
                        choices=["multiclass", "multiclass_with_spam", "binary_spam"],
                        default="multiclass", help="Dataset variant.")
    parser.add_argument("--model", required=True,
                        help="Transformer spec: Name=model_id_or_path")
    parser.add_argument("--n-estimators", type=int, default=5,
                        help="Number of bootstrap transformer members.")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Bootstrap sample size as a fraction of available train rows.")
    parser.add_argument("--methods", default="soft,max,hard",
                        help="Comma-separated aggregation methods: soft,max,hard")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of folds for mode=cv.")
    parser.add_argument("--fold-manifest", default=None,
                        help="Optional JSON fold manifest for mode=cv.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--freeze-layers", type=int, default=6,
                        help="Freeze first N encoder layers.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing.")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Early stopping patience.")
    parser.add_argument("--early-stopping-metric", choices=["f1", "loss"], default="loss",
                        help="Early stopping metric.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length.")
    parser.add_argument("--truncation-strategy", choices=["head", "head_tail", "middle_cut"],
                        default="head_tail", help="Truncation strategy.")
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


def parse_methods(raw: str) -> list[str]:
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    allowed = {"soft", "max", "hard"}
    bad = sorted(set(methods) - allowed)
    if bad:
        raise ValueError(f"Unsupported bagging methods: {bad}. Supported: soft,max,hard.")
    if not methods:
        raise ValueError("At least one method must be provided.")
    return methods


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "errors": int((y_true != y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def align_proba(proba: np.ndarray, classes: list[str], labels: list[str]) -> np.ndarray:
    aligned = np.zeros((len(proba), len(labels)), dtype=float)
    dst_index = {label: i for i, label in enumerate(labels)}
    for src_idx, cls in enumerate(classes):
        if cls in dst_index:
            aligned[:, dst_index[cls]] = proba[:, src_idx]
    return aligned


def bootstrap_indices(
    source_indices: np.ndarray,
    y_all: np.ndarray,
    sample_frac: float,
    labels: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_frac <= 0:
        raise ValueError("--sample-frac must be positive.")

    size = max(1, int(round(len(source_indices) * sample_frac)))
    by_label = {
        label: source_indices[y_all[source_indices] == label]
        for label in labels
    }

    boot = rng.choice(source_indices, size=size, replace=True)
    missing = [label for label in labels if not np.any(y_all[boot] == label)]
    if missing:
        # Rare but possible on small datasets. Force class coverage so weighted
        # CrossEntropyLoss can be built for all labels seen in validation.
        boot = boot.copy()
        if len(missing) <= len(boot):
            replace_pos = rng.choice(np.arange(len(boot)), size=len(missing), replace=False)
        else:
            replace_pos = np.arange(len(missing))
            boot = np.concatenate([boot, np.repeat(boot[:1], len(missing) - len(boot))])
        for pos, label in zip(replace_pos, missing):
            candidates = by_label.get(label)
            if candidates is not None and len(candidates) > 0:
                boot[pos] = rng.choice(candidates)
    return boot.astype(int)


def aggregate(prob_stack: np.ndarray, labels: list[str], method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "soft":
        return soft_vote(prob_stack, labels)
    if method == "max":
        return max_vote(prob_stack, labels)
    if method == "hard":
        return hard_vote(prob_stack, labels)
    raise ValueError(f"Unsupported method: {method}")


def finetune_kwargs(args: argparse.Namespace) -> dict:
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "grad_accum": args.grad_accum,
        "compile_model": args.compile,
        "lr": args.lr,
        "freeze_layers": args.freeze_layers,
        "label_smoothing": args.label_smoothing,
        "early_stopping": args.early_stopping,
        "early_stopping_metric": args.early_stopping_metric,
        "max_length": args.max_length,
        "truncation_strategy": args.truncation_strategy,
    }


def run_cv(args: argparse.Namespace, spec: ModelSpec, output_dir: Path) -> None:
    methods = parse_methods(args.methods)
    raw_df = load_training_frame(Path(args.input).resolve(), sep=args.sep)
    prepared_df = prepare_dataset(raw_df, args.target, args.dataset_variant)
    save_prepared_dataset(prepared_df, output_dir / "dataset_prepared.csv", sep=args.sep)

    texts = prepared_df["text"].astype(str).reset_index(drop=True)
    y_series = prepared_df[args.target].astype(str).reset_index(drop=True)
    y_all = y_series.to_numpy(dtype=object)
    labels = sorted(y_series.unique())
    n_samples = len(prepared_df)

    folds, manifest_path, manifest_loaded = resolve_cv_folds(
        texts,
        y_series,
        cv=args.cv,
        seed=args.seed,
        output_dir=output_dir,
        manifest_path=Path(args.fold_manifest).resolve() if args.fold_manifest else None,
    )

    print(f"Prepared dataset : {output_dir / 'dataset_prepared.csv'}")
    print(f"Samples          : {n_samples}")
    print(f"Classes          : {labels}")
    print(f"Base model       : {spec.name} = {spec.model_id}")
    print(f"Estimators       : {args.n_estimators}")
    print(f"Sample fraction  : {args.sample_frac}")
    print(f"Methods          : {methods}")
    print(f"Fold manifest    : {manifest_path} ({'loaded' if manifest_loaded else 'saved'})")

    pred_map = {method: np.empty(n_samples, dtype=object) for method in methods}
    conf_map = {method: np.zeros(n_samples, dtype=float) for method in methods}
    method_infer_ms = {method: 0.0 for method in methods}
    fold_scores = {method: [] for method in methods}
    member_scores: list[dict] = []

    fold_dir = output_dir / "_fold_metrics"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
        print(f"\nFold {fold_idx}/{args.cv} (train={len(tr_idx)}, val={len(val_idx)})")
        X_val = texts.iloc[val_idx].to_numpy(dtype=object)
        y_val = y_all[val_idx]
        fold_prob_list = []
        fold_infer_sum = 0.0

        for estimator_idx in range(args.n_estimators):
            member_seed = args.seed + fold_idx * 1000 + estimator_idx
            rng = np.random.default_rng(member_seed)
            boot_idx = bootstrap_indices(tr_idx, y_all, args.sample_frac, labels, rng)
            X_boot = texts.iloc[boot_idx].to_numpy(dtype=object)
            y_boot = y_all[boot_idx]

            print(
                f"  [{spec.name}] bag {estimator_idx + 1}/{args.n_estimators} "
                f"(seed={member_seed}, boot={len(boot_idx)})"
            )
            result = _finetune_transformer(
                model_name=spec.model_id,
                friendly_name=f"{spec.name} bag{estimator_idx + 1} fold{fold_idx}",
                X_train=X_boot,
                y_train=y_boot,
                X_test=X_val,
                y_test=y_val,
                store=None,
                output_dir=fold_dir,
                save_model=False,
                silent=True,
                return_details=True,
                seed=member_seed,
                **finetune_kwargs(args),
            )
            if result is None:
                raise RuntimeError(f"Fold {fold_idx}, bag {estimator_idx + 1}: training failed.")

            aligned = align_proba(
                np.asarray(result["proba"], dtype=float),
                list(result["classes"]),
                labels,
            )
            fold_prob_list.append(aligned)
            fold_infer_sum += float(result["infer_ms"])
            member_scores.append({
                "fold": fold_idx,
                "bag": estimator_idx + 1,
                "seed": member_seed,
                "bootstrap_size": int(len(boot_idx)),
                "f1_weighted": float(result["f1"]),
                "train_sec": float(result["train_sec"]),
                "infer_ms": float(result["infer_ms"]),
                "best_epoch": int(result["best_epoch"]),
                "best_val_loss": float(result["best_val_loss"]),
                "best_val_f1": float(result["best_val_f1"]),
            })
            print(f"    member F1 = {float(result['f1']):.3f}")

        prob_stack = np.stack(fold_prob_list, axis=0)
        for method in methods:
            t0 = time.perf_counter()
            proba, pred = aggregate(prob_stack, labels, method)
            combine_ms = (time.perf_counter() - t0) * 1000
            pred_map[method][val_idx] = pred
            conf_map[method][val_idx] = proba.max(axis=1)
            method_infer_ms[method] += fold_infer_sum + combine_ms
            fold_f1 = f1_score(y_val, pred, average="weighted", zero_division=0)
            fold_scores[method].append(float(fold_f1))
            print(f"  [Bagging {method}] F1 = {fold_f1:.3f}")

    per_sample = prepared_df.copy()
    per_sample["y_true"] = y_all
    rows = []
    for method in methods:
        pred = pred_map[method]
        per_sample[f"pred_bagging_{method}"] = pred
        per_sample[f"conf_bagging_{method}"] = conf_map[method]
        metrics = metric_bundle(y_all, pred)
        metrics["train_sec"] = np.nan
        metrics["infer_ms_per_sample"] = round(method_infer_ms[method] / max(n_samples, 1), 3)
        rows.append({
            "model": f"Bagging {spec.name} ({args.n_estimators}x, {method})",
            "method": f"bagging_{method}",
            "group": "ensemble",
            "notes": (
                f"cv={args.cv}; n_estimators={args.n_estimators}; "
                f"sample_frac={args.sample_frac}; "
                f"folds={','.join(f'{s:.4f}' for s in fold_scores[method])}"
            ),
            **metrics,
        })
        save_confusion_matrix(y_all.tolist(), pred.tolist(), f"Bagging_{spec.name}_{method}_CV{args.cv}", output_dir)

    summary_df = pd.DataFrame(rows).sort_values(
        ["f1_weighted", "accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_path = output_dir / "bagging_cv_comparison.csv"
    predictions_path = output_dir / "bagging_cv_predictions.csv"
    members_path = output_dir / "bagging_cv_members.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    per_sample.to_csv(predictions_path, index=False, encoding="utf-8", sep=args.sep)
    pd.DataFrame(member_scores).to_csv(members_path, index=False, encoding="utf-8")

    payload = {
        "mode": "cv",
        "input": str(Path(args.input).resolve()),
        "target": args.target,
        "dataset_variant": args.dataset_variant,
        "cv": args.cv,
        "seed": args.seed,
        "model": {"name": spec.name, "model_id": spec.model_id},
        "n_estimators": args.n_estimators,
        "sample_frac": args.sample_frac,
        "methods": methods,
        "fold_manifest": str(manifest_path),
        "results": summary_df.to_dict(orient="records"),
    }
    summary_json = output_dir / "bagging_cv_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nBagging CV comparison:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved comparison : {summary_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved members    : {members_path}")
    print(f"Saved summary    : {summary_json}")


def run_train_final(args: argparse.Namespace, spec: ModelSpec, output_dir: Path) -> None:
    raw_df = load_training_frame(Path(args.input).resolve(), sep=args.sep)
    prepared_df = prepare_dataset(raw_df, args.target, args.dataset_variant)
    save_prepared_dataset(prepared_df, output_dir / "dataset_prepared.csv", sep=args.sep)

    texts = prepared_df["text"].astype(str).reset_index(drop=True)
    y_series = prepared_df[args.target].astype(str).reset_index(drop=True)
    y_all = y_series.to_numpy(dtype=object)
    labels = sorted(y_series.unique())
    all_indices = np.arange(len(prepared_df))

    print(f"Prepared dataset : {output_dir / 'dataset_prepared.csv'}")
    print(f"Samples          : {len(prepared_df)}")
    print(f"Classes          : {labels}")
    print(f"Base model       : {spec.name} = {spec.model_id}")
    print(f"Estimators       : {args.n_estimators}")
    print(f"Sample fraction  : {args.sample_frac}")
    print("Validation       : out-of-bag rows for each bootstrap member")

    member_rows = []
    model_specs = []
    store = ResultStore()

    for estimator_idx in range(args.n_estimators):
        member_seed = args.seed + estimator_idx
        rng = np.random.default_rng(member_seed)
        boot_idx = bootstrap_indices(all_indices, y_all, args.sample_frac, labels, rng)
        boot_unique = np.unique(boot_idx)
        oob_idx = np.setdiff1d(all_indices, boot_unique, assume_unique=False)
        if len(oob_idx) == 0:
            raise RuntimeError(
                "Bootstrap produced no out-of-bag rows. Reduce --sample-frac or increase dataset size."
            )

        friendly_name = f"{spec.name}_bag{estimator_idx + 1}"
        print(
            f"\n[{friendly_name}] seed={member_seed}, "
            f"boot={len(boot_idx)}, unique={len(boot_unique)}, oob={len(oob_idx)}"
        )
        result = _finetune_transformer(
            model_name=spec.model_id,
            friendly_name=friendly_name,
            X_train=texts.iloc[boot_idx].to_numpy(dtype=object),
            y_train=y_all[boot_idx],
            X_test=texts.iloc[oob_idx].to_numpy(dtype=object),
            y_test=y_all[oob_idx],
            store=store,
            output_dir=output_dir,
            save_model=True,
            silent=False,
            return_details=True,
            seed=member_seed,
            **finetune_kwargs(args),
        )
        if result is None:
            raise RuntimeError(f"Final bag {estimator_idx + 1}: training failed.")

        model_dir = output_dir / friendly_name
        member_rows.append({
            "bag": estimator_idx + 1,
            "seed": member_seed,
            "model_name": friendly_name,
            "model_dir": str(model_dir),
            "bootstrap_size": int(len(boot_idx)),
            "bootstrap_unique_rows": int(len(boot_unique)),
            "oob_rows": int(len(oob_idx)),
            "oob_f1_weighted": float(result["f1"]),
            "train_sec": float(result["train_sec"]),
            "infer_ms": float(result["infer_ms"]),
            "best_epoch": int(result["best_epoch"]),
            "best_val_loss": float(result["best_val_loss"]),
            "best_val_f1": float(result["best_val_f1"]),
        })
        model_specs.append(f'--transformer-model "{friendly_name}={model_dir}"')

    members_df = pd.DataFrame(member_rows)
    members_path = output_dir / "bagging_members.csv"
    members_df.to_csv(members_path, index=False, encoding="utf-8")

    specs_path = output_dir / "bagging_model_specs.txt"
    with open(specs_path, "w", encoding="utf-8") as f:
        f.write("\\\n".join(model_specs))
        f.write("\n")

    summary_path = output_dir / "bagging_train_summary.json"
    payload = {
        "mode": "train-final",
        "input": str(Path(args.input).resolve()),
        "target": args.target,
        "dataset_variant": args.dataset_variant,
        "seed": args.seed,
        "model": {"name": spec.name, "model_id": spec.model_id},
        "n_estimators": args.n_estimators,
        "sample_frac": args.sample_frac,
        "members": member_rows,
        "note": (
            "Members are trained on bootstrap samples of the prepared dataset. "
            "Each member uses out-of-bag rows for early stopping. "
            "Use the saved member directories with ensemble_external_eval.py on an external eval set."
        ),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nSaved final bagging members:")
    print(members_df.to_string(index=False))
    print(f"\nSaved members CSV : {members_path}")
    print(f"Saved model specs : {specs_path}")
    print(f"Saved summary     : {summary_path}")


def main() -> int:
    args = parse_args()
    if args.n_estimators < 2:
        raise ValueError("--n-estimators must be at least 2.")

    spec = parse_model_spec(args.model)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "cv":
        run_cv(args, spec, output_dir)
    else:
        run_train_final(args, spec, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
