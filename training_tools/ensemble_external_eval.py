#!/usr/bin/env python3
"""Evaluate saved ensemble models on an external labeled eval set."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from training_tools.ensemble_experiments import (
    collect_model_outputs,
    hard_vote,
    max_vote,
    metric_bundle,
    parse_model_spec,
    save_confusion_matrix,
    soft_vote,
)

try:
    import joblib
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    joblib = None
    np = None
    pd = None


@dataclass
class StackerBundle:
    model: object
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved base models and ensembles on an external eval dataset."
    )
    parser.add_argument("--test-input", required=True,
                        help="External labeled eval CSV.")
    parser.add_argument("--target", required=True,
                        help="Target column name.")
    parser.add_argument("--sep", default=";",
                        help="CSV separator (default: ;)")
    parser.add_argument("-o", "--output", default="ensemble_external_eval",
                        help="Output directory.")
    parser.add_argument("--methods", default="soft,max,hard",
                        help="Comma-separated methods: soft,max,hard,stacking")
    parser.add_argument("--synthetic-col", default="is_synthetic",
                        help="Optional column for real/synthetic split metrics.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for transformer inference.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Fallback max sequence length.")
    parser.add_argument("--truncation-strategy",
                        choices=["head", "head_tail", "middle_cut"],
                        default="head",
                        help="Fallback truncation strategy.")
    parser.add_argument("--sklearn-model", action="append", default=[],
                        help="Saved sklearn model spec: Name=/abs/path/model.joblib")
    parser.add_argument("--transformer-model", action="append", default=[],
                        help="Saved transformer dir spec: Name=/abs/path/model_dir")
    parser.add_argument("--stacker-model", default=None,
                        help="Saved stacker .joblib from ensemble_cv.py.")
    parser.add_argument("--stacker-metadata", default=None,
                        help="Stacker metadata JSON from ensemble_cv.py.")
    return parser.parse_args()


def load_eval_frame(csv_path: Path, target: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=sep, dtype=str).fillna("")
    required = {"text", target}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}. Available: {list(df.columns)}")
    df = df[df["text"].astype(str).str.strip() != ""].copy()
    df[target] = df[target].astype(str).str.strip()
    df = df[df[target] != ""].copy()
    if df.empty:
        raise ValueError(f"No usable rows left in {csv_path}")
    return df.reset_index(drop=True)


def load_stacker(model_path: str | None, metadata_path: str | None) -> StackerBundle | None:
    if not model_path and not metadata_path:
        return None
    if not model_path or not metadata_path:
        raise ValueError("--stacker-model and --stacker-metadata must be provided together.")

    metadata_file = Path(metadata_path).resolve()
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return StackerBundle(model=joblib.load(Path(model_path).resolve()), metadata=metadata)


def reorder_specs_for_stacker(specs, metadata: dict):
    expected = [item["name"] for item in metadata.get("models", [])]
    if not expected:
        return specs

    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in expected if name not in by_name]
    if missing:
        raise ValueError(
            "Stacker expects these base model names, but they were not provided: "
            f"{missing}. Provided: {sorted(by_name)}"
        )
    return [by_name[name] for name in expected]


def flatten_meta_features(prob_stack: np.ndarray) -> np.ndarray:
    return np.transpose(prob_stack, (1, 0, 2)).reshape(prob_stack.shape[1], -1)


def normalize_synthetic_mask(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "yes", "y", "synthetic", "синтетика"})


def rows_for_subset(df: pd.DataFrame, subset: str, synthetic_col: str) -> np.ndarray:
    if subset == "all" or synthetic_col not in df.columns:
        return np.ones(len(df), dtype=bool)
    synth = normalize_synthetic_mask(df[synthetic_col]).to_numpy()
    if subset == "synthetic":
        return synth
    if subset == "real":
        return ~synth
    raise ValueError(f"Unknown subset: {subset}")


def add_metrics(
    rows: list[dict],
    model_name: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    infer_ms_per_sample: float,
    subset: str,
) -> None:
    metrics = metric_bundle(y_true, y_pred)
    metrics["infer_ms_per_sample"] = round(float(infer_ms_per_sample), 3)
    metrics["subset"] = subset
    rows.append({"model": model_name, "method": method, **metrics})


def main() -> int:
    args = parse_args()
    if joblib is None or np is None or pd is None:
        raise SystemExit("Missing dependencies: install joblib, numpy, pandas.")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"soft", "max", "hard", "stacking"}
    bad_methods = sorted(set(methods) - allowed)
    if bad_methods:
        raise ValueError(f"Unsupported methods: {bad_methods}")

    stacker = load_stacker(args.stacker_model, args.stacker_metadata)
    if "stacking" in methods and stacker is None:
        raise ValueError("Method 'stacking' requires --stacker-model and --stacker-metadata.")

    specs = [parse_model_spec(raw, "sklearn") for raw in args.sklearn_model]
    specs += [parse_model_spec(raw, "transformer") for raw in args.transformer_model]
    if len(specs) < 2:
        raise ValueError("Provide at least two base models.")
    if stacker is not None:
        specs = reorder_specs_for_stacker(specs, stacker.metadata)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval_frame(Path(args.test_input).resolve(), args.target, args.sep)
    y_true = df[args.target].astype(str).to_numpy()

    if stacker is not None and stacker.metadata.get("labels"):
        labels = list(stacker.metadata["labels"])
        unseen = sorted(set(y_true) - set(labels))
        if unseen:
            raise ValueError(f"External eval contains labels absent from stacker training: {unseen}")
    else:
        labels = sorted(set(y_true))

    names, prob_stack, pred_map, infer_ms_map = collect_model_outputs(
        specs,
        df["text"].astype(str),
        labels,
        args.batch_size,
        args.max_length,
        args.truncation_strategy,
    )

    total_base_infer_ms = sum(infer_ms_map.values())
    per_sample = df.copy()
    per_sample["y_true"] = y_true
    summary_rows = []
    subset_rows = []

    predictions: dict[str, tuple[str, np.ndarray, float]] = {}
    for name in names:
        predictions[name] = ("base", pred_map[name], infer_ms_map[name] / max(len(df), 1))
        per_sample[f"pred_{name}"] = pred_map[name]

    vote_handlers = {"soft": soft_vote, "max": max_vote, "hard": hard_vote}
    for method in [m for m in methods if m in vote_handlers]:
        proba, pred = vote_handlers[method](prob_stack, labels)
        predictions[f"Ensemble ({method})"] = (
            method,
            pred,
            total_base_infer_ms / max(len(df), 1),
        )
        per_sample[f"pred_ensemble_{method}"] = pred
        per_sample[f"conf_ensemble_{method}"] = proba.max(axis=1)

    if "stacking" in methods and stacker is not None:
        X_stack = flatten_meta_features(prob_stack)
        expected_features = stacker.metadata.get("feature_columns", [])
        if expected_features and X_stack.shape[1] != len(expected_features):
            raise ValueError(
                f"Stacker expects {len(expected_features)} features, got {X_stack.shape[1]}. "
                "Check base model order and label set."
            )
        t0 = time.perf_counter()
        stack_pred = stacker.model.predict(X_stack)
        if hasattr(stacker.model, "predict_proba"):
            stack_conf = stacker.model.predict_proba(X_stack).max(axis=1)
        else:
            stack_conf = np.full(len(stack_pred), np.nan)
        stack_ms = (time.perf_counter() - t0) * 1000
        stacker_name = stacker.metadata.get("stacker", "stacker")
        predictions[f"Stacking ({stacker_name})"] = (
            "stacking",
            stack_pred,
            (total_base_infer_ms + stack_ms) / max(len(df), 1),
        )
        per_sample[f"pred_stacking_{stacker_name}"] = stack_pred
        per_sample[f"conf_stacking_{stacker_name}"] = stack_conf

    subsets = ["all"]
    if args.synthetic_col in df.columns:
        subsets.extend(["real", "synthetic"])

    for model_name, (method, pred, infer_ms_per_sample) in predictions.items():
        add_metrics(summary_rows, model_name, method, y_true, pred, infer_ms_per_sample, "all")
        save_confusion_matrix(
            y_true,
            pred,
            labels,
            model_name,
            output_dir / f"cm_{model_name.replace(' ', '_').replace('/', '_')}.png",
        )
        for subset in subsets:
            mask = rows_for_subset(df, subset, args.synthetic_col)
            if mask.sum() == 0:
                continue
            add_metrics(
                subset_rows,
                model_name,
                method,
                y_true[mask],
                np.asarray(pred)[mask],
                infer_ms_per_sample,
                subset,
            )

    summary_df = (
        pd.DataFrame(summary_rows)
        .drop_duplicates(subset=["model", "method", "subset"])
        .sort_values(["f1_weighted", "accuracy"], ascending=[False, False])
        .reset_index(drop=True)
    )
    subset_df = (
        pd.DataFrame(subset_rows)
        .sort_values(["subset", "f1_weighted", "accuracy"], ascending=[True, False, False])
        .reset_index(drop=True)
    )

    summary_path = output_dir / "ensemble_external_comparison.csv"
    subset_path = output_dir / "ensemble_external_comparison_by_subset.csv"
    predictions_path = output_dir / "ensemble_external_predictions.csv"
    summary_json_path = output_dir / "ensemble_external_summary.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    subset_df.to_csv(subset_path, index=False, encoding="utf-8")
    per_sample.to_csv(predictions_path, index=False, encoding="utf-8", sep=args.sep)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_input": str(Path(args.test_input).resolve()),
            "target": args.target,
            "labels": labels,
            "methods": methods,
            "models": [{"name": spec.name, "path": str(spec.path), "kind": spec.kind} for spec in specs],
            "stacker_metadata": args.stacker_metadata,
            "results": summary_df.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)

    print("\nExternal ensemble comparison:")
    print(summary_df.to_string(index=False))
    if args.synthetic_col in df.columns:
        print("\nBy subset:")
        print(subset_df.to_string(index=False))
    print(f"\nSaved comparison : {summary_path}")
    print(f"Saved by subset  : {subset_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved summary    : {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
