#!/usr/bin/env python3
"""Run classical ML baselines excluding calibrated models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dataset_tools.dataset_variants import (
    load_training_frame,
    prepare_binary_spam_frame,
    prepare_multiclass_frame,
    save_prepared_dataset,
)
from training_tools.train_advanced import (
    MIN_SAMPLES_PER_CLASS,
    TEST_SIZE,
    ResultStore,
    _safe_artifact_name,
    build_baseline_pipelines,
    generate_comparison_report,
    print_gpu_info,
    save_confusion_matrix,
)


TARGETS = ["call_purpose", "priority", "assig_group"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical ML baselines without calibrated classifiers."
    )
    parser.add_argument("-i", "--input", required=True, help="Input CSV.")
    parser.add_argument("-t", "--target", default="call_purpose", choices=TARGETS + ["all"],
                        help="Target column (default: call_purpose)")
    parser.add_argument("-o", "--output", default="models_classic_no_calibration",
                        help="Output directory.")
    parser.add_argument("--sep", default=";", help="CSV separator (default: ;)")
    parser.add_argument("--dataset-variant", choices=["multiclass", "binary_spam"],
                        default="multiclass",
                        help="How to interpret input dataset (default: multiclass)")
    parser.add_argument("--eval-input", default=None,
                        help="Optional fixed validation/test CSV.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for hold-out split (default: 42)")
    return parser.parse_args()


def prepare_dataset(df: pd.DataFrame, target: str, dataset_variant: str) -> pd.DataFrame:
    if dataset_variant == "binary_spam":
        return prepare_binary_spam_frame(
            df, target=target, min_samples_per_class=MIN_SAMPLES_PER_CLASS
        )
    return prepare_multiclass_frame(
        df, target=target, min_samples_per_class=MIN_SAMPLES_PER_CLASS
    )


def build_non_calibrated_baselines():
    return {
        name: pipeline
        for name, pipeline in build_baseline_pipelines().items()
        if "Calibrated" not in name
    }


def run_models(X_train, y_train, X_test, y_test, store: ResultStore, output_dir: Path) -> None:
    print(f"\n{'═' * 60}")
    print("  Classical ML without calibration")
    print(f"{'═' * 60}")

    for name, pipeline in build_non_calibrated_baselines().items():
        print(f"\n  [{name}]")
        try:
            t0 = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = pipeline.predict(X_test)
            infer_ms = (time.perf_counter() - t1) * 1000

            f1 = store.record(name, "baseline", y_test, y_pred, train_sec, infer_ms)
            print(f"    Hold-out F1: {f1:.3f}  |  Train: {train_sec:.1f}s")
            print(classification_report(y_test, y_pred, zero_division=0))

            save_confusion_matrix(y_test, y_pred, name, output_dir)
            joblib.dump(pipeline, output_dir / f"{_safe_artifact_name(name)}.joblib")
        except Exception as exc:
            print(f"    ОШИБКА: {exc}")


def run_target(df: pd.DataFrame, eval_df: pd.DataFrame | None, target: str, output_dir: Path, args) -> None:
    print(f"\n{'━' * 60}")
    print(f"  ТАРГЕТ: {target}  |  variant: {args.dataset_variant}")
    print(f"{'━' * 60}")

    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    prepared_df = prepare_dataset(df, target, args.dataset_variant)
    snapshot_path = target_dir / "dataset_prepared.csv"
    save_prepared_dataset(prepared_df, snapshot_path, sep=args.sep)
    print(f"  Prepared dataset: {snapshot_path}")

    X = prepared_df["text"].reset_index(drop=True)
    y = prepared_df[target].reset_index(drop=True)
    print(f"  Всего: {len(X)} примеров, {y.nunique()} классов")

    if eval_df is not None:
        prepared_eval_df = prepare_dataset(eval_df, target, args.dataset_variant)
        train_labels = set(y.unique())
        prepared_eval_df = prepared_eval_df[prepared_eval_df[target].isin(train_labels)].copy()
        if prepared_eval_df.empty:
            print("  Пропуск: eval split не содержит ни одного класса из train split.")
            return

        eval_snapshot_path = target_dir / "dataset_eval_prepared.csv"
        save_prepared_dataset(prepared_eval_df, eval_snapshot_path, sep=args.sep)
        print(f"  Prepared eval dataset: {eval_snapshot_path}")

        X_train, y_train = X, y
        X_test = prepared_eval_df["text"].reset_index(drop=True)
        y_test = prepared_eval_df[target].reset_index(drop=True)
        print(f"\n  Train(fixed): {len(X_train)}  |  Eval(fixed): {len(X_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=args.seed, stratify=y
        )
        print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    store = ResultStore()
    run_models(X_train, y_train, X_test, y_test, store, target_dir)
    if store.records:
        generate_comparison_report(store, target_dir, target)


def main() -> int:
    args = parse_args()
    print_gpu_info()

    csv_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_frame(csv_path, args.sep)
    eval_df = load_training_frame(Path(args.eval_input).resolve(), args.sep) if args.eval_input else None

    targets = TARGETS if args.target == "all" else [args.target]
    for target in targets:
        run_target(df, eval_df, target, output_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
