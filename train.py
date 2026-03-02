#!/usr/bin/env python3
"""
ML Training Testbed for support call classification.

Trains and evaluates multiple models for three targets:
  - call_purpose    (цель звонка)
  - priority        (приоритет)
  - assigned_group  (группа специалистов)

Models:
  - TF-IDF + LinearSVC  (fast baseline)
  - TF-IDF + Random Forest
  - TF-IDF + KNN

Usage:
    pip install scikit-learn pandas matplotlib seaborn joblib
    python train.py --input dataset_clean.csv
    python train.py --input dataset_clean.csv --target priority
    python train.py --input dataset_clean.csv --target all
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

TARGETS = ["call_purpose", "priority", "assigned_group"]
SEP = ";"  # CSV separator — change to "," if needed
MIN_SAMPLES_PER_CLASS = 5  # classes with fewer examples are dropped


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(csv_path: Path, target: str) -> tuple[pd.Series, pd.Series]:
    """Load CSV, keep only training samples, drop empty/rare classes."""
    df = pd.read_csv(csv_path, sep=SEP, dtype=str).fillna("")

    # Keep only rows explicitly marked for training
    if "is_training_sample" in df.columns:
        df = df[df["is_training_sample"].str.strip() == "1"]

    # Drop rows with empty text or empty target
    df = df[df["text"].str.strip() != ""]
    df = df[df[target].str.strip() != ""]

    # Drop rare classes
    counts = df[target].value_counts()
    valid_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    dropped = counts[counts < MIN_SAMPLES_PER_CLASS]
    if len(dropped) > 0:
        print(f"  Dropping {len(dropped)} rare class(es) "
              f"(< {MIN_SAMPLES_PER_CLASS} samples): {dropped.index.tolist()}")
    df = df[df[target].isin(valid_classes)]

    if len(df) == 0:
        raise ValueError(
            f"No usable samples for target '{target}'. "
            f"Make sure is_training_sample=1 and '{target}' is filled."
        )

    return df["text"], df[target]


# ── Model definitions ─────────────────────────────────────────────────────────

def build_pipelines() -> dict[str, Pipeline]:
    tfidf = dict(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=2,
    )
    return {
        "TF-IDF + SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   LinearSVC(max_iter=2000, class_weight="balanced")),
        ]),
        "TF-IDF + RandomForest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "TF-IDF + KNN": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf",   KNeighborsClassifier(n_neighbors=5, metric="cosine")),
        ]),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, pipeline: Pipeline,
             X: pd.Series, y: pd.Series,
             output_dir: Path) -> dict:
    """Cross-validate, fit on full data, save model + confusion matrix."""

    print(f"\n  [{name}]")

    # Cross-validation (stratified 5-fold)
    n_splits = min(5, y.value_counts().min())
    if n_splits < 2:
        print("    Not enough samples for CV — skipping cross-validation.")
        cv_scores = np.array([0.0])
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        print(f"    CV F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full dataset
    pipeline.fit(X, y)

    # Training-set metrics (for reference)
    y_pred = pipeline.predict(X)
    train_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    print(f"    Train F1 (weighted): {train_f1:.3f}")
    print()
    print(classification_report(y, y_pred, zero_division=0))

    # Confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    _save_confusion_matrix(cm, labels, name, output_dir)

    # Save model
    safe_name = name.replace(" ", "_").replace("+", "plus").replace("/", "-")
    model_path = output_dir / f"{safe_name}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"    Saved: {model_path.name}")

    return {
        "model": name,
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std":  round(float(cv_scores.std()), 4),
        "train_f1":   round(train_f1, 4),
        "model_path": str(model_path),
    }


def _save_confusion_matrix(cm, labels, model_name: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    safe = model_name.replace(" ", "_").replace("+", "plus").replace("/", "-")
    ax.set_title(f"{model_name}")
    plt.tight_layout()
    path = output_dir / f"cm_{safe}.png"
    plt.savefig(path, dpi=120)
    plt.close()


def _print_class_distribution(y: pd.Series):
    counts = y.value_counts()
    print(f"  Classes ({len(counts)}):")
    for cls, cnt in counts.items():
        bar = "█" * min(cnt, 40)
        print(f"    {cls:<30} {cnt:>4}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_target(csv_path: Path, target: str, output_dir: Path):
    print(f"\n{'═' * 60}")
    print(f"  TARGET: {target}")
    print(f"{'═' * 60}")

    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        X, y = load_data(csv_path, target)
    except ValueError as e:
        print(f"  Skipping: {e}")
        return

    print(f"  Samples : {len(X)}")
    _print_class_distribution(y)

    pipelines = build_pipelines()
    results = []

    for name, pipeline in pipelines.items():
        try:
            result = evaluate(name, pipeline, X, y, target_dir)
            results.append(result)
        except Exception as exc:
            print(f"  ERROR training {name}: {exc}")

    # Summary table
    if results:
        print(f"\n  {'─' * 50}")
        print(f"  Summary for '{target}':")
        print(f"  {'Model':<30} {'CV F1':>8} {'±':>6} {'Train F1':>10}")
        print(f"  {'─' * 50}")
        for r in sorted(results, key=lambda x: x["cv_f1_mean"], reverse=True):
            print(f"  {r['model']:<30} {r['cv_f1_mean']:>8.3f} {r['cv_f1_std']:>6.3f} {r['train_f1']:>10.3f}")

        # Save results JSON
        results_path = target_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML classifiers on support call data."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Cleaned CSV file (output of prepare_dataset.py)")
    parser.add_argument("--target", "-t", default="call_purpose",
                        choices=TARGETS + ["all"],
                        help="Column to predict (default: call_purpose)")
    parser.add_argument("--output", "-o", default="models",
                        help="Output directory for models and plots (default: ./models)")
    parser.add_argument("--sep", default=";",
                        help="CSV separator (default: ;)")

    args = parser.parse_args()

    global SEP
    SEP = args.sep

    csv_path   = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: file '{csv_path}' not found.")
        return

    targets = TARGETS if args.target == "all" else [args.target]
    for target in targets:
        run_target(csv_path, target, output_dir)

    print(f"\n{'═' * 60}")
    print(f"  Done. Models and plots saved to: {output_dir}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
