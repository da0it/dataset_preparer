#!/usr/bin/env python3
"""
ensemble.py — Voting ensemble for support call classification.

Base models (all TF-IDF based):
  1. TF-IDF + SVM        (Calibrated LinearSVC → soft probabilities)
  2. TF-IDF + RandomForest
  3. TF-IDF + XGBoost

Optional:
  4. SBERT + SVM         (requires sentence-transformers, enabled via --sbert)

Voting: soft (averaged class probabilities across all base models).

Outputs (per target):
  - Confusion matrices for each base model + ensemble
  - JSON run log with per-model and ensemble metrics
  - Summary table printed to stdout

Usage:
    python ensemble.py -i dataset_clean.csv
    python ensemble.py -i dataset_clean.csv --sbert
    python ensemble.py -i dataset_clean.csv -t all --sbert
    python ensemble.py -i dataset_clean.csv -t priority -o results/
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dataset_variants import (
    load_training_frame,
    prepare_binary_spam_frame,
    prepare_multiclass_frame,
    save_prepared_dataset,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

TARGETS = ["call_purpose", "priority", "assig_group"]
SEP = ";"
MIN_SAMPLES_PER_CLASS = 5
TEST_SIZE = 0.2

# ── Optional: SBERT transformer ───────────────────────────────────────────────

class SBERTTransformer(BaseEstimator, TransformerMixin):
    """Wraps sentence-transformers as an sklearn transformer."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self._load()
        return self._model.encode(list(X), show_progress_bar=False, batch_size=64)


# ── Data loading ──────────────────────────────────────────────────────────────

def prepare_dataset(df: pd.DataFrame, target: str, dataset_variant: str) -> pd.DataFrame:
    if dataset_variant == "binary_spam":
        return prepare_binary_spam_frame(
            df, target=target, min_samples_per_class=MIN_SAMPLES_PER_CLASS
        )
    return prepare_multiclass_frame(
        df, target=target, min_samples_per_class=MIN_SAMPLES_PER_CLASS
    )


# ── Base model builders ───────────────────────────────────────────────────────

def _tfidf_params():
    return dict(analyzer="word", ngram_range=(1, 2), max_features=15000,
                sublinear_tf=True, min_df=1)


def build_svm_pipeline() -> Pipeline:
    """TF-IDF + Calibrated SVM (soft probabilities via CalibratedClassifierCV)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_tfidf_params())),
        ("clf",   CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=3000, class_weight="balanced"), cv=3
        )),
    ])


def build_rf_pipeline() -> Pipeline:
    """TF-IDF + Random Forest (has predict_proba natively)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_tfidf_params())),
        ("clf",   RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])


def build_xgb_pipeline() -> Pipeline:
    """TF-IDF + XGBoost (has predict_proba natively)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_tfidf_params())),
        ("clf",   XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )),
    ])


def build_sbert_pipeline(model_name: str) -> Pipeline:
    """SBERT embeddings + Calibrated SVM."""
    return Pipeline([
        ("sbert", SBERTTransformer(model_name=model_name)),
        ("clf",   CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=3000, class_weight="balanced"), cv=3
        )),
    ])


# ── Ensemble builder ──────────────────────────────────────────────────────────

def build_ensemble(use_sbert: bool, sbert_model: str) -> tuple[VotingClassifier, list[str]]:
    """Build VotingClassifier from base models. Returns (ensemble, base_model_names)."""
    estimators = [
        ("svm", build_svm_pipeline()),
        ("rf",  build_rf_pipeline()),
        ("xgb", build_xgb_pipeline()),
    ]
    names = ["TF-IDF + SVM", "TF-IDF + RF", "TF-IDF + XGBoost"]

    if use_sbert:
        estimators.append(("sbert", build_sbert_pipeline(sbert_model)))
        names.append("SBERT + SVM")

    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return ensemble, names


# ── Confusion matrix ──────────────────────────────────────────────────────────

def save_confusion_matrix(cm, labels, title: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("+", "plus").replace("/", "-")
    path = output_dir / f"cm_{safe}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


# ── Per-model evaluation helper ───────────────────────────────────────────────

def eval_single(name: str, model: Pipeline,
                X_train: pd.Series, y_train: pd.Series,
                X_test: pd.Series, y_test: pd.Series,
                output_dir: Path) -> dict:
    """Train and evaluate a single base model."""
    n_splits = min(5, y_train.value_counts().min())
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring="f1_weighted", n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    else:
        cv_mean, cv_std = 0.0, 0.0

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    save_confusion_matrix(cm, labels, name, output_dir)

    return {
        "model":      name,
        "cv_f1_mean": round(float(cv_mean), 4),
        "cv_f1_std":  round(float(cv_std), 4),
        "test_f1":    round(float(test_f1), 4),
    }


# ── Main per-target routine ───────────────────────────────────────────────────

def run_target(df: pd.DataFrame, target: str, output_dir: Path,
               use_sbert: bool, sbert_model: str, run_meta: dict,
               dataset_variant: str) -> list[dict]:

    print(f"\n{'═' * 60}")
    print(f"  TARGET: {target}  |  variant: {dataset_variant}")
    print(f"{'═' * 60}")

    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        prepared_df = prepare_dataset(df, target, dataset_variant)
    except ValueError as e:
        print(f"  Skipping: {e}")
        return []

    snapshot_path = target_dir / "dataset_prepared.csv"
    save_prepared_dataset(prepared_df, snapshot_path, sep=SEP)
    print(f"  Prepared dataset: {snapshot_path}")

    X = prepared_df["text"]
    y = prepared_df[target]
    class_dist = y.value_counts().to_dict()
    print(f"  Samples: {len(X)}  |  Classes: {y.nunique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    ensemble, base_names = build_ensemble(use_sbert, sbert_model)
    results = []

    # ── Evaluate each base model individually ──
    print("  Base models:")
    for (est_name, model), display_name in zip(ensemble.estimators, base_names):
        print(f"    [{display_name}] training...")
        try:
            res = eval_single(display_name, model, X_train, y_train, X_test, y_test, target_dir)
            results.append(res)
            print(f"      CV F1: {res['cv_f1_mean']:.3f} ± {res['cv_f1_std']:.3f}  "
                  f"| Test F1: {res['test_f1']:.3f}")
        except Exception as e:
            print(f"      ERROR: {e}")

    # ── Evaluate ensemble ──
    print(f"\n  [Ensemble (soft voting)] training...")
    try:
        n_splits = min(5, y_train.value_counts().min())
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(ensemble, X_train, y_train,
                                        cv=cv, scoring="f1_weighted", n_jobs=1)
            ens_cv_mean, ens_cv_std = cv_scores.mean(), cv_scores.std()
        else:
            ens_cv_mean, ens_cv_std = 0.0, 0.0

        ensemble.fit(X_train, y_train)
        y_pred_ens = ensemble.predict(X_test)
        ens_test_f1 = f1_score(y_test, y_pred_ens, average="weighted", zero_division=0)

        print(f"      CV F1: {ens_cv_mean:.3f} ± {ens_cv_std:.3f}  "
              f"| Test F1: {ens_test_f1:.3f}")
        print()
        print(classification_report(y_test, y_pred_ens, zero_division=0))

        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred_ens, labels=labels)
        save_confusion_matrix(cm, labels, "Ensemble", target_dir)

        # Save ensemble model
        model_path = target_dir / "ensemble_voting.joblib"
        joblib.dump(ensemble, model_path)
        print(f"  Saved: {model_path.name}")

        ens_result = {
            "model":      "Ensemble (soft voting)",
            "cv_f1_mean": round(float(ens_cv_mean), 4),
            "cv_f1_std":  round(float(ens_cv_std), 4),
            "test_f1":    round(float(ens_test_f1), 4),
        }
        results.append(ens_result)

    except Exception as e:
        print(f"      ERROR building ensemble: {e}")

    # ── Summary table ──
    if results:
        print(f"\n  {'─' * 52}")
        print(f"  Summary for '{target}':")
        print(f"  {'Model':<32} {'CV F1':>8} {'±':>6} {'Test F1':>10}")
        print(f"  {'─' * 52}")
        for r in sorted(results, key=lambda x: x["test_f1"], reverse=True):
            marker = " ◀ ensemble" if r["model"].startswith("Ensemble") else ""
            print(f"  {r['model']:<32} {r['cv_f1_mean']:>8.3f} {r['cv_f1_std']:>6.3f} "
                  f"{r['test_f1']:>10.3f}{marker}")

    # ── Save JSON log ──
    log = {
        "run":    run_meta,
        "target": target,
        "dataset_variant": dataset_variant,
        "data": {
            "n_total":   len(X),
            "n_train":   len(X_train),
            "n_test":    len(X_test),
            "test_size": TEST_SIZE,
            "class_dist": {str(k): int(v) for k, v in class_dist.items()},
            "prepared_dataset": str(snapshot_path),
        },
        "models": results,
    }
    log_path = target_dir / "ensemble_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n  Log saved to: {log_path}")

    return [dict(target=target, dataset_variant=dataset_variant, **r) for r in results]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Voting ensemble for support call classification."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Cleaned CSV file (dataset_clean.csv)")
    parser.add_argument("--target", "-t", default="call_purpose",
                        choices=TARGETS + ["all"],
                        help="Target column (default: call_purpose)")
    parser.add_argument("--output", "-o", default="models_ensemble",
                        help="Output directory (default: ./models_ensemble)")
    parser.add_argument("--sep", default=";",
                        help="CSV separator (default: ;)")
    parser.add_argument("--binary-input", default=None,
                        help="Optional second CSV for spam/non-spam evaluation. "
                             "All non-spam call_purpose labels collapse to 'non_spam'.")
    parser.add_argument("--sbert", action="store_true",
                        help="Add SBERT + SVM as a 4th base model (requires sentence-transformers)")
    parser.add_argument("--sbert-model",
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        help="SBERT model name (default: paraphrase-multilingual-MiniLM-L12-v2)")
    args = parser.parse_args()

    global SEP
    SEP = args.sep

    csv_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: '{csv_path}' not found.")
        return

    try:
        multiclass_df = load_training_frame(csv_path, sep=SEP)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    binary_path = Path(args.binary_input).resolve() if args.binary_input else None
    binary_df = None
    if binary_path is not None:
        if not binary_path.exists():
            print(f"Error: '{binary_path}' not found.")
            return
        try:
            binary_df = load_training_frame(binary_path, sep=SEP)
        except ValueError as exc:
            print(f"Error: {exc}")
            return

    if args.sbert:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            print(f"  SBERT enabled: {args.sbert_model}")
        except Exception as e:
            msg = str(e)
            hint = " (try: pip uninstall codecarbon -y)" if "CodeCarbon" in msg else ""
            print(f"  WARNING: sentence-transformers unavailable ({type(e).__name__}: {msg}){hint}. Skipping SBERT.")
            args.sbert = False

    run_meta = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "input_file":  str(csv_path),
        "output_dir":  str(output_dir),
        "use_sbert":   args.sbert,
        "sbert_model": args.sbert_model if args.sbert else None,
        "test_size":   TEST_SIZE,
        "random_state": 42,
    }

    targets = TARGETS if args.target == "all" else [args.target]
    all_results = []
    multiclass_output_dir = output_dir / "multiclass" if binary_df is not None else output_dir
    for target in targets:
        results = run_target(multiclass_df, target, multiclass_output_dir,
                             args.sbert, args.sbert_model,
                             {**run_meta, "dataset_variant": "multiclass"},
                             "multiclass")
        all_results.extend(results)

    if binary_df is not None and "call_purpose" in targets:
        results = run_target(binary_df, "call_purpose", output_dir / "binary_spam",
                             args.sbert, args.sbert_model,
                             {
                                 **run_meta,
                                 "input_file": str(binary_path),
                                 "dataset_variant": "binary_spam",
                             },
                             "binary_spam")
        all_results.extend(results)

    if all_results:
        summary_path = output_dir / "ensemble_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"run": run_meta, "results": all_results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n  Summary saved to: {summary_path}")

    print(f"\n{'═' * 60}")
    print(f"  Done. Results saved to: {output_dir}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
