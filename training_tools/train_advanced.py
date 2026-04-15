#!/usr/bin/env python3
"""
train_advanced.py — Полный пайплайн ML/DL классификации транскриптов звонков.

Секция 3.2  Базовые алгоритмы ML
    Векторизаторы : Bag-of-Words (CountVectorizer), TF-IDF, N-gram TF-IDF
    Классификаторы: Logistic Regression, SVM, Naive Bayes, Random Forest, Decision Tree

Секция 3.3  Нейросетевые модели
  3.3.1 Эмбеддинги : Word2Vec, Doc2Vec, fastText, SentenceTransformers (SBERT)
  3.3.2 Трансформеры: RuBERT, XLM-RoBERTa (fine-tuning)

Секция 3.4  Сравнительный анализ — таблица метрик + графики

Зависимости (устанавливать по мере необходимости):
    pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
    pip install gensim                                    # Word2Vec/Doc2Vec/fastText
    pip install sentence-transformers                     # SBERT
    pip install transformers torch accelerate             # RuBERT/XLM-RoBERTa
    # LLM API использует urllib из стандартной библиотеки — дополнительных пакетов не нужно

Запуск:
    # Только базовые модели (быстро, без GPU):
    python train_advanced.py -i dataset_clean.csv -g baseline

    # Базовые + эмбеддинги:
    python train_advanced.py -i dataset_clean.csv -g baseline,embeddings

    # Трансформеры с GPU + fp16 (≈2–3× быстрее):
    python train_advanced.py -i dataset_clean.csv -g baseline,rubert,xlmr --epochs 3 --fp16

    # Полная модель RuBERT (лучше качество, нужен GPU ≥ 8 GB):
    python train_advanced.py -i dataset_clean.csv -g rubert --fp16 \\
        --rubert-model DeepPavlov/rubert-base-cased --batch-size 32

    # bf16 + gradient accumulation (Ampere+, eff. batch = 16×4 = 64):
    python train_advanced.py -i dataset_clean.csv -g rubert,xlmr \\
        --bf16 --batch-size 16 --grad-accum 4

    # LLM через Ollama:
    python train_advanced.py -i dataset_clean.csv -g llm \\
        --llm-api-base http://localhost:11434/v1 --llm-model llama3

    # Всё сразу, все таргеты:
    python train_advanced.py -i dataset_clean.csv -g all -t all --fp16
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import argparse
import contextlib
import hashlib
import json
import random
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset_tools.dataset_variants import (
    load_training_frame,
    prepare_binary_spam_frame,
    prepare_multiclass_frame,
    save_prepared_dataset,
)
from training_tools.legacy_baseline import build_legacy_baseline_pipelines
from training_tools.tokenization_utils import (
    encode_text_batch,
    load_hf_tokenizer,
    save_inference_config,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ==============================================================================
# CONSTANTS
# ==============================================================================

TARGETS = ["call_purpose", "priority", "assig_group"]
SEP = ";"
MIN_SAMPLES_PER_CLASS = 5
TEST_SIZE = 0.2

RESOURCE_MAP = {
    "baseline":     "низкие",
    "legacy_baseline": "низкие",
    "embeddings":   "средние",
    "transformers": "высокие",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def print_gpu_info() -> None:
    """Выводит информацию о доступном GPU при запуске."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("  GPU: не обнаружен — используется CPU")
            return
        idx   = torch.cuda.current_device()
        name  = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        vram  = props.total_memory / 1024 ** 3
        print(f"  GPU : {name}")
        print(f"  VRAM: {vram:.1f} GB  |  CUDA {torch.version.cuda}"
              f"  |  PyTorch {torch.__version__}")
        # Совет зависит от объёма VRAM и поколения GPU
        cap = props.major  # compute capability major version
        if vram >= 12 and cap >= 9:  # Blackwell (9.x) или Ada (8.9)
            print("  Совет (Blackwell/Ada, ≥12 GB): "
                  "--bf16 --batch-size 32 --compile "
                  "--rubert-model DeepPavlov/rubert-base-cased")
        elif vram >= 12:             # Ampere/Turing с большим VRAM
            print("  Совет (≥12 GB VRAM): "
                  "--fp16 --batch-size 32 "
                  "--rubert-model DeepPavlov/rubert-base-cased")
        elif vram >= 8:
            print("  Совет (8–11 GB VRAM): --fp16 --batch-size 16 --grad-accum 2")
        else:
            print("  Совет (< 8 GB): --fp16 --batch-size 8 --grad-accum 4")
    except Exception:
        pass


# ==============================================================================
# DATA LOADING
# ==============================================================================

def prepare_dataset(df: pd.DataFrame, target: str, dataset_variant: str) -> pd.DataFrame:
    if dataset_variant == "binary_spam":
        return prepare_binary_spam_frame(
            df, target=target, min_samples_per_class=MIN_SAMPLES_PER_CLASS
        )
    return prepare_multiclass_frame(
        df, target=target, min_samples_per_class=MIN_SAMPLES_PER_CLASS
    )


def _cv_row_hashes(X_full, y_full) -> list[str]:
    hashes = []
    for text, label in zip(X_full, y_full):
        payload = f"{label}\x1f{text}".encode("utf-8", errors="ignore")
        hashes.append(hashlib.sha1(payload).hexdigest())
    return hashes


def resolve_cv_folds(
    X_full,
    y_full,
    cv: int,
    seed: int,
    output_dir: Path,
    manifest_path: Optional[Path] = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], Path, bool]:
    from sklearn.model_selection import StratifiedKFold

    if cv < 2:
        raise ValueError("resolve_cv_folds requires cv >= 2")

    X_list = list(X_full)
    y_list = list(y_full)
    if len(X_list) != len(y_list):
        raise ValueError("X_full and y_full must have the same length for CV.")

    resolved_path = Path(manifest_path).resolve() if manifest_path else (output_dir / f"fold_manifest_cv{cv}.json")
    current_hashes = _cv_row_hashes(X_list, y_list)
    current_labels = sorted(set(y_list))

    if resolved_path.exists():
        with open(resolved_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        manifest_cv = int(payload.get("cv", 0))
        manifest_rows = int(payload.get("n_samples", -1))
        manifest_hashes = payload.get("row_hashes")
        manifest_labels = payload.get("labels", [])
        if manifest_cv != cv:
            raise ValueError(
                f"Fold manifest '{resolved_path}' was built for cv={manifest_cv}, not cv={cv}."
            )
        if manifest_rows != len(X_list):
            raise ValueError(
                f"Fold manifest '{resolved_path}' expects {manifest_rows} rows, "
                f"but current dataset has {len(X_list)} rows."
            )
        if manifest_hashes and manifest_hashes != current_hashes:
            raise ValueError(
                f"Fold manifest '{resolved_path}' does not match the current prepared dataset."
            )
        if manifest_labels and sorted(manifest_labels) != current_labels:
            raise ValueError(
                f"Fold manifest '{resolved_path}' has a different label set than the current dataset."
            )

        folds = [
            (
                np.asarray(fold["train_indices"], dtype=int),
                np.asarray(fold["val_indices"], dtype=int),
            )
            for fold in payload.get("folds", [])
        ]
        if len(folds) != cv:
            raise ValueError(
                f"Fold manifest '{resolved_path}' contains {len(folds)} folds, expected {cv}."
            )
        return folds, resolved_path, True

    y_arr = np.asarray(y_list, dtype=object)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    folds = [
        (train_idx.astype(int), val_idx.astype(int))
        for train_idx, val_idx in skf.split(np.zeros(len(y_arr)), y_arr)
    ]
    payload = {
        "cv": cv,
        "seed": seed,
        "n_samples": len(X_list),
        "labels": current_labels,
        "row_hashes": current_hashes,
        "folds": [
            {
                "fold": fold_idx + 1,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
            }
            for fold_idx, (train_idx, val_idx) in enumerate(folds)
        ],
    }
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return folds, resolved_path, False


# ==============================================================================
# RESULT STORE  (сбор и отображение результатов)
# ==============================================================================

class ResultStore:
    """Накапливает результаты всех моделей для итогового сравнения (раздел 3.4)."""

    def __init__(self):
        self.records: list[dict] = []

    def record(
        self,
        name: str,
        group: str,
        y_test,
        y_pred,
        train_sec: float,
        infer_ms_total: float,
        notes: str = "",
    ) -> float:
        n = len(y_test)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        self.records.append({
            "model":              name,
            "group":              group,
            "f1_weighted":        round(f1, 4),
            "precision":          round(prec, 4),
            "recall":             round(rec, 4),
            "train_sec":          round(train_sec, 1),
            "infer_ms_per_sample": round(infer_ms_total / max(n, 1), 3),
            "resource":           RESOURCE_MAP.get(group, "—"),
            "notes":              notes,
            "_report":            classification_report(y_test, y_pred, zero_division=0, output_dict=True),
            "_y_test":            list(y_test),
            "_y_pred":            list(y_pred),
        })
        return f1

    def summary_df(self) -> pd.DataFrame:
        rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.records]
        return (
            pd.DataFrame(rows)
            .sort_values("f1_weighted", ascending=False)
            .reset_index(drop=True)
        )

    def print_summary(self):
        df = self.summary_df()
        print(f"\n  {'Модель':<40} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Train':>8} {'ms/сэмпл':>10}  Ресурсы")
        print(f"  {'─' * 95}")
        for _, row in df.iterrows():
            print(
                f"  {row['model']:<40} {row['f1_weighted']:>6.3f} "
                f"{row['precision']:>6.3f} {row['recall']:>6.3f} "
                f"{row['train_sec']:>7.1f}s {row['infer_ms_per_sample']:>9.2f}ms"
                f"  {row['resource']}"
            )


# ==============================================================================
# VISUALIZATION HELPERS
# ==============================================================================

def save_confusion_matrix(y_test, y_pred, model_name: str, output_dir: Path):
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(model_name)
    plt.tight_layout()
    safe = (model_name.replace(" ", "_").replace("/", "-")
            .replace("+", "plus").replace("(", "").replace(")", ""))
    path = output_dir / f"cm_{safe}.png"
    plt.savefig(path, dpi=110)
    plt.close()


def save_comparison_chart(store: ResultStore, output_dir: Path, target: str):
    df = store.summary_df()
    if df.empty:
        return

    COLORS = {
        "baseline":     "#4C72B0",
        "embeddings":   "#55A868",
        "transformers": "#C44E52",
        "llm":          "#8172B2",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(df) * 0.5 + 2)))
    fig.suptitle(f"Сравнение моделей — {target}", fontsize=13, fontweight="bold")

    # ── Left: F1 bar chart ──────────────────────────────────────────────────
    ax = axes[0]
    bar_colors = [COLORS.get(r, "#777777") for r in df["group"]]
    bars = ax.barh(df["model"], df["f1_weighted"], color=bar_colors, height=0.6)
    ax.set_xlabel("F1 (weighted)")
    ax.set_title("F1-score по моделям")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, df["f1_weighted"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.invert_yaxis()

    # ── Right: F1 vs speed scatter ──────────────────────────────────────────
    ax2 = axes[1]
    for _, row in df.iterrows():
        color = COLORS.get(row["group"], "#777777")
        ax2.scatter(row["infer_ms_per_sample"], row["f1_weighted"],
                    c=color, s=90, zorder=3)
        ax2.annotate(
            row["model"],
            (row["infer_ms_per_sample"], row["f1_weighted"]),
            fontsize=7, xytext=(4, 3), textcoords="offset points",
        )
    ax2.set_xlabel("Время инференса (мс / образец)")
    ax2.set_ylabel("F1 (weighted)")
    ax2.set_title("Качество vs Скорость")
    ax2.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=v, label=k)
        for k, v in COLORS.items()
        if k in df["group"].values
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    path = output_dir / f"comparison_{target}.png"
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"  График сравнения: {path.name}")


def _safe_artifact_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "-")
        .replace("+", "plus")
        .replace("(", "")
        .replace(")", "")
    )


def save_epoch_history(
    history: list[dict],
    output_dir: Path,
    model_name: str,
    meta: dict,
) -> tuple[Path, Path]:
    """Save per-epoch training history as JSON and CSV."""
    hist_dir = output_dir / "_epoch_metrics"
    hist_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_artifact_name(model_name)
    json_path = hist_dir / f"{safe_name}.json"
    csv_path = hist_dir / f"{safe_name}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "history": history}, f, ensure_ascii=False, indent=2)

    pd.DataFrame(history).to_csv(csv_path, index=False, encoding="utf-8")
    return json_path, csv_path


def augment_training_split(
    X_train,
    y_train,
    method: str,
    target: str,
    output_dir: Path,
    seed: int = 42,
):
    """
    Apply train-only augmentation and save an audit trail.

    Current methods:
      - none: no augmentation
      - oversample: duplicate minority-class samples up to the majority class size
    """
    X_series = pd.Series(list(X_train), name="text").reset_index(drop=True)
    y_series = pd.Series(list(y_train), name=target).reset_index(drop=True)

    if method == "none":
        return X_series, y_series, None

    if method != "oversample":
        raise ValueError(f"Unsupported train augmentation method: {method}")

    train_df = pd.DataFrame({
        "text": X_series,
        target: y_series,
        "_aug_source": "original",
    })
    before_counts = train_df[target].value_counts().sort_index()
    max_count = int(before_counts.max())

    parts = []
    for class_name, class_df in train_df.groupby(target, sort=False):
        parts.append(class_df)
        deficit = max_count - len(class_df)
        if deficit <= 0:
            continue
        extra = class_df.sample(
            n=deficit,
            replace=True,
            random_state=seed,
        ).copy()
        extra["_aug_source"] = "oversample"
        parts.append(extra)

    augmented_df = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    after_counts = augmented_df[target].value_counts().sort_index()

    aug_dir = output_dir / "_augmentation"
    aug_dir.mkdir(parents=True, exist_ok=True)
    csv_path = aug_dir / f"train_augmented_{target}.csv"
    summary_path = aug_dir / f"train_augmented_{target}.json"
    augmented_df.to_csv(csv_path, index=False, encoding="utf-8")

    summary = {
        "method": method,
        "target": target,
        "seed": seed,
        "n_train_before": int(len(train_df)),
        "n_train_after": int(len(augmented_df)),
        "before_class_counts": {str(k): int(v) for k, v in before_counts.items()},
        "after_class_counts": {str(k): int(v) for k, v in after_counts.items()},
        "n_synthetic_rows": int((augmented_df["_aug_source"] != "original").sum()),
        "csv_path": str(csv_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  Аугментация train: {method}")
    print(f"  Train до : {len(train_df)}")
    print(f"  Train после: {len(augmented_df)}")
    print("  Классы до аугментации:")
    for cls, cnt in before_counts.items():
        print(f"    {cls:<35} {int(cnt):>4}")
    print("  Классы после аугментации:")
    for cls, cnt in after_counts.items():
        print(f"    {cls:<35} {int(cnt):>4}")
    print(f"  Сохранено: {csv_path}")

    return (
        augmented_df["text"].reset_index(drop=True),
        augmented_df[target].reset_index(drop=True),
        summary,
    )


# ==============================================================================
# SECTION 3.2 — БАЗОВЫЕ АЛГОРИТМЫ КЛАССИФИКАЦИИ
# ==============================================================================

def build_baseline_pipelines() -> dict[str, Pipeline]:
    """
    3.2.1 Векторизация: BoW, TF-IDF, N-gram TF-IDF.
    3.2.2 Классификаторы: LogReg, SVM, NaiveBayes, RandomForest, DecisionTree.

    Итого 15 пайплайнов для всестороннего baseline-сравнения.
    """
    bow_kw    = dict(max_features=15_000, min_df=1)
    tfidf_kw  = dict(ngram_range=(1, 2), max_features=15_000,
                     sublinear_tf=True, min_df=1)
    ngram_kw  = dict(ngram_range=(1, 3), max_features=20_000,
                     sublinear_tf=True, min_df=1)

    lr_kw  = dict(max_iter=1000, class_weight="balanced", C=1.0,
                  solver="lbfgs")
    svm_kw = dict(max_iter=3000, class_weight="balanced", C=1.0)
    rf_kw  = dict(n_estimators=200, class_weight="balanced",
                  random_state=42, n_jobs=-1)
    dt_kw  = dict(
        class_weight="balanced",
        random_state=42,
        max_depth=40,
        min_samples_leaf=2,
    )

    return {
        # ── BoW ────────────────────────────────────────────────────────────
        "BoW + LogReg": Pipeline([
            ("vec", CountVectorizer(**bow_kw)),
            ("clf", LogisticRegression(**lr_kw)),
        ]),
        "BoW + SVM": Pipeline([
            ("vec", CountVectorizer(**bow_kw)),
            ("clf", LinearSVC(**svm_kw)),
        ]),
        "BoW + NaiveBayes": Pipeline([
            ("vec", CountVectorizer(**bow_kw)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "BoW + RandomForest": Pipeline([
            ("vec", CountVectorizer(**bow_kw)),
            ("clf", RandomForestClassifier(**rf_kw)),
        ]),
        "BoW + DecisionTree": Pipeline([
            ("vec", CountVectorizer(**bow_kw)),
            ("clf", DecisionTreeClassifier(**dt_kw)),
        ]),
        # ── TF-IDF ─────────────────────────────────────────────────────────
        "TF-IDF + LogReg": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_kw)),
            ("clf", LogisticRegression(**lr_kw)),
        ]),
        "TF-IDF + SVM": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_kw)),
            ("clf", LinearSVC(**svm_kw)),
        ]),
        "TF-IDF + Calibrated SVM": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_kw)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(**svm_kw),
                cv=3,
            )),
        ]),
        "TF-IDF + NaiveBayes": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_kw)),
            # ComplementNB лучше MNB для несбалансированных данных с TF-IDF
            ("clf", ComplementNB(alpha=0.1)),
        ]),
        "TF-IDF + RandomForest": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_kw)),
            ("clf", RandomForestClassifier(**rf_kw)),
        ]),
        "TF-IDF + DecisionTree": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_kw)),
            ("clf", DecisionTreeClassifier(**dt_kw)),
        ]),
        # ── N-gram TF-IDF ──────────────────────────────────────────────────
        "N-gram(1-3) + SVM": Pipeline([
            ("vec", TfidfVectorizer(**ngram_kw)),
            ("clf", LinearSVC(**svm_kw)),
        ]),
        "N-gram(1-3) + Calibrated SVM": Pipeline([
            ("vec", TfidfVectorizer(**ngram_kw)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(**svm_kw),
                cv=3,
            )),
        ]),
        "N-gram(1-3) + LogReg": Pipeline([
            ("vec", TfidfVectorizer(**ngram_kw)),
            ("clf", LogisticRegression(**lr_kw)),
        ]),
        "N-gram(1-3) + DecisionTree": Pipeline([
            ("vec", TfidfVectorizer(**ngram_kw)),
            ("clf", DecisionTreeClassifier(**dt_kw)),
        ]),
    }


def run_baseline_models(
    X_train, y_train, X_test, y_test,
    store: ResultStore, output_dir: Path,
    cv: int = 0,
    cv_only: bool = False,
    seed: int = 42,
    X_full=None,
    y_full=None,
    cv_manifest_path: Optional[Path] = None,
):
    """
    3.2 Базовые алгоритмы классификации.
    Precision / Recall / F1 / матрица ошибок для каждого пайплайна.
    """
    print(f"\n{'═' * 60}")
    print("  3.2  Базовые алгоритмы (BoW / TF-IDF / N-gram)")
    print(f"{'═' * 60}")

    for name, pipeline in build_baseline_pipelines().items():
        print(f"\n  [{name}]")
        try:
            # ── Кросс-валидация (опционально) ─────────────────────────────────
            if cv >= 2:
                X_all = np.array(
                    list(X_full) if X_full is not None else list(X_train) + list(X_test),
                    dtype=object,
                )
                y_all = np.array(
                    list(y_full) if y_full is not None else list(y_train) + list(y_test),
                    dtype=object,
                )
                folds, resolved_manifest_path, manifest_loaded = resolve_cv_folds(
                    X_all,
                    y_all,
                    cv=cv,
                    seed=seed,
                    output_dir=output_dir,
                    manifest_path=cv_manifest_path,
                )
                fold_scores = []
                oof_pred = np.empty(len(y_all), dtype=object)
                total_train_sec = 0.0
                total_infer_ms = 0.0

                print(f"    CV {cv}-fold")
                print(
                    f"    Fold manifest: {resolved_manifest_path} "
                    f"({'loaded' if manifest_loaded else 'saved'})"
                )
                for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
                    fold_pipeline = clone(pipeline)
                    t0 = time.perf_counter()
                    fold_pipeline.fit(X_all[tr_idx], y_all[tr_idx])
                    total_train_sec += time.perf_counter() - t0

                    t1 = time.perf_counter()
                    fold_pred = fold_pipeline.predict(X_all[val_idx])
                    total_infer_ms += (time.perf_counter() - t1) * 1000

                    oof_pred[val_idx] = fold_pred
                    fold_f1 = f1_score(y_all[val_idx], fold_pred, average="weighted", zero_division=0)
                    fold_scores.append(fold_f1)
                    print(f"      Фолд {fold_idx}/{cv}: F1 = {fold_f1:.3f}")

                print(f"    CV F1: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}  "
                      f"(folds: {', '.join(f'{s:.3f}' for s in fold_scores)})")

                if cv_only:
                    f1 = store.record(
                        f"{name} [CV{cv}]",
                        "baseline",
                        y_all.tolist(),
                        oof_pred.tolist(),
                        total_train_sec,
                        total_infer_ms,
                        notes=f"cv_only={cv}; folds={','.join(f'{s:.4f}' for s in fold_scores)}",
                    )
                    print(f"    CV-only F1: {f1:.3f}  |  Train(sum): {total_train_sec:.1f}s")
                    print(classification_report(y_all.tolist(), oof_pred.tolist(), zero_division=0))
                    save_confusion_matrix(y_all.tolist(), oof_pred.tolist(), f"{name}_CV{cv}", output_dir)
                    continue

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

            safe = (name.replace(" ", "_").replace("+", "plus")
                    .replace("/", "-").replace("(", "").replace(")", ""))
            joblib.dump(pipeline, output_dir / f"{safe}.joblib")

        except Exception as exc:
            print(f"    ОШИБКА: {exc}")


def run_legacy_baseline_models(
    X_train, y_train, X_test, y_test,
    store: ResultStore, output_dir: Path,
):
    """
    Legacy classical models kept for compatibility with the old train.py script.
    """
    print(f"\n{'═' * 60}")
    print("  3.2L  Legacy baseline (train.py compatibility)")
    print(f"{'═' * 60}")

    for name, pipeline in build_legacy_baseline_pipelines().items():
        print(f"\n  [{name}]")
        try:
            t0 = time.perf_counter()
            if hasattr(pipeline, "best_estimator_") or pipeline.__class__.__name__ == "GridSearchCV":
                pipeline.fit(X_train, y_train)
                best_score = pipeline.best_score_
                print(f"    Best params : {pipeline.best_params_}")
                print(f"    Best CV F1  (train, weighted): {best_score:.3f}")
                model_to_save = pipeline.best_estimator_
            else:
                pipeline.fit(X_train, y_train)
                model_to_save = pipeline
            train_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = model_to_save.predict(X_test)
            infer_ms = (time.perf_counter() - t1) * 1000

            f1 = store.record(name, "legacy_baseline", y_test, y_pred, train_sec, infer_ms)
            print(f"    Hold-out F1: {f1:.3f}  |  Train: {train_sec:.1f}s")
            print(classification_report(y_test, y_pred, zero_division=0))

            save_confusion_matrix(y_test, y_pred, name, output_dir)
            safe = _safe_artifact_name(name)
            joblib.dump(model_to_save, output_dir / f"{safe}.joblib")
        except Exception as exc:
            print(f"    ОШИБКА: {exc}")



# ==============================================================================
# SECTION 3.3.1 — ЭМБЕДДИНГИ СЛОВ И ПРЕДЛОЖЕНИЙ
# ==============================================================================

class MeanEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Усредняет векторы слов из Word2Vec или fastText (gensim).
    Обучается на тренировочном корпусе «с нуля».
    Для OOV: Word2Vec возвращает нулевой вектор; fastText использует субслова.
    """

    def __init__(
        self,
        model_type: str = "word2vec",
        vector_size: int = 150,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 15,
    ):
        self.model_type = model_type
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model_ = None

    @staticmethod
    def _tokenize(texts):
        return [str(t).lower().split() for t in texts]

    def fit(self, X, y=None):
        sentences = self._tokenize(X)
        if self.model_type == "fasttext":
            from gensim.models import FastText
            self.model_ = FastText(
                sentences=sentences, vector_size=self.vector_size,
                window=self.window, min_count=self.min_count,
                epochs=self.epochs, workers=4, seed=42,
            )
        else:
            from gensim.models import Word2Vec
            self.model_ = Word2Vec(
                sentences=sentences, vector_size=self.vector_size,
                window=self.window, min_count=self.min_count,
                epochs=self.epochs, workers=4, seed=42,
            )
        return self

    def transform(self, X):
        result = []
        for text in X:
            words = str(text).lower().split()
            vecs = [self.model_.wv[w] for w in words if w in self.model_.wv]
            result.append(
                np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)
            )
        return np.array(result, dtype=np.float32)


class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Doc2Vec (Paragraph Vectors, Le & Mikolov 2014).
    Каждому документу обучается отдельный вектор; для инференса используется
    infer_vector (итеративная оптимизация).
    """

    def __init__(
        self,
        vector_size: int = 150,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 20,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model_ = None

    def fit(self, X, y=None):
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        corpus = [
            TaggedDocument(str(t).lower().split(), [i])
            for i, t in enumerate(X)
        ]
        self.model_ = Doc2Vec(
            corpus, vector_size=self.vector_size,
            window=self.window, min_count=self.min_count,
            epochs=self.epochs, workers=4, seed=42,
        )
        return self

    def transform(self, X):
        return np.array([
            self.model_.infer_vector(str(t).lower().split(), epochs=self.epochs)
            for t in X
        ], dtype=np.float32)


class SBERTTransformer(BaseEstimator, TransformerMixin):
    """
    SentenceTransformers (SBERT) — контекстуальные эмбеддинги предложений.
    Использует предобученные веса; дополнительное обучение не требуется.
    Рекомендуемая модель: paraphrase-multilingual-MiniLM-L12-v2
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_ = None

    def fit(self, X, y=None):
        from sentence_transformers import SentenceTransformer
        self.model_ = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        return self.model_.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


def _pkg_available(name: str) -> bool:
    import importlib
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def build_embedding_pipelines(sbert_model: str) -> dict[str, dict]:
    """3.3.1: Пайплайны с векторными представлениями."""
    lr_kw  = dict(max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs")
    svm_kw = dict(max_iter=3000, class_weight="balanced", C=1.0)

    return {
        "Word2Vec + LogReg": {
            "pkg": "gensim",
            "pipeline": Pipeline([
                ("emb", MeanEmbeddingTransformer(model_type="word2vec")),
                ("clf", LogisticRegression(**lr_kw)),
            ]),
        },
        "Doc2Vec + LogReg": {
            "pkg": "gensim",
            "pipeline": Pipeline([
                ("emb", Doc2VecTransformer()),
                ("clf", LogisticRegression(**lr_kw)),
            ]),
        },
        "fastText + LogReg": {
            "pkg": "gensim",
            "pipeline": Pipeline([
                ("emb", MeanEmbeddingTransformer(model_type="fasttext")),
                ("clf", LogisticRegression(**lr_kw)),
            ]),
        },
        "fastText + SVM": {
            "pkg": "gensim",
            "pipeline": Pipeline([
                ("emb", MeanEmbeddingTransformer(model_type="fasttext")),
                ("clf", LinearSVC(**svm_kw)),
            ]),
        },
        "SBERT + LogReg": {
            "pkg": "sentence_transformers",
            "pipeline": Pipeline([
                ("emb", SBERTTransformer(model_name=sbert_model)),
                ("clf", LogisticRegression(**lr_kw)),
            ]),
        },
        "SBERT + SVM": {
            "pkg": "sentence_transformers",
            "pipeline": Pipeline([
                ("emb", SBERTTransformer(model_name=sbert_model)),
                ("clf", LinearSVC(**svm_kw)),
            ]),
        },
    }


def run_embedding_models(
    X_train, y_train, X_test, y_test,
    store: ResultStore, output_dir: Path,
    sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
):
    """
    3.3.1 Эмбеддинги слов и предложений.
    Word2Vec, Doc2Vec, fastText (gensim) + SBERT (sentence-transformers).
    """
    print(f"\n{'═' * 60}")
    print("  3.3.1  Эмбеддинги (Word2Vec / Doc2Vec / fastText / SBERT)")
    print(f"{'═' * 60}")

    for name, spec in build_embedding_pipelines(sbert_model).items():
        pkg = spec["pkg"]
        if not _pkg_available(pkg):
            print(f"\n  [{name}]  ПРОПУСК — пакет '{pkg}' не установлен")
            print(f"    Установка: pip install {pkg.replace('_', '-')}")
            continue

        print(f"\n  [{name}]")
        try:
            t0 = time.perf_counter()
            spec["pipeline"].fit(X_train, y_train)
            train_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = spec["pipeline"].predict(X_test)
            infer_ms = (time.perf_counter() - t1) * 1000

            f1 = store.record(name, "embeddings", y_test, y_pred, train_sec, infer_ms)
            print(f"    F1: {f1:.3f}  |  Train: {train_sec:.1f}s")
            print(classification_report(y_test, y_pred, zero_division=0))

            save_confusion_matrix(y_test, y_pred, name, output_dir)

            safe = (name.replace(" ", "_").replace("+", "plus")
                    .replace("/", "-").replace("(", "").replace(")", ""))
            joblib.dump(spec["pipeline"], output_dir / f"{safe}.joblib")

        except Exception as exc:
            print(f"    ОШИБКА: {exc}")


# ==============================================================================
# SECTION 3.3.2 — ТРАНСФОРМЕРЫ (FINE-TUNING)
# ==============================================================================

def _finetune_transformer(
    model_name: str,
    friendly_name: str,
    X_train, y_train, X_test, y_test,
    store: ResultStore, output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 256,
    truncation_strategy: str = "head",
    lr: float = 2e-5,
    fp16: bool = False,
    bf16: bool = False,
    grad_accum: int = 1,
    compile_model: bool = False,
    freeze_layers: int = 0,
    label_smoothing: float = 0.0,
    early_stopping: int = 0,
    early_stopping_metric: str = "f1",
    save_model: bool = True,
    silent: bool = False,
    return_details: bool = False,
    seed: int = 42,
):
    """
    Fine-tuning AutoModelForSequenceClassification (RuBERT или XLM-RoBERTa).

    Особенности:
    - Взвешенная кросс-энтропия для несбалансированных классов.
    - Линейный warmup (10 % шагов) + линейный decay.
    - Динамический padding (per-batch) для ускорения.
    - Mixed precision: fp16 (GradScaler) или bf16 (Ampere+/Blackwell).
    - Gradient accumulation для эффективного увеличения batch size.
    - torch.compile() для дополнительного ускорения (PyTorch ≥ 2.0).
    - Сохранение лучших весов и LabelEncoder.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset as TorchDataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as e:
        print(f"\n  [{friendly_name}]  ПРОПУСК — {e}")
        print("    Установка: pip install transformers torch accelerate")
        return

    set_global_seed(seed)

    def _load_tokenizer(model_ref: str):
        return load_hf_tokenizer(model_ref)

    print(f"\n  [{friendly_name}]  модель: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Устройство: {device}")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(f"    GPU: {torch.cuda.get_device_name(device)}  |  VRAM: {vram:.1f} GB")
        eff_batch = batch_size * grad_accum
        prec_str  = "bf16" if bf16 else ("fp16" if fp16 else "fp32")
        print(f"    Precision: {prec_str}  |  "
              f"batch={batch_size}  ×  grad_accum={grad_accum}  →  eff_batch={eff_batch}")

    # ── Label encoding ────────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(list(y_train) + list(y_test))
    y_tr = le.transform(y_train)
    y_te = le.transform(y_test)
    num_labels = len(le.classes_)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    try:
        tokenizer = _load_tokenizer(model_name)
    except Exception as e:
        print(f"\n  [{friendly_name}]  ПРОПУСК — не удалось загрузить токенизатор: {e}")
        print(f"    Проверьте правильность model id: {model_name}")
        return

    # ── Dataset с динамическим padding (collate_fn) ────────────────────────
    class TextDataset(TorchDataset):
        def __init__(self, texts, labels):
            self.texts  = list(texts)
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    def collate_fn(batch):
        texts, labels = zip(*batch)
        enc = encode_text_batch(
            tokenizer,
            list(texts),
            max_length=max_length,
            truncation_strategy=truncation_strategy,
        )
        return enc, torch.tensor(labels, dtype=torch.long)

    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        TextDataset(X_train, y_tr), batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=2, pin_memory=use_pin, persistent_workers=True,
    )
    test_loader = DataLoader(
        TextDataset(X_test, y_te), batch_size=batch_size * 2,
        collate_fn=collate_fn,
        num_workers=2, pin_memory=use_pin, persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        ).to(device)
    except Exception as e:
        print(f"\n  [{friendly_name}]  ПРОПУСК — не удалось загрузить модель: {e}")
        print(f"    Проверьте правильность model id: {model_name}")
        return

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # ── Layer freezing (техника для малых датасетов) ──────────────────────────
    if freeze_layers > 0:
        # Заморозить embeddings
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        # Заморозить первые N слоёв энкодера
        enc_layers = model.base_model.encoder.layer
        total_layers = len(enc_layers)
        # Всегда оставляем минимум 2 верхних слоя обучаемыми
        n_freeze = min(freeze_layers, max(0, total_layers - 2))
        if n_freeze < freeze_layers:
            print(f"    ВНИМАНИЕ: модель имеет {total_layers} слоёв, "
                  f"заморозка скорректирована: {freeze_layers} → {n_freeze} "
                  f"(оставлено минимум 2 обучаемых слоя)")
        for layer in enc_layers[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"    Заморожено: embeddings + {n_freeze}/{total_layers} слоёв  "
              f"| Обучаемых параметров: {trainable:,} / {total:,}")

    # ── torch.compile() (PyTorch ≥ 2.0, +15–30 % на Ampere/Blackwell) ────────
    if compile_model:
        if not hasattr(torch, "compile"):
            print("    ПРЕДУПРЕЖДЕНИЕ: torch.compile() требует PyTorch ≥ 2.0, пропуск")
        else:
            # Inductor не поддерживает SM >= 120 (Blackwell consumer, RTX 50xx) в текущих билдах
            _cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            _sm = _cap[0] * 10 + _cap[1]  # например 8.9 → 89, 12.0 → 120
            if _sm >= 120:
                print(f"    ПРЕДУПРЕЖДЕНИЕ: torch.compile(inductor) не поддерживает "
                      f"SM{_sm} (Blackwell consumer) в PyTorch {torch.__version__}, пропуск. "
                      "Запускайте без --compile.")
            else:
                print("    torch.compile() — компиляция графа (первый батч медленнее)...")
                model = torch.compile(model)

    # ── Weighted loss + label smoothing ───────────────────────────────────────
    cw = compute_class_weight("balanced", classes=np.arange(num_labels), y=y_tr)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(device),
        label_smoothing=label_smoothing,
    )
    if label_smoothing > 0:
        print(f"    Label smoothing: {label_smoothing}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Число оптимизаторных шагов с учётом gradient accumulation
    opt_steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps  = opt_steps_per_epoch * epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Mixed precision ────────────────────────────────────────────────────────
    use_amp   = (fp16 or bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    # GradScaler нужен только для fp16 (bf16 не требует масштабирования)
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    best_model_state = None
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch_history = []
    if early_stopping > 0:
        print(f"    Early stopping: patience={early_stopping} эпох "
              f"| metric={early_stopping_metric}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (enc, labels) in enumerate(tqdm(
            train_loader, desc=f"    Epoch {epoch + 1}/{epochs}", leave=False
        )):
            enc    = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(device, non_blocking=True)

            # autocast: включается только при use_amp и только для CUDA
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                outputs = model(**enc)
                # Делим loss на grad_accum для корректного среднего за accum-шаг
                loss = loss_fn(outputs.logits, labels) / grad_accum

            if fp16 and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * grad_accum

            is_last = (step + 1) == len(train_loader)
            if (step + 1) % grad_accum == 0 or is_last:
                if fp16 and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # Quick val on test to track best epoch
        model.eval()
        preds_epoch = []
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for enc, labels in test_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = labels.to(device)
                out = model(**enc)
                val_loss_total += loss_fn(out.logits, labels).item()
                val_batches += 1
                preds_epoch.extend(torch.argmax(out.logits, 1).cpu().tolist())
        epoch_val_loss = val_loss_total / max(val_batches, 1)
        epoch_f1 = f1_score(y_te, preds_epoch, average="weighted", zero_division=0)
        print(f"    Epoch {epoch + 1}: loss={avg_loss:.4f}  "
              f"val_loss={epoch_val_loss:.4f}  val_F1={epoch_f1:.3f}")

        if early_stopping_metric == "loss":
            improved = epoch_val_loss < best_loss
        else:
            improved = epoch_f1 > best_f1

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(float(avg_loss), 6),
            "val_loss": round(float(epoch_val_loss), 6),
            "val_f1_weighted": round(float(epoch_f1), 6),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 12),
            "improved": bool(improved),
            "patience_counter": int(patience_counter),
            "early_stopping_metric": early_stopping_metric,
        })

        if improved:
            best_f1 = epoch_f1
            best_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            if early_stopping > 0:
                patience_counter += 1
                epoch_history[-1]["patience_counter"] = int(patience_counter)
                if patience_counter >= early_stopping:
                    if early_stopping_metric == "loss":
                        best_metric_msg = f"лучший val_loss={best_loss:.4f}"
                    else:
                        best_metric_msg = f"лучший val_F1={best_f1:.3f}"
                    print(f"    Early stopping: нет улучшения {early_stopping} эпох подряд, "
                          f"{best_metric_msg} (epoch {epoch + 1 - early_stopping})")
                    break

    train_sec = time.perf_counter() - t0

    # ── Load best checkpoint ──────────────────────────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    all_preds = []
    all_probs = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for enc, _ in test_loader:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                out = model(**enc)
            probs = torch.softmax(out.logits, dim=1).float().cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(np.argmax(probs, axis=1).tolist())
    infer_ms = (time.perf_counter() - t1) * 1000

    y_pred      = le.inverse_transform(np.array(all_preds))
    y_test_orig = list(y_test)
    proba = np.vstack(all_probs) if all_probs else np.empty((0, num_labels), dtype=float)

    if silent:
        f1 = f1_score(y_test_orig, y_pred, average="weighted", zero_division=0)
    else:
        f1 = store.record(
            friendly_name, "transformers", y_test_orig, y_pred,
            train_sec, infer_ms, notes=model_name,
        )

    history_meta = {
        "model_name": model_name,
        "friendly_name": friendly_name,
        "n_train": len(X_train),
        "n_val": len(X_test),
        "epochs_requested": epochs,
        "epochs_completed": len(epoch_history),
        "best_epoch": best_epoch,
        "best_val_f1": round(float(best_f1), 6),
        "best_val_loss": round(float(best_loss), 6),
        "early_stopping": early_stopping,
        "early_stopping_metric": early_stopping_metric,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "max_length": max_length,
        "truncation_strategy": truncation_strategy,
        "learning_rate": lr,
        "fp16": fp16,
        "bf16": bf16,
        "freeze_layers": freeze_layers,
        "label_smoothing": label_smoothing,
        "train_sec": round(float(train_sec), 3),
        "final_test_f1": round(float(f1), 6),
    }
    hist_json_path, hist_csv_path = save_epoch_history(
        epoch_history, output_dir, friendly_name, history_meta
    )

    if not silent:
        print(f"    F1: {f1:.3f}  |  Train: {train_sec:.1f}s")
        print(classification_report(y_test_orig, y_pred, zero_division=0))
        save_confusion_matrix(y_test_orig, y_pred, friendly_name, output_dir)
        print(f"    Epoch metrics: {hist_csv_path.name}, {hist_json_path.name}")

    # ── Save model ────────────────────────────────────────────────────────────
    if save_model and not silent:
        safe = _safe_artifact_name(friendly_name)
        model_out = output_dir / safe
        model_out.mkdir(exist_ok=True)
        model.save_pretrained(str(model_out))
        tokenizer.save_pretrained(str(model_out))
        joblib.dump(le, model_out / "label_encoder.joblib")
        save_inference_config(model_out, max_length, truncation_strategy)
        print(f"    Сохранено: {model_out.name}/")

    if return_details:
        return {
            "f1": float(f1),
            "y_true": list(y_test_orig),
            "y_pred": list(y_pred),
            "proba": proba.tolist(),
            "classes": list(le.classes_),
            "train_sec": float(train_sec),
            "infer_ms": float(infer_ms),
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_f1),
            "best_val_loss": float(best_loss),
        }

    return f1


def run_transformer_models(
    X_train, y_train, X_test, y_test,
    store: ResultStore, output_dir: Path,
    rubert_model: str = "cointegrated/rubert-tiny2",
    xlmr_model:   str = "xlm-roberta-base",
    epochs: int = 3,
    batch_size: int = 16,
    groups: set = None,
    fp16: bool = False,
    bf16: bool = False,
    grad_accum: int = 1,
    compile_model: bool = False,
    lr: float = 2e-5,
    freeze_layers: int = 0,
    label_smoothing: float = 0.0,
    early_stopping: int = 0,
    early_stopping_metric: str = "f1",
    extra_models: list = None,
    max_length: int = 256,
    truncation_strategy: str = "head",
    cv: int = 0,
    cv_only: bool = False,
    seed: int = 42,
    X_full=None,
    y_full=None,
    cv_manifest_path: Optional[Path] = None,
):
    """
    3.3.2 Модели на основе трансформеров.
    Fine-tuning с учётом дисбаланса классов и линейным lr-расписанием.
    """
    print(f"\n{'═' * 60}")
    print("  3.3.2  Трансформеры (RuBERT / XLM-RoBERTa fine-tuning)")
    print(f"{'═' * 60}")

    run_all = groups is None or "all" in groups or "transformers" in groups

    models = []
    if run_all or "rubert" in (groups or set()):
        models.append((rubert_model, "RuBERT"))
    if run_all or "xlmr" in (groups or set()):
        models.append((xlmr_model, "XLM-RoBERTa"))

    # Дополнительные модели из --extra-models
    if extra_models:
        for entry in extra_models:
            entry = entry.strip()
            if not entry:
                continue
            # Поддержка формата "model_id:Название" или просто "model_id"
            if ":" in entry:
                m_id, m_name = entry.split(":", 1)
            else:
                m_name = entry.split("/")[-1]  # короткое имя из HF path
            models.append((entry.split(":")[0].strip(), m_name.strip()))

    # Общие kwargs для _finetune_transformer
    _ft_kwargs = dict(
        store=store, output_dir=output_dir,
        epochs=epochs, batch_size=batch_size,
        fp16=fp16, bf16=bf16, grad_accum=grad_accum,
        compile_model=compile_model, lr=lr,
        freeze_layers=freeze_layers, label_smoothing=label_smoothing,
        early_stopping=early_stopping,
        early_stopping_metric=early_stopping_metric,
        max_length=max_length,
        truncation_strategy=truncation_strategy,
        seed=seed,
    )

    for model_id, name in models:
        try:
            # ── Кросс-валидация ───────────────────────────────────────────────
            if cv >= 2:
                X_all = np.array(list(X_full) if X_full is not None else list(X_train) + list(X_test))
                y_all = np.array(list(y_full) if y_full is not None else list(y_train) + list(y_test))
                folds, resolved_manifest_path, manifest_loaded = resolve_cv_folds(
                    X_all,
                    y_all,
                    cv=cv,
                    seed=seed,
                    output_dir=output_dir,
                    manifest_path=cv_manifest_path,
                )
                fold_scores = []
                cv_y_true = []
                cv_y_pred = []
                cv_train_sec = 0.0
                cv_infer_ms = 0.0
                print(f"\n  [{name}]  CV {cv}-fold  (модель: {model_id})")
                print(
                    f"    Fold manifest: {resolved_manifest_path} "
                    f"({'loaded' if manifest_loaded else 'saved'})"
                )
                for fold_idx, (tr_idx, val_idx) in enumerate(folds):
                    print(f"    — Фолд {fold_idx + 1}/{cv}  "
                          f"(train={len(tr_idx)}, val={len(val_idx)})")
                    fold_result = _finetune_transformer(
                        model_name=model_id, friendly_name=f"{name} fold{fold_idx+1}",
                        X_train=X_all[tr_idx], y_train=y_all[tr_idx],
                        X_test=X_all[val_idx],  y_test=y_all[val_idx],
                        save_model=False, silent=True,
                        return_details=cv_only,
                        **_ft_kwargs,
                    )
                    if fold_result is not None:
                        if cv_only:
                            fold_f1 = float(fold_result["f1"])
                            cv_y_true.extend(fold_result["y_true"])
                            cv_y_pred.extend(fold_result["y_pred"])
                            cv_train_sec += float(fold_result["train_sec"])
                            cv_infer_ms += float(fold_result["infer_ms"])
                        else:
                            fold_f1 = float(fold_result)
                        fold_scores.append(fold_f1)
                        print(f"      F1 = {fold_f1:.3f}")
                if fold_scores:
                    mean_f1 = np.mean(fold_scores)
                    std_f1  = np.std(fold_scores)
                    folds_str = ", ".join(f"{s:.3f}" for s in fold_scores)
                    print(f"    CV F1: {mean_f1:.3f} ± {std_f1:.3f}  "
                          f"(фолды: {folds_str})")
                    if cv_only and cv_y_pred:
                        f1 = store.record(
                            f"{name} [CV{cv}]",
                            "transformers",
                            cv_y_true,
                            cv_y_pred,
                            cv_train_sec,
                            cv_infer_ms,
                            notes=f"{model_id}; cv_only={cv}; folds={','.join(f'{s:.4f}' for s in fold_scores)}",
                        )
                        print(f"    CV-only F1: {f1:.3f}  |  Train(sum): {cv_train_sec:.1f}s")
                        print(classification_report(cv_y_true, cv_y_pred, zero_division=0))
                        save_confusion_matrix(cv_y_true, cv_y_pred, f"{name}_CV{cv}", output_dir)

                if cv_only:
                    continue

            # ── Финальное обучение на полном train-сете ───────────────────────
            _finetune_transformer(
                model_name=model_id, friendly_name=name,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                save_model=True, silent=False,
                **_ft_kwargs,
            )
        except Exception as exc:
            print(f"\n  [{name}]  ОШИБКА/ПРОПУСК — {exc}")
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


# ==============================================================================
# SECTION 3.4 — СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ
# ==============================================================================

def generate_comparison_report(store: ResultStore, output_dir: Path, target: str):
    """
    3.4 Сравнительный анализ.

    Формирует:
    - итоговую таблицу в stdout (F1, Precision, Recall, скорость, ресурсы)
    - comparison.csv
    - run_log.json (полные метрики)
    - comparison_<target>.png (bar chart + speed scatter)
    """
    print(f"\n{'═' * 60}")
    print("  3.4  Сравнительный анализ моделей")
    print(f"{'═' * 60}")

    store.print_summary()

    df = store.summary_df()

    # CSV
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  CSV-таблица    : {csv_path.name}")

    # JSON
    json_path = output_dir / "run_log.json"
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target":    target,
        "models": [
            {k: v for k, v in r.items() if k not in ("_y_test", "_y_pred")}
            for r in store.records
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"  Подробный лог  : {json_path.name}")

    # Chart
    save_comparison_chart(store, output_dir, target)

    # Thesis-style table
    def speed_label(ms):
        if ms < 1.0:  return "высокая"
        if ms < 50:   return "средняя"
        return "низкая"

    print(f"\n  {'─' * 80}")
    print(f"  {'Модель':<38} {'F1':>8} {'Скорость':>10} {'Ресурсы':<18} Примечание")
    print(f"  {'─' * 80}")
    for _, row in df.iterrows():
        print(
            f"  {row['model']:<38} {row['f1_weighted']:>8.3f} "
            f"{speed_label(row['infer_ms_per_sample']):>10} "
            f"{row['resource']:<18} {row['notes']}"
        )
    print(f"  {'─' * 80}")


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

_TARGET_DESCRIPTIONS = {
    "call_purpose": "цель звонка",
    "priority":     "приоритет обращения",
    "assig_group":  "группу специалистов",
}


def run_target(df: pd.DataFrame, target: str, output_dir: Path, args, dataset_variant: str) -> None:
    """Запускает все запрошенные группы моделей для одного таргета."""
    print(f"\n{'━' * 60}")
    print(f"  ТАРГЕТ: {target}  |  variant: {dataset_variant}")
    print(f"{'━' * 60}")

    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        prepared_df = prepare_dataset(df, target, dataset_variant)
    except ValueError as e:
        print(f"  Пропуск: {e}")
        return

    snapshot_path = target_dir / "dataset_prepared.csv"
    save_prepared_dataset(prepared_df, snapshot_path, sep=args.sep)
    print(f"  Prepared dataset: {snapshot_path}")

    X = prepared_df["text"].reset_index(drop=True)
    y = prepared_df[target].reset_index(drop=True)
    print(f"  Всего: {len(X)} примеров, {y.nunique()} классов")
    for cls, cnt in y.value_counts().items():
        print(f"    {cls:<35} {cnt:>4}  {'█' * min(cnt, 40)}")

    if args.eval_input is not None:
        try:
            prepared_eval_df = prepare_dataset(args.eval_df, target, dataset_variant)
        except ValueError as e:
            print(f"  Пропуск eval split: {e}")
            return

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

    augmentation_summary = None
    if args.train_augmentation != "none":
        X_train, y_train, augmentation_summary = augment_training_split(
            X_train,
            y_train,
            method=args.train_augmentation,
            target=target,
            output_dir=target_dir,
            seed=args.seed,
        )

    groups = set(g.strip() for g in args.groups.split(","))
    run_all = "all" in groups
    effective_cv = args.cv
    cv_manifest_path = Path(args.fold_manifest).resolve() if args.fold_manifest else None
    if augmentation_summary is not None and effective_cv >= 2:
        print(
            "\n  Предупреждение: CV отключена, потому что train augmentation "
            "дублирует строки и исказит fold-оценку."
        )
        effective_cv = 0

    store = ResultStore()

    # ── 3.2 ──────────────────────────────────────────────────────────────────
    if run_all or "baseline" in groups:
        run_baseline_models(X_train, y_train, X_test, y_test, store, target_dir,
                            cv=effective_cv, cv_only=args.cv_only, seed=args.seed,
                            X_full=X, y_full=y, cv_manifest_path=cv_manifest_path)

    if "legacy-baseline" in groups:
        run_legacy_baseline_models(X_train, y_train, X_test, y_test, store, target_dir)

    # ── 3.3.1 ─────────────────────────────────────────────────────────────────
    embedding_groups = {"embeddings", "word2vec", "doc2vec", "fasttext", "sbert"}
    if run_all or groups & embedding_groups:
        run_embedding_models(
            X_train, y_train, X_test, y_test, store, target_dir,
            sbert_model=args.sbert_model,
        )

    # ── 3.3.2 ─────────────────────────────────────────────────────────────────
    transformer_groups = {"transformers", "rubert", "xlmr", "extra-models"}
    if run_all or groups & transformer_groups:
        run_transformer_models(
            X_train, y_train, X_test, y_test, store, target_dir,
            rubert_model=args.rubert_model,
            xlmr_model=args.xlmr_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            groups=groups,
            fp16=args.fp16,
            bf16=args.bf16,
            grad_accum=args.grad_accum,
            compile_model=args.compile,
            lr=args.lr,
            freeze_layers=args.freeze_layers,
            label_smoothing=args.label_smoothing,
            early_stopping=args.early_stopping,
            early_stopping_metric=args.early_stopping_metric,
            extra_models=args.extra_models,
            max_length=args.max_length,
            truncation_strategy=args.truncation_strategy,
            cv=effective_cv,
            cv_only=args.cv_only,
            seed=args.seed,
            X_full=X,
            y_full=y,
            cv_manifest_path=cv_manifest_path,
        )

    # ── 3.4 ──────────────────────────────────────────────────────────────────
    if store.records:
        generate_comparison_report(store, target_dir, target)
    else:
        print("\n  Нет результатов для сравнения.")


# ==============================================================================
# CLI
# ==============================================================================

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Полный пайплайн ML/DL классификации транскриптов звонков (3.2–3.4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Группы моделей (--groups):
  baseline     — BoW/TF-IDF/N-gram + LogReg/SVM/NB/RF/DecisionTree
  legacy-baseline — старый набор моделей из train.py
  embeddings   — Word2Vec, Doc2Vec, fastText, SBERT
  rubert       — RuBERT fine-tuning (требует transformers + torch)
  xlmr         — XLM-RoBERTa fine-tuning
  transformers — rubert + xlmr вместе
  extra-models — только модели из --extra-models
  cnn          — 7 CNN/RNN архитектур: TextCNN, StackedCNN, DPCNN,
                  CNN-BiLSTM, BiLSTM, BiGRU, BiLSTM+Attention (требует tensorflow)
  llm          — zero-shot + few-shot через OpenAI-совместимый API
  all          — все группы

Примеры:
  python train_advanced.py -i dataset_clean.csv -g baseline
  python train_advanced.py -i dataset_clean.csv -g legacy-baseline
  python train_advanced.py -i dataset_clean.csv -g baseline,embeddings -t all
  python train_advanced.py -i dataset_clean.csv -g baseline,rubert --epochs 5
  python train_advanced.py -i dataset_clean.csv -g llm \\
      --llm-api-base http://localhost:11434/v1 --llm-model llama3
  python train_advanced.py -i dataset_clean.csv -g all -t all
        """,
    )

    # ── Основные аргументы ────────────────────────────────────────────────────
    parser.add_argument("-i", "--input", required=True,
                        help="Входной CSV (выход prepare_dataset.py)")
    parser.add_argument("-t", "--target", default="call_purpose",
                        choices=TARGETS + ["all"],
                        help="Целевая колонка (default: call_purpose)")
    parser.add_argument("-g", "--groups", default="baseline",
                        help="Группы моделей через запятую (default: baseline)")
    parser.add_argument("-o", "--output", default="models_advanced",
                        help="Папка для моделей и отчётов (default: models_advanced)")
    parser.add_argument("--sep", default=";",
                        help="Разделитель CSV (default: ;)")
    parser.add_argument("--dataset-variant", choices=["multiclass", "binary_spam"],
                        default="multiclass",
                        help="Как интерпретировать основной --input "
                             "(default: multiclass)")
    parser.add_argument("--binary-input", default=None,
                        help="Опциональный второй CSV для spam/non-spam. "
                             "Все call_purpose != spam будут схлопнуты в non_spam.")
    parser.add_argument("--eval-input", default=None,
                        help="Фиксированный validation/test CSV. Если задан, внутренний train_test_split отключается.")

    # ── Эмбеддинги ────────────────────────────────────────────────────────────
    parser.add_argument("--sbert-model",
                        default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="HuggingFace ID для SBERT")

    # ── Трансформеры ──────────────────────────────────────────────────────────
    parser.add_argument("--rubert-model", default="cointegrated/rubert-tiny2",
                        help="HuggingFace ID для RuBERT (default: cointegrated/rubert-tiny2)")
    parser.add_argument("--xlmr-model", default="xlm-roberta-base",
                        help="HuggingFace ID для XLM-RoBERTa (default: xlm-roberta-base)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Эпохи fine-tuning (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size для трансформеров (default: 16)")
    parser.add_argument("--fp16", action="store_true",
                        help="Mixed precision fp16 (быстрее на Volta/Turing/Ampere+)")
    parser.add_argument("--bf16", action="store_true",
                        help="Mixed precision bf16 (стабильнее, только Ampere+)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation шагов (default: 1). "
                             "Eff. batch = batch-size × grad-accum")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile() — ускорение графа (PyTorch ≥ 2.0, "
                             "первый батч медленнее из-за JIT)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate для трансформеров (default: 2e-5). "
                             "Для малых датасетов попробуй 1e-5")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Заморозить первые N слоёв энкодера (default: 0). "
                             "Для малых датасетов: 8-10 из 12 слоёв RuBERT")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing [0..1] (default: 0). "
                             "Рекомендуется 0.1 для малых датасетов")
    parser.add_argument("--early-stopping", type=int, default=0,
                        help="Остановить если нет улучшения N эпох подряд (default: 0 = выкл)")
    parser.add_argument("--early-stopping-metric", choices=["f1", "loss"], default="f1",
                        help="Метрика для early stopping у трансформеров "
                             "(default: f1)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Макс. длина последовательности для трансформеров (default: 256, "
                             "макс: 512). Увеличение до 512 захватывает больше контекста, "
                             "но требует больше VRAM и замедляет обучение")
    parser.add_argument("--truncation-strategy",
                        choices=["head", "head_tail", "middle_cut"],
                        default="head",
                        help="Стратегия усечения длинных текстов: "
                             "head = оставить начало; "
                             "head_tail/middle_cut = оставить начало и конец, вырезать середину")
    parser.add_argument("--cv", type=int, default=0,
                        help="Кросс-валидация для baseline моделей: число фолдов (default: 0 = выкл). "
                             "Рекомендуется 5 для малых датасетов")
    parser.add_argument("--cv-only", action="store_true",
                        help="Считать и сохранять только CV-метрики без финального hold-out прогона. "
                             "Требует --cv >= 2")
    parser.add_argument("--fold-manifest", default=None,
                        help="Опциональный JSON с индексами фолдов. "
                             "Если файл уже существует, CV будет выполнена по нему; "
                             "иначе манифест будет создан автоматически.")
    parser.add_argument("--train-augmentation",
                        choices=["none", "oversample"],
                        default="none",
                        help="Train-only аугментация: "
                             "none = без аугментации; "
                             "oversample = дублировать редкие классы до размера мажоритарного")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split / CV / transformer fine-tuning (default: 42)")
    parser.add_argument("--extra-models", type=lambda s: s.split(","), default=None,
                        help="Дополнительные HuggingFace модели через запятую. "
                             "Формат: model/id или model/id:Название. "
                             "Пример: ai-forever/ruBert-base,microsoft/mdeberta-v3-base")
    

    args = parser.parse_args(argv)
    requested_groups = set(g.strip() for g in args.groups.split(",") if g.strip())

    if args.fp16 and args.bf16:
        print("Ошибка: --fp16 и --bf16 нельзя использовать одновременно.")
        return 1

    if args.cv_only and args.cv < 2:
        print("Ошибка: --cv-only требует --cv >= 2.")
        return 1

    if args.cv_only and args.eval_input:
        print("Ошибка: --cv-only нельзя использовать вместе с --eval-input.")
        return 1

    if args.cv_only and args.train_augmentation != "none":
        print("Ошибка: --cv-only нельзя использовать вместе с --train-augmentation.")
        return 1

    if args.cv_only:
        unsupported_for_cv_only = requested_groups & {
            "legacy-baseline", "embeddings", "word2vec", "doc2vec", "fasttext",
            "sbert", "llm", "cnn", "all",
        }
        if unsupported_for_cv_only:
            bad = ", ".join(sorted(unsupported_for_cv_only))
            print(
                "Ошибка: --cv-only сейчас поддержан только для baseline и transformer-групп "
                f"(rubert/xlmr/extra-models). Уберите группы: {bad}."
            )
            return 1

    print_gpu_info()

    csv_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Ошибка: файл '{csv_path}' не найден.")
        return 1

    if args.dataset_variant == "binary_spam" and args.target not in ("call_purpose", "all"):
        print("Ошибка: --dataset-variant binary_spam поддерживает только target=call_purpose.")
        return 1

    if args.dataset_variant == "binary_spam" and args.binary_input:
        print("Ошибка: не используйте --binary-input вместе с --dataset-variant binary_spam.")
        return 1

    if args.eval_input and args.binary_input:
        print("Ошибка: --eval-input пока нельзя использовать вместе с --binary-input.")
        return 1

    try:
        multiclass_df = load_training_frame(csv_path, sep=args.sep)
    except ValueError as exc:
        print(f"Ошибка: {exc}")
        return 1

    binary_path = Path(args.binary_input).resolve() if args.binary_input else None
    binary_df = None
    if binary_path is not None:
        if not binary_path.exists():
            print(f"Ошибка: файл '{binary_path}' не найден.")
            return 1
        try:
            binary_df = load_training_frame(binary_path, sep=args.sep)
        except ValueError as exc:
            print(f"Ошибка: {exc}")
            return 1

    eval_path = Path(args.eval_input).resolve() if args.eval_input else None
    args.eval_df = None
    if eval_path is not None:
        if not eval_path.exists():
            print(f"Ошибка: файл eval '{eval_path}' не найден.")
            return 1
        try:
            args.eval_df = load_training_frame(eval_path, sep=args.sep)
        except ValueError as exc:
            print(f"Ошибка eval split: {exc}")
            return 1

    targets = TARGETS if args.target == "all" else [args.target]

    if args.dataset_variant == "binary_spam":
        run_target(multiclass_df, "call_purpose", output_dir, args, "binary_spam")
    else:
        multiclass_output_dir = output_dir / "multiclass" if binary_df is not None else output_dir
        for target in targets:
            run_target(multiclass_df, target, multiclass_output_dir, args, "multiclass")

        if binary_df is not None and "call_purpose" in targets:
            run_target(binary_df, "call_purpose", output_dir / "binary_spam", args, "binary_spam")

    print(f"\n{'━' * 60}")
    print(f"  Готово. Результаты: {output_dir}")
    print(f"{'━' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
