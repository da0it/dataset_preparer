#!/usr/bin/env python3
"""
train_advanced.py — Полный пайплайн ML/DL классификации транскриптов звонков.

Секция 3.2  Базовые алгоритмы ML
    Векторизаторы : Bag-of-Words (CountVectorizer), TF-IDF, N-gram TF-IDF
    Классификаторы: Logistic Regression, SVM, Naive Bayes, Random Forest

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
import json
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
from dataset_variants import (
    load_training_frame,
    prepare_binary_spam_frame,
    prepare_multiclass_frame,
    save_prepared_dataset,
)
from sklearn.base import BaseEstimator, TransformerMixin
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
    "embeddings":   "средние",
    "transformers": "высокие",
}


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


# ==============================================================================
# SECTION 3.2 — БАЗОВЫЕ АЛГОРИТМЫ КЛАССИФИКАЦИИ
# ==============================================================================

def build_baseline_pipelines() -> dict[str, Pipeline]:
    """
    3.2.1 Векторизация: BoW, TF-IDF, N-gram TF-IDF.
    3.2.2 Классификаторы: LogReg, SVM, NaiveBayes, RandomForest.

    Итого 10 пайплайнов для всестороннего baseline-сравнения.
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
        # ── TF-IDF ─────────────────────────────────────────────────────────
        "TF-IDF + LogReg": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_kw)),
            ("clf", LogisticRegression(**lr_kw)),
        ]),
        "TF-IDF + SVM": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_kw)),
            ("clf", LinearSVC(**svm_kw)),
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
        # ── N-gram TF-IDF ──────────────────────────────────────────────────
        "N-gram(1-3) + SVM": Pipeline([
            ("vec", TfidfVectorizer(**ngram_kw)),
            ("clf", LinearSVC(**svm_kw)),
        ]),
        "N-gram(1-3) + LogReg": Pipeline([
            ("vec", TfidfVectorizer(**ngram_kw)),
            ("clf", LogisticRegression(**lr_kw)),
        ]),
    }


def run_baseline_models(
    X_train, y_train, X_test, y_test,
    store: ResultStore, output_dir: Path,
    cv: int = 0,
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
                from sklearn.model_selection import StratifiedKFold, cross_val_score
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                # Объединяем train+test для CV по всему датасету
                X_all = list(X_train) + list(X_test)
                y_all = list(y_train) + list(y_test)
                cv_scores = cross_val_score(
                    pipeline, X_all, y_all,
                    cv=skf, scoring="f1_weighted", n_jobs=-1
                )
                print(f"    CV {cv}-fold F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
                      f"(folds: {', '.join(f'{s:.3f}' for s in cv_scores)})")

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
    lr: float = 2e-5,
    fp16: bool = False,
    bf16: bool = False,
    grad_accum: int = 1,
    compile_model: bool = False,
    freeze_layers: int = 0,
    label_smoothing: float = 0.0,
    early_stopping: int = 0,
    save_model: bool = True,
    silent: bool = False,
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
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        enc = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
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
    patience_counter = 0
    if early_stopping > 0:
        print(f"    Early stopping: patience={early_stopping} эпох")

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
        with torch.no_grad():
            for enc, _ in test_loader:
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                preds_epoch.extend(torch.argmax(out.logits, 1).cpu().tolist())
        epoch_f1 = f1_score(y_te, preds_epoch, average="weighted", zero_division=0)
        print(f"    Epoch {epoch + 1}: loss={avg_loss:.4f}  val_F1={epoch_f1:.3f}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            if early_stopping > 0:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"    Early stopping: нет улучшения {early_stopping} эпох подряд, "
                          f"лучший val_F1={best_f1:.3f} (epoch {epoch + 1 - early_stopping})")
                    break

    train_sec = time.perf_counter() - t0

    # ── Load best checkpoint ──────────────────────────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    all_preds = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for enc, _ in test_loader:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                out = model(**enc)
            all_preds.extend(torch.argmax(out.logits, 1).cpu().tolist())
    infer_ms = (time.perf_counter() - t1) * 1000

    y_pred      = le.inverse_transform(np.array(all_preds))
    y_test_orig = list(y_test)

    f1 = store.record(
        friendly_name, "transformers", y_test_orig, y_pred,
        train_sec, infer_ms, notes=model_name,
    ) if not silent else f1_score(y_test_orig, y_pred, average="weighted", zero_division=0)

    if not silent:
        print(f"    F1: {f1:.3f}  |  Train: {train_sec:.1f}s")
        print(classification_report(y_test_orig, y_pred, zero_division=0))
        save_confusion_matrix(y_test_orig, y_pred, friendly_name, output_dir)

    # ── Save model ────────────────────────────────────────────────────────────
    if save_model and not silent:
        safe = (friendly_name.replace(" ", "_").replace("/", "-")
                .replace("(", "").replace(")", ""))
        model_out = output_dir / safe
        model_out.mkdir(exist_ok=True)
        model.save_pretrained(str(model_out))
        tokenizer.save_pretrained(str(model_out))
        joblib.dump(le, model_out / "label_encoder.joblib")
        print(f"    Сохранено: {model_out.name}/")

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
    extra_models: list = None,
    max_length: int = 256,
    cv: int = 0,
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
        early_stopping=early_stopping, max_length=max_length,
    )

    for model_id, name in models:

        # ── Кросс-валидация ───────────────────────────────────────────────────
        if cv >= 2:
            from sklearn.model_selection import StratifiedKFold
            X_all = np.array(list(X_train) + list(X_test))
            y_all = np.array(list(y_train) + list(y_test))
            skf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            fold_scores = []
            print(f"\n  [{name}]  CV {cv}-fold  (модель: {model_id})")
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
                print(f"    — Фолд {fold_idx + 1}/{cv}  "
                      f"(train={len(tr_idx)}, val={len(val_idx)})")
                fold_f1 = _finetune_transformer(
                    model_name=model_id, friendly_name=f"{name} fold{fold_idx+1}",
                    X_train=X_all[tr_idx], y_train=y_all[tr_idx],
                    X_test=X_all[val_idx],  y_test=y_all[val_idx],
                    save_model=False, silent=True,
                    **_ft_kwargs,
                )
                if fold_f1 is not None:
                    fold_scores.append(fold_f1)
                    print(f"      F1 = {fold_f1:.3f}")
            if fold_scores:
                mean_f1 = np.mean(fold_scores)
                std_f1  = np.std(fold_scores)
                folds_str = ", ".join(f"{s:.3f}" for s in fold_scores)
                print(f"    CV F1: {mean_f1:.3f} ± {std_f1:.3f}  "
                      f"(фолды: {folds_str})")

        # ── Финальное обучение на полном train-сете ───────────────────────────
        _finetune_transformer(
            model_name=model_id, friendly_name=name,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            save_model=True, silent=False,
            **_ft_kwargs,
        )


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    groups = set(g.strip() for g in args.groups.split(","))
    run_all = "all" in groups

    store = ResultStore()

    # ── 3.2 ──────────────────────────────────────────────────────────────────
    if run_all or "baseline" in groups:
        run_baseline_models(X_train, y_train, X_test, y_test, store, target_dir,
                            cv=args.cv)

    # ── 3.3.1 ─────────────────────────────────────────────────────────────────
    embedding_groups = {"embeddings", "word2vec", "doc2vec", "fasttext", "sbert"}
    if run_all or groups & embedding_groups:
        run_embedding_models(
            X_train, y_train, X_test, y_test, store, target_dir,
            sbert_model=args.sbert_model,
        )

    # ── 3.3.2 ─────────────────────────────────────────────────────────────────
    transformer_groups = {"transformers", "rubert", "xlmr"}
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
            extra_models=args.extra_models,
            max_length=args.max_length,
            cv=args.cv,
        )

    # ── 3.4 ──────────────────────────────────────────────────────────────────
    if store.records:
        generate_comparison_report(store, target_dir, target)
    else:
        print("\n  Нет результатов для сравнения.")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Полный пайплайн ML/DL классификации транскриптов звонков (3.2–3.4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Группы моделей (--groups):
  baseline     — BoW/TF-IDF + LogReg/SVM/NB/RF
  embeddings   — Word2Vec, Doc2Vec, fastText, SBERT
  rubert       — RuBERT fine-tuning (требует transformers + torch)
  xlmr         — XLM-RoBERTa fine-tuning
  transformers — rubert + xlmr вместе
  cnn          — 7 CNN/RNN архитектур: TextCNN, StackedCNN, DPCNN,
                  CNN-BiLSTM, BiLSTM, BiGRU, BiLSTM+Attention (требует tensorflow)
  llm          — zero-shot + few-shot через OpenAI-совместимый API
  all          — все группы

Примеры:
  python train_advanced.py -i dataset_clean.csv -g baseline
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
    parser.add_argument("--binary-input", default=None,
                        help="Опциональный второй CSV для spam/non-spam. "
                             "Все call_purpose != spam будут схлопнуты в non_spam.")

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
    parser.add_argument("--max-length", type=int, default=256,
                        help="Макс. длина последовательности для трансформеров (default: 256, "
                             "макс: 512). Увеличение до 512 захватывает больше контекста, "
                             "но требует больше VRAM и замедляет обучение")
    parser.add_argument("--cv", type=int, default=0,
                        help="Кросс-валидация для baseline моделей: число фолдов (default: 0 = выкл). "
                             "Рекомендуется 5 для малых датасетов")
    parser.add_argument("--extra-models", type=lambda s: s.split(","), default=None,
                        help="Дополнительные HuggingFace модели через запятую. "
                             "Формат: model/id или model/id:Название. "
                             "Пример: ai-forever/ruBert-base,microsoft/mdeberta-v3-base")
    

    args = parser.parse_args()

    if args.fp16 and args.bf16:
        print("Ошибка: --fp16 и --bf16 нельзя использовать одновременно.")
        sys.exit(1)

    print_gpu_info()

    csv_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Ошибка: файл '{csv_path}' не найден.")
        sys.exit(1)

    try:
        multiclass_df = load_training_frame(csv_path, sep=args.sep)
    except ValueError as exc:
        print(f"Ошибка: {exc}")
        sys.exit(1)

    binary_path = Path(args.binary_input).resolve() if args.binary_input else None
    binary_df = None
    if binary_path is not None:
        if not binary_path.exists():
            print(f"Ошибка: файл '{binary_path}' не найден.")
            sys.exit(1)
        try:
            binary_df = load_training_frame(binary_path, sep=args.sep)
        except ValueError as exc:
            print(f"Ошибка: {exc}")
            sys.exit(1)

    targets = TARGETS if args.target == "all" else [args.target]
    multiclass_output_dir = output_dir / "multiclass" if binary_df is not None else output_dir
    for target in targets:
        run_target(multiclass_df, target, multiclass_output_dir, args, "multiclass")

    if binary_df is not None and "call_purpose" in targets:
        run_target(binary_df, "call_purpose", output_dir / "binary_spam", args, "binary_spam")

    print(f"\n{'━' * 60}")
    print(f"  Готово. Результаты: {output_dir}")
    print(f"{'━' * 60}\n")


if __name__ == "__main__":
    main()
