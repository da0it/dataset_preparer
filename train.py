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
import re
import warnings
from datetime import datetime
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

TARGETS = ["call_purpose", "priority", "assig_group"]
SEP = ";"  # CSV separator — change to "," if needed
MIN_SAMPLES_PER_CLASS = 5  # classes with fewer examples are dropped
TEST_SIZE = 0.2            # доля данных в hold-out тесте (20%)

# ── Keyword feature engineering ───────────────────────────────────────────────
#
# Явные маркеры классов — помогают модели различить consulting и license
# когда TF-IDF не справляется из-за похожего словаря.
#
# Принцип: если слово точно указывает на класс — добавляем как отдельный признак.
# Слова подобраны по реальным примерам из датасета.

_COMMERCIAL_KEYWORDS = re.compile(
    r"\b(?:"
    r"стоимост[ьи]|цен[ауе]|прайс|лицензи[яию]|лицензирован\w+"
    r"|купить|приобрести|закупк[аи]|бюджет\w*"
    r"|коммерческ\w+|предложени[яе]|прайс.?лист"
    r"|менеджер\w*\s+(?:по\s+)?продаж\w+"
    r"|отдел\s+продаж|тендер\w*|договор\w*"
    r")\b",
    re.IGNORECASE,
)

_TECHNICAL_KEYWORDS = re.compile(
    r"\b(?:"
    r"установ\w+|настро\w+|развернуть|деплой\w*|конфигур\w+"
    r"|ошибк[аи]|баг\w*|верси[яию]|кластер\w*|нод[аы]"
    r"|питон|python|ansible|docker|linux|bash|shell"
    r"|adcm|adh|adpg|adb|арендата\s+дб"
    r"|не\s+работает|не\s+запускается|не\s+подключается|упал\w*"
    r"|логи|дебаг\w*|трейс\w*|стектрейс\w*"
    r")\b",
    re.IGNORECASE,
)


class KeywordFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Добавляет два числовых признака поверх TF-IDF:
      - has_commercial : 1 если текст содержит коммерческие маркеры (→ license)
      - has_technical  : 1 если текст содержит технические маркеры  (→ consulting)

    Признаки масштабируются MaxAbsScaler перед конкатенацией с TF-IDF,
    чтобы их вес был сопоставим с TF-IDF значениями.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        has_commercial = np.array(
            [1.0 if _COMMERCIAL_KEYWORDS.search(t) else 0.0 for t in X]
        ).reshape(-1, 1)
        has_technical = np.array(
            [1.0 if _TECHNICAL_KEYWORDS.search(t) else 0.0 for t in X]
        ).reshape(-1, 1)
        return np.hstack([has_commercial, has_technical])


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
        min_df=1,
    )
    return {
        # GridSearchCV автоматически подбирает лучший C и ngram_range на train-fold'ах
        "TF-IDF + SVM": _build_svm_grid(tfidf),
        # TF-IDF + явные keyword-признаки + SVM
        "TF-IDF + Keywords + SVM": _build_svm_keywords_grid(tfidf),
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


def _build_svm_grid(tfidf_params: dict) -> GridSearchCV:
    """
    SVM с автоподбором гиперпараметров через GridSearchCV.

    Перебирает комбинации:
      C            — регуляризация (меньше = мягче граница)
      ngram_range  — учитывать ли биграммы и триграммы
      min_df       — минимальная частота слова

    Подбор идёт на train-части через 5-fold CV, тест не трогается.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LinearSVC(max_iter=3000, class_weight="balanced")),
    ])
    param_grid = {
        "clf__C":              [0.1, 0.3, 0.5, 1.0, 3.0],
        "tfidf__ngram_range":  [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df":       [1, 2],
    }
    return GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
        refit=True,   # после подбора переобучает на всём train с лучшими параметрами
        verbose=0,
    )


def _build_svm_keywords_grid(tfidf_params: dict) -> GridSearchCV:
    """
    SVM с TF-IDF + keyword-признаками.

    FeatureUnion конкатенирует:
      - TF-IDF матрицу (sparse)
      - 2 числовых признака из KeywordFeaturesTransformer

    MaxAbsScaler масштабирует keyword-признаки чтобы их вес
    был сопоставим с TF-IDF значениями.
    """
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("tfidf",    TfidfVectorizer(**tfidf_params)),
            ("keywords", KeywordFeaturesTransformer()),
        ])),
        ("clf", LinearSVC(max_iter=3000, class_weight="balanced")),
    ])
    param_grid = {
        "clf__C":                          [0.1, 0.3, 0.5, 1.0, 3.0],
        "features__tfidf__ngram_range":    [(1, 1), (1, 2), (1, 3)],
        "features__tfidf__min_df":         [1, 2],
    }
    return GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


def _pipeline_params(pipeline) -> dict:
    """Извлечь параметры пайплайна. Поддерживает GridSearchCV и обычный Pipeline."""
    if isinstance(pipeline, GridSearchCV):
        return {
            "best_params": {k: str(v) for k, v in pipeline.best_params_.items()},
            "best_cv_score": round(pipeline.best_score_, 4),
        }
    params = {}
    for step_name, step in pipeline.steps:
        step_params = step.get_params()
        params[step_name] = {k: str(v) for k, v in step_params.items()}
    return params


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, pipeline: Pipeline,
             X_train: pd.Series, y_train: pd.Series,
             X_test: pd.Series,  y_test: pd.Series,
             output_dir: Path) -> dict:
    """Cross-validate on train, evaluate on hold-out test, save model + confusion matrix."""

    print(f"\n  [{name}]")

    is_grid = isinstance(pipeline, GridSearchCV)

    if is_grid:
        # GridSearchCV сам делает CV внутри при fit()
        pipeline.fit(X_train, y_train)
        best_score = pipeline.best_score_
        cv_scores = np.array([best_score])
        print(f"    Best params : {pipeline.best_params_}")
        print(f"    Best CV F1  (train, weighted): {best_score:.3f}")
    else:
        # Обычный Pipeline — CV отдельно, потом fit на всём train
        n_splits = min(5, y_train.value_counts().min())
        if n_splits < 2:
            print("    Not enough samples for CV — skipping cross-validation.")
            cv_scores = np.array([0.0])
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1
            )
            print(f"    CV F1  (train, weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        pipeline.fit(X_train, y_train)

    # Честная оценка на hold-out тесте
    y_pred_test = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    print(f"    Test F1 (hold-out, weighted): {test_f1:.3f}  ← честная оценка")
    print()
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # Confusion matrix по тестовой выборке
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    _save_confusion_matrix(cm, labels, name, output_dir)

    # Сохранить модель (для GridSearchCV сохраняем best_estimator_)
    safe_name = name.replace(" ", "_").replace("+", "plus").replace("/", "-")
    model_path = output_dir / f"{safe_name}.joblib"
    save_obj = pipeline.best_estimator_ if is_grid else pipeline
    joblib.dump(save_obj, model_path)
    print(f"    Saved: {model_path.name}")

    # Подробный classification report по классам (для лога)
    report_dict = classification_report(
        y_test, y_pred_test, zero_division=0, output_dict=True
    )

    return {
        "model":            name,
        "cv_f1_mean":       round(float(cv_scores.mean()), 4),
        "cv_f1_std":        round(float(cv_scores.std()), 4),
        "cv_f1_folds":      [round(float(s), 4) for s in cv_scores],
        "test_f1":          round(test_f1, 4),
        "model_path":       str(model_path),
        "params":           _pipeline_params(pipeline),
        "per_class_report": report_dict,
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

def run_target(csv_path: Path, target: str, output_dir: Path,
               run_meta: dict) -> list[dict]:
    """Обучает все модели для одного таргета. Возвращает список результатов."""
    print(f"\n{'═' * 60}")
    print(f"  TARGET: {target}")
    print(f"{'═' * 60}")

    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        X, y = load_data(csv_path, target)
    except ValueError as e:
        print(f"  Skipping: {e}")
        return []

    class_dist = y.value_counts().to_dict()
    print(f"  Всего samples : {len(X)}")
    _print_class_distribution(y)

    # Hold-out split 80/20, стратифицированный по классам
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"\n  Train : {len(X_train)} | Test (hold-out) : {len(X_test)}")

    pipelines = build_pipelines()
    results = []

    for name, pipeline in pipelines.items():
        try:
            result = evaluate(name, pipeline, X_train, y_train, X_test, y_test, target_dir)
            result["target"]        = target
            result["n_total"]       = len(X)
            result["n_train"]       = len(X_train)
            result["n_test"]        = len(X_test)
            result["class_dist"]    = {str(k): int(v) for k, v in class_dist.items()}
            results.append(result)
        except Exception as exc:
            print(f"  ERROR training {name}: {exc}")

    # Summary table
    if results:
        print(f"\n  {'─' * 50}")
        print(f"  Summary for '{target}':")
        print(f"  {'Model':<30} {'CV F1':>8} {'±':>6} {'Test F1':>10}")
        print(f"  {'─' * 50}")
        for r in sorted(results, key=lambda x: x["test_f1"], reverse=True):
            print(f"  {r['model']:<30} {r['cv_f1_mean']:>8.3f} {r['cv_f1_std']:>6.3f} {r['test_f1']:>10.3f}")

        # Подробный лог для этого таргета
        target_log = {
            "run": run_meta,
            "target": target,
            "data": {
                "n_total":    len(X),
                "n_train":    len(X_train),
                "n_test":     len(X_test),
                "test_size":  TEST_SIZE,
                "class_dist": {str(k): int(v) for k, v in class_dist.items()},
            },
            "models": results,
        }
        log_path = target_dir / "run_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(target_log, f, ensure_ascii=False, indent=2)
        print(f"\n  Full log saved to : {log_path}")

    return results


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

    # Метаинформация о запуске — пишется в каждый лог
    run_meta = {
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "input_file":        str(csv_path),
        "output_dir":        str(output_dir),
        "min_samples_class": MIN_SAMPLES_PER_CLASS,
        "test_size":         TEST_SIZE,
        "random_state":      42,
    }

    targets = TARGETS if args.target == "all" else [args.target]
    all_results = []
    for target in targets:
        results = run_target(csv_path, target, output_dir, run_meta)
        all_results.extend(results)

    # Сводный лог всего запуска
    if all_results:
        summary = {
            "run": run_meta,
            "summary": [
                {
                    "target":       r["target"],
                    "model":        r["model"],
                    "cv_f1_mean":   r["cv_f1_mean"],
                    "cv_f1_std":    r["cv_f1_std"],
                    "test_f1":      r["test_f1"],
                    "n_train":      r["n_train"],
                    "n_test":       r["n_test"],
                }
                for r in sorted(all_results, key=lambda x: (x["target"], -x["test_f1"]))
            ],
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n  Сводный лог : {summary_path}")

    print(f"\n{'═' * 60}")
    print(f"  Done. Models and plots saved to: {output_dir}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
