#!/usr/bin/env python3
"""
Legacy baseline models originally defined in train.py.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

COMMERCIAL_KEYWORDS = re.compile(
    r"\b(?:"
    r"стоимост[ьи]|цен[ауе]|прайс|лицензи[яию]|лицензирован\w+"
    r"|купить|приобрести|закупк[аи]|бюджет\w*"
    r"|коммерческ\w+|предложени[яе]|прайс.?лист"
    r"|менеджер\w*\s+(?:по\s+)?продаж\w+"
    r"|отдел\s+продаж|тендер\w*|договор\w*"
    r")\b",
    re.IGNORECASE,
)

TECHNICAL_KEYWORDS = re.compile(
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
    """Add explicit commercial and technical flags alongside TF-IDF."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        has_commercial = np.array(
            [1.0 if COMMERCIAL_KEYWORDS.search(t) else 0.0 for t in X]
        ).reshape(-1, 1)
        has_technical = np.array(
            [1.0 if TECHNICAL_KEYWORDS.search(t) else 0.0 for t in X]
        ).reshape(-1, 1)
        return np.hstack([has_commercial, has_technical])


def build_legacy_baseline_pipelines() -> dict[str, Pipeline | GridSearchCV]:
    tfidf = dict(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=1,
    )
    return {
        "Legacy TF-IDF + SVM": _build_svm_grid(tfidf),
        "Legacy TF-IDF + Keywords + SVM": _build_svm_keywords_grid(tfidf),
        "Legacy TF-IDF + RandomForest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Legacy TF-IDF + KNN": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="cosine")),
        ]),
    }


def _build_svm_grid(tfidf_params: dict) -> GridSearchCV:
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf", LinearSVC(max_iter=3000, class_weight="balanced")),
    ])
    param_grid = {
        "clf__C": [0.1, 0.3, 0.5, 1.0, 3.0],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": [1, 2],
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


def _build_svm_keywords_grid(tfidf_params: dict) -> GridSearchCV:
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("keywords", KeywordFeaturesTransformer()),
        ])),
        ("clf", LinearSVC(max_iter=3000, class_weight="balanced")),
    ])
    param_grid = {
        "clf__C": [0.1, 0.3, 0.5, 1.0, 3.0],
        "features__tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "features__tfidf__min_df": [1, 2],
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
