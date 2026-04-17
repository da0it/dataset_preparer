#!/usr/bin/env python3
"""
Helpers for preparing multiclass and binary spam/non-spam dataset variants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPAM_LABEL = "spam"
NON_SPAM_LABEL = "non_spam"


def load_training_frame(csv_path: Path, sep: str = ";") -> pd.DataFrame:
    """Load CSV and keep only rows suitable for training."""
    df = pd.read_csv(csv_path, sep=sep, dtype=str).fillna("")

    if "is_training_sample" in df.columns:
        df = df[df["is_training_sample"].str.strip() == "1"]

    if "text" not in df.columns:
        raise ValueError("Column 'text' not found in dataset.")

    df = df[df["text"].str.strip() != ""].copy()
    return df.reset_index(drop=True)


def prepare_multiclass_frame(
    df: pd.DataFrame,
    target: str,
    min_samples_per_class: int,
    spam_label: str = SPAM_LABEL,
    include_spam: bool = False,
) -> pd.DataFrame:
    """
    Prepare the main multiclass dataset.

    Spam calls are removed from every target because they should not participate
    in downstream routing/classification by default. Set include_spam=True for
    experiments where spam is an explicit multiclass label.
    """
    frame = df.copy()

    if not include_spam and "call_purpose" in frame.columns:
        purpose_norm = frame["call_purpose"].astype(str).str.strip().str.lower()
        frame = frame[purpose_norm != spam_label].copy()

    return _finalize_target_frame(frame, target, min_samples_per_class)


def prepare_binary_spam_frame(
    df: pd.DataFrame,
    target: str,
    min_samples_per_class: int,
    spam_label: str = SPAM_LABEL,
    non_spam_label: str = NON_SPAM_LABEL,
) -> pd.DataFrame:
    """
    Prepare a binary dataset where every non-spam call purpose becomes non_spam.
    """
    if target != "call_purpose":
        raise ValueError("Binary spam dataset is supported only for target='call_purpose'.")

    if "call_purpose" not in df.columns:
        raise ValueError("Column 'call_purpose' not found in binary dataset.")

    frame = df.copy()
    labels = frame["call_purpose"].astype(str).str.strip().str.lower()
    frame = frame[labels != ""].copy()
    labels = frame["call_purpose"].astype(str).str.strip().str.lower()
    frame["call_purpose"] = np.where(labels == spam_label, spam_label, non_spam_label)

    return _finalize_target_frame(frame, target, min_samples_per_class)


def save_prepared_dataset(df: pd.DataFrame, output_path: Path, sep: str = ";") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8", sep=sep)


def _finalize_target_frame(
    df: pd.DataFrame,
    target: str,
    min_samples_per_class: int,
) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in dataset.")

    frame = df.copy()
    frame[target] = frame[target].astype(str).str.strip()
    frame = frame[frame[target] != ""].copy()

    counts = frame[target].value_counts()
    valid_classes = counts[counts >= min_samples_per_class].index
    frame = frame[frame[target].isin(valid_classes)].copy()

    if frame.empty:
        raise ValueError(
            f"No usable samples for target '{target}' after filtering rare/empty classes."
        )

    return frame.reset_index(drop=True)
