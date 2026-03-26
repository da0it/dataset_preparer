#!/usr/bin/env python3
"""Shared tokenization helpers for transformer training and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


INFERENCE_CONFIG_NAME = "inference_config.json"


def normalize_truncation_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("-", "_")
    if normalized == "middle_cut":
        return "head_tail"
    if normalized not in {"head", "head_tail"}:
        raise ValueError(
            f"Unsupported truncation strategy: {strategy}. "
            "Expected one of: head, head_tail, middle_cut."
        )
    return normalized


def build_single_sequence_features(tokenizer, text: str, max_length: int, truncation_strategy: str):
    strategy = normalize_truncation_strategy(truncation_strategy)
    if strategy == "head":
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=True,
        )

    body_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    available = max_length - special_tokens
    if available <= 0:
        raise ValueError(
            f"max_length={max_length} is too small for tokenizer special tokens."
        )

    if len(body_ids) > available:
        head_len = (available + 1) // 2
        tail_len = available - head_len
        kept_ids = body_ids[:head_len]
        if tail_len > 0:
            kept_ids += body_ids[-tail_len:]
    else:
        kept_ids = body_ids

    return tokenizer.prepare_for_model(
        kept_ids,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=True,
    )


def encode_text_batch(
    tokenizer,
    texts: Iterable[str],
    max_length: int,
    truncation_strategy: str,
    return_tensors: str = "pt",
):
    text_list = list(texts)
    strategy = normalize_truncation_strategy(truncation_strategy)
    if strategy == "head":
        return tokenizer(
            text_list,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors=return_tensors,
        )

    features = [
        build_single_sequence_features(tokenizer, text, max_length, strategy)
        for text in text_list
    ]
    return tokenizer.pad(features, padding=True, return_tensors=return_tensors)


def save_inference_config(model_dir: Path, max_length: int, truncation_strategy: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / INFERENCE_CONFIG_NAME
    payload = {
        "max_length": int(max_length),
        "truncation_strategy": normalize_truncation_strategy(truncation_strategy),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return config_path


def load_inference_config(model_dir: Path) -> dict | None:
    config_path = model_dir / INFERENCE_CONFIG_NAME
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "max_length": int(payload["max_length"]),
        "truncation_strategy": normalize_truncation_strategy(payload["truncation_strategy"]),
    }


def resolve_inference_config(
    model_dir: Path,
    fallback_max_length: int,
    fallback_strategy: str,
) -> tuple[int, str]:
    saved = load_inference_config(model_dir)
    if saved is not None:
        return saved["max_length"], saved["truncation_strategy"]
    return int(fallback_max_length), normalize_truncation_strategy(fallback_strategy)
