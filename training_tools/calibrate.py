#!/usr/bin/env python3
"""
calibrate.py — Калибровка уверенности трансформер-модели (XLM-RoBERTa, RuBERT и др.).

Что делает:
  1. Загружает обученную HuggingFace-модель (сохранённую train_advanced.py).
  2. Прогоняет тестовую выборку, получает сырые logits.
  3. Считает метрики калибровки ДО: ECE, Brier score, NLL.
  4. Подбирает оптимальную температуру T (temperature scaling).
  5. Считает метрики калибровки ПОСЛЕ: ECE, Brier score, NLL.
  6. Строит reliability diagram до/после калибровки.
  7. Выводит таблицу по тирам уверенности (>0.80 / 0.60-0.80 / <0.60).
  8. Сохраняет T в calibration.json рядом с моделью.

Структура сохранённой модели (train_advanced.py):
  models/<target>/<ModelName>/
    config.json
    model.safetensors (или pytorch_model.bin)
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    label_encoder.joblib

Использование:
    python calibrate.py \\
        --model-dir models_advanced/call_purpose/XLM-RoBERTa_large \\
        --input dataset_train_ready.csv \\
        --target call_purpose

    python calibrate.py \\
        --model-dir models_advanced/call_purpose/XLM-RoBERTa_large \\
        --input dataset_train_ready.csv \\
        --target call_purpose \\
        --output calibration_results/

    python calibrate.py \\
        --model-dir models_advanced/priority/XLM-RoBERTa_large \\
        --input dataset_train_ready.csv \\
        --target priority \\
        --tiers 0.70 0.85
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from training_tools.tokenization_utils import (
    encode_text_batch,
    load_hf_tokenizer,
    resolve_inference_config,
)

warnings.filterwarnings("ignore")

# ── Константы (должны совпадать с train_advanced.py) ──────────────────────────
SEP = ";"
MIN_SAMPLES_PER_CLASS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_TIERS = [0.60, 0.80]


# ==============================================================================
# УТИЛИТЫ КАЛИБРОВКИ
# ==============================================================================

def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    Разбивает [0,1] на n_bins корзин, в каждой считает |avg_conf - avg_acc|,
    взвешивает по числу примеров.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correct[mask].mean()
        ece += mask.sum() / n * abs(avg_conf - avg_acc)
    return float(ece)


def compute_brier(proba: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> float:
    """
    Multiclass Brier Score = среднее sum((p_k - y_k)^2) по всем классам.
    """
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    return float(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))


def compute_nll(proba: np.ndarray, y_true: np.ndarray, classes: np.ndarray,
                eps: float = 1e-9) -> float:
    """
    Negative Log-Likelihood (кросс-энтропия).
    Штрафует за самоуверенные ошибки.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    nll = 0.0
    for i, label in enumerate(y_true):
        idx = class_to_idx[label]
        nll -= np.log(proba[i, idx] + eps)
    return float(nll / len(y_true))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """softmax(logits / T)."""
    return softmax(logits / T, axis=1)


def find_optimal_temperature(logits: np.ndarray, y_true: np.ndarray,
                              classes: np.ndarray) -> float:
    """
    Temperature Scaling: подбирает T в (0.1, 10.0), минимизируя NLL.
    """
    def nll_with_T(T):
        proba = apply_temperature(logits, T)
        return compute_nll(proba, y_true, classes)

    result = minimize_scalar(nll_with_T, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


# ==============================================================================
# RELIABILITY DIAGRAM
# ==============================================================================

def plot_reliability_diagram(
    confidences_before: np.ndarray,
    correct_before: np.ndarray,
    confidences_after: np.ndarray,
    correct_after: np.ndarray,
    ece_before: float,
    ece_after: float,
    title: str,
    output_path: Path,
    n_bins: int = 10,
):
    """Строит reliability diagram до и после temperature scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    pairs = [
        (axes[0], confidences_before, correct_before,
         f"До калибровки  (ECE={ece_before:.3f})"),
        (axes[1], confidences_after,  correct_after,
         f"После калибровки  (ECE={ece_after:.3f})"),
    ]

    for ax, conf, corr, label in pairs:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accs   = []
        bin_confs  = []
        bin_counts = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() == 0:
                bin_accs.append(np.nan)
                bin_confs.append((lo + hi) / 2)
                bin_counts.append(0)
            else:
                bin_accs.append(corr[mask].mean())
                bin_confs.append(conf[mask].mean())
                bin_counts.append(int(mask.sum()))

        bin_centers = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
        bin_accs    = np.array(bin_accs)

        for bc, ba, cnt in zip(bin_centers, bin_accs, bin_counts):
            if np.isnan(ba):
                continue
            color = "#d73027" if ba < bc else "#4575b4"
            ax.bar(bc, ba, width=0.09, alpha=0.75, color=color,
                   align="center", edgecolor="white")
            ax.bar(bc, bc, width=0.09, alpha=0.18, color="gray",
                   align="center", edgecolor="white")
            ax.text(bc, 0.02, str(cnt), ha="center", va="bottom",
                    fontsize=7, color="black")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Идеальная калибровка")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Средняя уверенность (confidence)")
        ax.set_ylabel("Реальная точность (accuracy)")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
    print(f"  Reliability diagram  -> {output_path.name}")


# ==============================================================================
# АНАЛИЗ ПО ТИРАМ УВЕРЕННОСТИ
# ==============================================================================

def tier_analysis(
    confidences: np.ndarray,
    correct: np.ndarray,
    tiers: list,
    label: str = "",
) -> list:
    """
    Разбивает предсказания на тиры по confidence.
    tiers = [0.60, 0.80] -> три тира: <0.60, 0.60-0.80, >0.80.
    """
    boundaries = [0.0] + sorted(tiers) + [1.0]
    tier_labels = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == 0:
            tier_labels.append(f"< {hi:.2f}  (низкий)")
        elif i == len(boundaries) - 2:
            tier_labels.append(f"> {lo:.2f}  (высокий / авто)")
        else:
            tier_labels.append(f"{lo:.2f} - {hi:.2f}  (средний / валидация)")

    results = []
    n_total = len(confidences)

    print(f"\n  {'─' * 66}")
    if label:
        print(f"  Тиры уверенности [{label}]")
    print(f"  {'Тир':<32} {'Кол-во':>8} {'Coverage':>10} {'Accuracy':>10}")
    print(f"  {'─' * 66}")

    for i, tlabel in enumerate(tier_labels):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == 0:
            mask = confidences <= hi
        elif i == len(boundaries) - 2:
            mask = confidences > lo
        else:
            mask = (confidences > lo) & (confidences <= hi)

        count    = int(mask.sum())
        coverage = count / n_total if n_total > 0 else 0.0
        accuracy = float(correct[mask].mean()) if count > 0 else float("nan")

        print(f"  {tlabel:<32} {count:>8} {coverage:>9.1%} {accuracy:>10.3f}")
        results.append({
            "tier":     tlabel,
            "count":    count,
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4) if not np.isnan(accuracy) else None,
        })

    print(f"  {'─' * 66}")
    return results


# ==============================================================================
# ЗАГРУЗКА МОДЕЛИ И ДАННЫХ
# ==============================================================================

def load_model_and_tokenizer(model_dir: Path):
    """Загружает HuggingFace-модель, токенизатор и LabelEncoder."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("Установите: pip install transformers torch")

    le_path = model_dir / "label_encoder.joblib"
    if not le_path.exists():
        raise FileNotFoundError(f"label_encoder.joblib не найден в {model_dir}")

    le        = joblib.load(le_path)
    tokenizer = load_hf_tokenizer(str(model_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    print(f"  Устройство: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        vram     = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(f"  GPU: {gpu_name}  |  VRAM: {vram:.1f} GB")

    return model, tokenizer, le, device


def get_logits(model, tokenizer, texts: list, device,
               batch_size: int = 32, max_length: int = 256,
               truncation_strategy: str = "head") -> np.ndarray:
    """Прогоняет тексты через модель, возвращает numpy-матрицу logits."""
    import torch
    all_logits = []
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc = encode_text_batch(
            tokenizer,
            batch,
            max_length=max_length,
            truncation_strategy=truncation_strategy,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        all_logits.append(out.logits.cpu().float().numpy())

    return np.vstack(all_logits)


def load_data(csv_path: Path, target: str, sep: str) -> tuple:
    """Загружает данные так же, как train_advanced.py (те же фильтры и split)."""
    df = pd.read_csv(csv_path, sep=sep, dtype=str).fillna("")
    if "is_training_sample" in df.columns:
        df = df[df["is_training_sample"].str.strip() == "1"]
    df = df[df["text"].str.strip() != ""]
    df = df[df[target].str.strip() != ""]

    counts = df[target].value_counts()
    valid  = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    df     = df[df[target].isin(valid)]

    if len(df) == 0:
        raise ValueError(f"Нет данных для таргета '{target}'.")

    X, y = df["text"], df[target]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return list(X_test), list(y_test)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Калибровка confidence трансформер-модели (temperature scaling)."
    )
    parser.add_argument("--model-dir", "-m", required=True,
                        help="Папка с обученной моделью (из train_advanced.py)")
    parser.add_argument("--input", "-i", required=True,
                        help="CSV датасет (тот же, что использовался при обучении)")
    parser.add_argument("--target", "-t", required=True,
                        choices=["call_purpose", "priority", "assig_group"],
                        help="Целевой столбец")
    parser.add_argument("--output", "-o", default=None,
                        help="Папка для графиков (по умолчанию: рядом с моделью)")
    parser.add_argument("--sep", default=";",
                        help="Разделитель CSV (по умолчанию: ;)")
    parser.add_argument("--tiers", type=float, nargs="+", default=DEFAULT_TIERS,
                        help="Границы тиров уверенности (по умолчанию: 0.60 0.80)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Размер батча при инференсе (по умолчанию: 32)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Макс. длина токенов (по умолчанию: 256)")
    parser.add_argument("--truncation-strategy",
                        choices=["head", "head_tail", "middle_cut"],
                        default="head",
                        help="Fallback-стратегия усечения, если рядом с моделью нет inference_config.json")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Число корзин для ECE / reliability diagram (по умолчанию: 10)")
    args = parser.parse_args()

    model_dir  = Path(args.model_dir).resolve()
    csv_path   = Path(args.input).resolve()
    output_dir = Path(args.output).resolve() if args.output else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        print(f"Ошибка: папка модели не найдена: {model_dir}")
        return
    if not csv_path.exists():
        print(f"Ошибка: CSV не найден: {csv_path}")
        return

    model_name = model_dir.name
    effective_max_length, effective_truncation_strategy = resolve_inference_config(
        model_dir, args.max_length, args.truncation_strategy
    )

    print(f"\n{'=' * 60}")
    print(f"  Калибровка: {model_name}")
    print(f"  Таргет:     {args.target}")
    print(f"  Тиры:       {args.tiers}")
    print(f"  Tokenizer:  max_length={effective_max_length}, "
          f"truncation={effective_truncation_strategy}")
    print(f"{'=' * 60}\n")

    # ── Загрузка модели ────────────────────────────────────────────────────────
    print("  Загрузка модели...")
    model, tokenizer, le, device = load_model_and_tokenizer(model_dir)
    classes = le.classes_

    # ── Загрузка тестовых данных ───────────────────────────────────────────────
    print("  Загрузка тестовой выборки...")
    X_test, y_test = load_data(csv_path, args.target, args.sep)
    print(f"  Тестовых примеров: {len(X_test)}")

    # ── Инференс -> logits ────────────────────────────────────────────────────
    print("  Инференс (получение logits)...")
    logits = get_logits(
        model,
        tokenizer,
        X_test,
        device,
        batch_size=args.batch_size,
        max_length=effective_max_length,
        truncation_strategy=effective_truncation_strategy,
    )

    # ── До калибровки ─────────────────────────────────────────────────────────
    proba_before  = softmax(logits, axis=1)
    conf_before   = proba_before.max(axis=1)
    pred_idx      = proba_before.argmax(axis=1)
    y_pred_before = le.inverse_transform(pred_idx)
    correct_before = (np.array(y_pred_before) == np.array(y_test)).astype(float)

    ece_before   = compute_ece(conf_before, correct_before, n_bins=args.n_bins)
    brier_before = compute_brier(proba_before, np.array(y_test), classes)
    nll_before   = compute_nll(proba_before, np.array(y_test), classes)
    f1_before    = f1_score(y_test, y_pred_before, average="weighted", zero_division=0)

    print(f"\n  -- До калибровки {'─' * 38}")
    print(f"  F1-weighted : {f1_before:.4f}")
    print(f"  ECE         : {ece_before:.4f}  (< 0.05 хорошо, > 0.15 плохо)")
    print(f"  Brier score : {brier_before:.4f}  (чем меньше, тем лучше)")
    print(f"  NLL         : {nll_before:.4f}  (чем меньше, тем лучше)")

    # ── Подбор температуры ────────────────────────────────────────────────────
    print("\n  Подбор температуры (temperature scaling)...")
    T_opt = find_optimal_temperature(logits, np.array(y_test), classes)
    print(f"  Оптимальная температура T = {T_opt:.4f}")
    if T_opt > 1.0:
        print("  -> T > 1: модель была переуверена (overconfident)")
    elif T_opt < 1.0:
        print("  -> T < 1: модель была недоуверена (underconfident)")
    else:
        print("  -> T ~ 1: модель уже хорошо откалибрована")

    # ── После калибровки ──────────────────────────────────────────────────────
    proba_after   = apply_temperature(logits, T_opt)
    conf_after    = proba_after.max(axis=1)
    pred_idx_af   = proba_after.argmax(axis=1)
    y_pred_after  = le.inverse_transform(pred_idx_af)
    correct_after = (np.array(y_pred_after) == np.array(y_test)).astype(float)

    ece_after   = compute_ece(conf_after, correct_after, n_bins=args.n_bins)
    brier_after = compute_brier(proba_after, np.array(y_test), classes)
    nll_after   = compute_nll(proba_after, np.array(y_test), classes)
    f1_after    = f1_score(y_test, y_pred_after, average="weighted", zero_division=0)

    print(f"\n  -- После калибровки (T={T_opt:.4f}) {'─' * 28}")
    print(f"  F1-weighted : {f1_after:.4f}  (не должен меняться)")
    print(f"  ECE         : {ece_after:.4f}")
    print(f"  Brier score : {brier_after:.4f}")
    print(f"  NLL         : {nll_after:.4f}")

    # ── Сравнительная таблица ─────────────────────────────────────────────────
    print(f"\n  {'─' * 54}")
    print(f"  {'Метрика':<20} {'До':>10} {'После':>10} {'Дельта':>10}")
    print(f"  {'─' * 54}")
    for mname, before, after in [
        ("ECE",         ece_before,   ece_after),
        ("Brier score", brier_before, brier_after),
        ("NLL",         nll_before,   nll_after),
        ("F1-weighted", f1_before,    f1_after),
    ]:
        delta = after - before
        arrow = "v" if delta < 0 else ("^" if delta > 0 else "=")
        print(f"  {mname:<20} {before:>10.4f} {after:>10.4f} {delta:>+9.4f} {arrow}")
    print(f"  {'─' * 54}")

    # ── Анализ по тирам ────────────────────────────────────────────────────────
    tiers_before = tier_analysis(conf_before, correct_before, args.tiers,
                                  label="до калибровки")
    tiers_after  = tier_analysis(conf_after,  correct_after,  args.tiers,
                                  label=f"после калибровки (T={T_opt:.4f})")

    # ── Reliability diagram ────────────────────────────────────────────────────
    diagram_path = output_dir / f"reliability_{args.target}.png"
    plot_reliability_diagram(
        conf_before, correct_before,
        conf_after,  correct_after,
        ece_before, ece_after,
        title=f"{model_name} — {args.target}",
        output_path=diagram_path,
        n_bins=args.n_bins,
    )

    # ── Сохранение результатов ────────────────────────────────────────────────
    calib_data = {
        "model":       model_name,
        "target":      args.target,
        "n_test":      len(X_test),
        "temperature": round(T_opt, 6),
        "tiers":       args.tiers,
        "before": {
            "f1_weighted": round(f1_before, 4),
            "ece":         round(ece_before, 4),
            "brier":       round(brier_before, 4),
            "nll":         round(nll_before, 4),
            "tiers":       tiers_before,
        },
        "after": {
            "f1_weighted": round(f1_after, 4),
            "ece":         round(ece_after, 4),
            "brier":       round(brier_after, 4),
            "nll":         round(nll_after, 4),
            "tiers":       tiers_after,
        },
    }

    calib_json = output_dir / f"calibration_{args.target}.json"
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Результаты  -> {calib_json.name}")

    # Температуру сохраняем рядом с моделью — для последующего инференса
    T_path = model_dir / f"temperature_{args.target}.json"
    with open(T_path, "w", encoding="utf-8") as f:
        json.dump({"temperature": round(T_opt, 6), "target": args.target}, f, indent=2)
    print(f"  Температура -> {T_path.name}  (используйте при инференсе)")

    print(f"\n{'=' * 60}")
    print(f"  Готово. T = {T_opt:.4f}  |  ECE: {ece_before:.4f} -> {ece_after:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
