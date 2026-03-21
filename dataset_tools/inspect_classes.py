#!/usr/bin/env python3
"""
Инспекция классов датасета.

Показывает случайные примеры текстов для каждого класса —
удобно для проверки качества разметки и размытых границ между классами.

Usage:
    # Показать по 5 примеров для каждого класса
    python inspect_classes.py --input dataset_ready.csv --target call_purpose

    # Показать 10 примеров только для конкретных классов
    python inspect_classes.py --input dataset_ready.csv --target call_purpose \
        --classes consulting license --n 10

    # Показать примеры где модель ошиблась (confusion-режим)
    python inspect_classes.py --input dataset_ready.csv --target call_purpose \
        --confused consulting license
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SEP = ";"


def load(csv_path: Path, target: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=SEP, dtype=str).fillna("")
    df = df[df["text"].str.strip() != ""]
    df = df[df[target].str.strip() != ""]
    return df


def show_samples(df: pd.DataFrame, target: str,
                 classes: list[str] | None, n: int):
    """Показать n случайных примеров для каждого класса."""
    all_classes = sorted(df[target].unique())
    selected = classes if classes else all_classes

    for cls in selected:
        if cls not in all_classes:
            print(f"  ⚠️  Класс '{cls}' не найден. Доступные: {all_classes}")
            continue

        subset = df[df[target] == cls]
        sample = subset.sample(min(n, len(subset)), random_state=42)

        print(f"\n{'═' * 65}")
        print(f"  КЛАСС: {cls}  ({len(subset)} примеров всего)")
        print(f"{'═' * 65}")
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            text = row["text"].strip()
            # Обрезаем длинные тексты для читаемости
            if len(text) > 300:
                text = text[:300] + "…"
            print(f"\n  [{i}] {text}")
            print(f"  {'─' * 60}")


def show_confused(df: pd.DataFrame, target: str,
                  cls_a: str, cls_b: str, n: int):
    """
    Показать примеры обоих классов рядом — для оценки
    насколько они текстово похожи.
    """
    for cls in [cls_a, cls_b]:
        if cls not in df[target].unique():
            print(f"  ⚠️  Класс '{cls}' не найден.")
            sys.exit(1)

    subset_a = df[df[target] == cls_a].sample(
        min(n, len(df[df[target] == cls_a])), random_state=42
    )
    subset_b = df[df[target] == cls_b].sample(
        min(n, len(df[df[target] == cls_b])), random_state=42
    )

    # Чередуем примеры A и B — удобно сравнивать
    print(f"\n{'═' * 65}")
    print(f"  СРАВНЕНИЕ: «{cls_a}» vs «{cls_b}»")
    print(f"  Если тексты неотличимы — граница размыта, нужна переразметка")
    print(f"{'═' * 65}")

    rows_a = list(subset_a.iterrows())
    rows_b = list(subset_b.iterrows())
    for i in range(max(len(rows_a), len(rows_b))):
        if i < len(rows_a):
            text = rows_a[i][1]["text"].strip()[:300]
            print(f"\n  [{cls_a}]  {text}")
        if i < len(rows_b):
            text = rows_b[i][1]["text"].strip()[:300]
            print(f"  [{cls_b}]  {text}")
        print(f"  {'─' * 60}")


def show_distribution(df: pd.DataFrame, target: str):
    """Показать распределение классов."""
    counts = df[target].value_counts()
    total = len(df)
    print(f"\n  Распределение классов по '{target}' (всего {total}):")
    print(f"  {'Класс':<30} {'N':>5}  {'%':>6}  {'bar'}")
    print(f"  {'─' * 60}")
    for cls, cnt in counts.items():
        pct = cnt / total * 100
        bar = "█" * min(int(pct), 40)
        print(f"  {cls:<30} {cnt:>5}  {pct:>5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Инспекция текстов по классам для проверки разметки."
    )
    parser.add_argument("--input",   "-i", required=True,
                        help="CSV файл датасета")
    parser.add_argument("--target",  "-t", default="call_purpose",
                        help="Колонка с классами (default: call_purpose)")
    parser.add_argument("--classes", "-c", nargs="+", default=None,
                        help="Конкретные классы для показа (default: все)")
    parser.add_argument("--confused", nargs=2, metavar=("CLASS_A", "CLASS_B"),
                        help="Показать два класса чередуя примеры для сравнения")
    parser.add_argument("--n",       type=int, default=5,
                        help="Количество примеров на класс (default: 5)")
    parser.add_argument("--sep",     default=";",
                        help="Разделитель CSV (default: ;)")
    parser.add_argument("--dist",    action="store_true",
                        help="Показать только распределение классов")

    args = parser.parse_args()

    global SEP
    SEP = args.sep

    csv_path = Path(args.input).resolve()
    if not csv_path.exists():
        print(f"Error: файл '{csv_path}' не найден.", file=sys.stderr)
        sys.exit(1)

    df = load(csv_path, args.target)

    if args.dist:
        show_distribution(df, args.target)
        return

    show_distribution(df, args.target)

    if args.confused:
        show_confused(df, args.target, args.confused[0], args.confused[1], args.n)
    else:
        show_samples(df, args.target, args.classes, args.n)


if __name__ == "__main__":
    main()
