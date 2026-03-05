#!/usr/bin/env python3
"""
Постобработка датасета:
  1. Оставляет только строки где заполнен call_purpose
  2. В колонке filename оставляет только дату и время:
     2022-03-05__17-18-34__79000000000__мусор.mp3  →  2022-03-05__17-18-34

Usage:
    python filter_and_clean.py --input dataset_final.csv --output dataset_ready.csv
    python filter_and_clean.py --input dataset_final.csv --output dataset_ready.csv --sep ";"
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Колонки которые нужны для обучения — всё остальное удаляется
_DEFAULT_KEEP = ["filename", "text", "call_purpose", "priority", "assig_group"]

# Паттерн: дата__время — всё что идёт после второго блока отбрасывается
# Формат: YYYY-MM-DD__HH-MM-SS
_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2})")


def clean_filename(value: str) -> str:
    """Оставить только дату и время из имени файла."""
    if not isinstance(value, str) or not value.strip():
        return value
    m = _FILENAME_RE.match(value.strip())
    return m.group(1) if m else value


def main():
    parser = argparse.ArgumentParser(
        description="Фильтрация по call_purpose и очистка колонки filename."
    )
    parser.add_argument("--input",  "-i", required=True,  help="Входной CSV")
    parser.add_argument("--output", "-o", required=True,  help="Выходной CSV")
    parser.add_argument("--sep",    default=";",           help="Разделитель CSV (default: ;)")
    parser.add_argument("--keep-all-cols", action="store_true",
                        help="Не удалять служебные колонки (по умолчанию оставляет только "
                             "text, call_purpose, priority, assig_group)")
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        print(f"Error: файл '{input_path}' не найден.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path, sep=args.sep, dtype=str).fillna("")

    # ── 1. Фильтр по call_purpose ─────────────────────────────────────────
    if "call_purpose" not in df.columns:
        print("Error: колонка 'call_purpose' не найдена.", file=sys.stderr)
        print(f"Доступные колонки: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    n_before = len(df)
    df = df[df["call_purpose"].str.strip() != ""]
    n_after  = len(df)
    n_dropped = n_before - n_after

    # ── 2. Очистка filename ───────────────────────────────────────────────
    if "filename" in df.columns:
        df["filename"] = df["filename"].apply(clean_filename)
    else:
        print("Warning: колонка 'filename' не найдена, пропускаю.", file=sys.stderr)

    # ── 3. Удаление служебных колонок ────────────────────────────────────
    if not args.keep_all_cols:
        keep = [c for c in _DEFAULT_KEEP if c in df.columns]
        missing = [c for c in _DEFAULT_KEEP if c not in df.columns]
        if missing:
            print(f"Warning: колонки не найдены и пропущены: {missing}", file=sys.stderr)
        df = df[keep]
        print(f"  Оставлены колонки   : {keep}")

    df.to_csv(output_path, index=False, sep=args.sep, encoding="utf-8")

    print(f"\n{'─'*50}")
    print(f"Done. Output : {output_path}")
    print(f"{'─'*50}")
    print(f"  Строк до фильтра    : {n_before}")
    print(f"  Удалено (нет цели)  : {n_dropped}")
    print(f"  Строк в результате  : {n_after}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
