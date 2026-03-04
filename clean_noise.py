#!/usr/bin/env python3
"""
Noise phrase removal for call-center transcription datasets.

Removes templated "first-line" phrases (greetings, farewells, service
phrases) that don't carry classification signal and pollute training.

Examples of removed phrases:
  - вы позвонили в компанию / добро пожаловать
  - чем могу помочь / слушаю вас
  - я вас понял / принято / понятно
  - передам ваше обращение / с вами свяжутся
  - всего доброго / до свидания / спасибо за звонок

Usage:
    python clean_noise.py --input dataset_clean.csv --output dataset_clean_v2.csv
    python clean_noise.py --input dataset_clean.csv --output out.csv --min-tokens 5

The script:
  1. Removes matched noise phrases (regex, case-insensitive)
  2. Collapses leftover whitespace / punctuation
  3. Flags rows where text became too short after cleaning
  4. Saves original text to 'text_before_noise_clean' for audit
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Noise phrase patterns ────────────────────────────────────────────────────
#
# Rules:
#   • Patterns are applied to already-lowercased text (prepare_dataset.py
#     normalizes to lowercase). If you run on raw text, add re.IGNORECASE.
#   • Each pattern is a full phrase or a flexible regex that covers common
#     surface forms. Word-boundary anchors (\b) prevent partial matches.
#   • Order doesn't matter — all patterns are applied independently.
#   • Add new patterns to the list as you discover more noise in your data.

_RAW_PATTERNS = [
    # ── Greetings ──────────────────────────────────────────────────────────
    r"\bдобрый\s+день\b",
    r"\bдоброе\s+утро\b",
    r"\bдобрый\s+вечер\b",
    r"\bздравствуйте\b",
    r"\bдобро\s+пожаловать\b",

    # ── Call-center opening ────────────────────────────────────────────────
    r"\bвы\s+позвонили\s+в\s+компанию\b",
    r"\bвы\s+дозвонились\b",
    r"\bслужба\s+(?:поддержки|технической\s+поддержки|клиентского\s+сервиса)\b",
    r"\bтехническая\s+поддержка\b",
    r"\bудобно\s+(?:вам\s+)?(?:сейчас\s+)?говорить\b",

    # ── Offer to help ──────────────────────────────────────────────────────
    r"\bчем\s+могу\s+(?:вам\s+)?помочь\b",
    r"\bслушаю\s+(?:вас\b)?",
    r"\bслушаю\b",
    r"\bкак\s+могу\s+(?:вам\s+)?помочь\b",
    r"\bгот(?:ов|ова)\s+(?:вам\s+)?помочь\b",

    # ── Acknowledgement ────────────────────────────────────────────────────
    r"\bя\s+(?:вас\s+)?(?:понял|поняла|слышу|слышал[а]?)\b",
    r"\bпринято\b",
    r"\bхорошо\b",
    r"\bпонятно\b",
    r"\bконечно\b",
    r"\bнет(?:[,.]?\s+нет)*\b",
    r"\bпожалуйста\b",

    # ── Hold / wait phrases ────────────────────────────────────────────────
    r"\bподождите\s+(?:пожалуйста\s+)?(?:минуту|секунду|немного|одну\s+минуту)?\b",
    r"\bодну\s+(?:секунду|минуту)\b",
    r"\bпозвольте\s+уточнить\b",
    r"\bпроверяю\b",
    r"\bсмотрю\b",

    # ── Transfer / escalation ──────────────────────────────────────────────
    r"\bпередам\s+(?:ваше?\s+)?(?:обращение|вопрос|заявк[уи])\b",
    r"\bпередам\s+(?:вас|информацию|данные)\b",
    r"\bс\s+вами\s+свяжутся\b",
    r"\bс\s+вами\s+свяжется\s+(?:наш\s+)?(?:специалист|менеджер|сотрудник)\b",
    r"\bнаш\s+(?:специалист|менеджер|сотрудник)\s+(?:вам\s+)?(?:перезвонит|свяжется)\b",
    r"\bперезвоним\s+(?:вам\b)?",
    r"\bоставьте\s+(?:свои\s+)?данные\b",
    r"\bоставьте\s+(?:свой\s+)?(?:телефон|номер|email|почту)\b",

    # ── Farewells ──────────────────────────────────────────────────────────
    r"\bвсего\s+(?:вам\s+)?доброго\b",
    r"\bвсего\s+хорошего\b",
    r"\bдо\s+свидания\b",
    r"\bдо\s+встречи\b",
    r"\bспасибо\s+за\s+(?:ваш\s+)?звонок\b",
    r"\bспасибо\s+за\s+обращение\b",
    r"\bспасибо\s+(?:вам\b)?",
    r"\bблагодарю\b",
    r"\bхорошего\s+(?:вам\s+)?дня\b",
    r"\bхорошего\s+вечера\b",

    # ── Contact-data prompts ───────────────────────────────────────────────
    r"\bваш\s+номер\s+телефона\b",
    r"\bваша?\s+(?:электронная\s+)?почта\b",
    r"\bдиктуйте\b",
    r"\bзаписываю\b",
    r"\bзаписал[а]?\b",

    # ── Closing confirmation ───────────────────────────────────────────────
    r"\bмогу\s+(?:ли\s+)?(?:я\s+)?(?:ещё\s+)?чем-?(?:то|нибудь)\s+помочь\b",
    r"\bесть\s+(?:ещё\s+)?(?:какие-?(?:то|нибудь)\s+)?вопросы\b",
    r"\bесли\s+(?:у\s+вас\s+)?(?:будут\s+)?(?:ещё\s+)?вопросы\b",
    r"\bобращайтесь\b",
]

# Compile once
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in _RAW_PATTERNS]

# Trailing junk after phrase removal
_PUNCT_CLEANUP_RE = re.compile(r"[\s,.:;!?\-]+$")
_MULTI_SPACE_RE   = re.compile(r"\s{2,}")
_LEADING_PUNCT_RE = re.compile(r"^[\s,.:;!?\-]+")


def remove_noise(text: str) -> str:
    """Remove all noise phrases from a single text string."""
    for pat in _NOISE_RE:
        text = pat.sub(" ", text)
    text = _LEADING_PUNCT_RE.sub("", text)
    text = _PUNCT_CLEANUP_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Remove call-center noise phrases from a transcription CSV."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input CSV (output of prepare_dataset.py)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV (default: <input>_no_noise.csv)")
    parser.add_argument("--min-tokens", type=int, default=3,
                        help="Minimum token count after cleaning (default: 3)")
    parser.add_argument("--source-column", default="text",
                        help="Column to clean (default: text)")
    parser.add_argument("--sep", default=",",
                        help="CSV field separator (default: comma)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print 5 random before/after examples and exit")

    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else \
                  input_path.with_stem(input_path.stem + "_no_noise")

    if not input_path.exists():
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path, sep=args.sep, dtype=str).fillna("")

    src_col = args.source_column
    if src_col not in df.columns:
        print(f"Error: column '{src_col}' not found. "
              f"Columns: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    # ── Dry-run mode ──────────────────────────────────────────────────────
    if args.dry_run:
        sample = df[df[src_col].str.len() > 20].sample(min(5, len(df)), random_state=42)
        for _, row in sample.iterrows():
            before = row[src_col]
            after  = remove_noise(before)
            print("BEFORE:", before[:200])
            print("AFTER :", after[:200])
            print("─" * 60)
        return

    # ── Full run ──────────────────────────────────────────────────────────
    tqdm.pandas(desc="Removing noise")

    # Preserve original text for audit
    df["text_before_noise_clean"] = df[src_col]

    df[src_col] = df[src_col].progress_apply(
        lambda t: remove_noise(t) if isinstance(t, str) else t
    )

    # Flag rows that became too short
    df["too_short"] = df[src_col].apply(
        lambda t: str(len(t.split()) < args.min_tokens)
        if isinstance(t, str) and not t.startswith("[TRANSCRIPTION_ERROR")
        else "False"
    )

    n_short = (df["too_short"] == "True").sum()
    n_empty = (df[src_col] == "").sum()
    n_total = len(df)

    df.to_csv(output_path, index=False, encoding="utf-8", sep=args.sep)

    print(f"\n{'─'*50}")
    print(f"Done. Output : {output_path}")
    print(f"{'─'*50}")
    print(f"  Total rows         : {n_total}")
    print(f"  Too short (<{args.min_tokens} tok)  : {n_short}  ← check 'too_short' column")
    print(f"  Empty after clean  : {n_empty}")
    print(f"{'─'*50}")
    if n_short > 0:
        print("Tip: filter before training:")
        print("     df = df[df['too_short'].str.lower() != 'true']")


if __name__ == "__main__":
    main()
