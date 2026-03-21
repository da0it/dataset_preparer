#!/usr/bin/env python3
"""
Shared dataset-processing operations used by CLI entrypoints.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PRIORITY_MAP = {
    "high": "high",
    "medium": "medium",
    "low": "low",
}

CALL_PURPOSE_MAP = {
    "consulting": "consulting",
    "cooperation": "cooperation",
    "courses": "courses",
    "customer_support": "customer_support",
    "general": "general",
    "hr": "hr",
    "license": "license",
    "ned": "need",
    "public_events": "public_events",
    "spam": "spam",
    "triage": "triage",
    "portal_access": "portal_access",
}

ASSIGNED_GROUP_MAP = {
    "education_center": "education_center",
    "general": "general",
    "hr_team": "hr_team",
    "sales": "sales",
    "consulting.general": "consulting.general",
    "consulting.greenplum": "consulting.greenplum",
    "consulting.hadoop": "consulting.hadoop",
    "cooperation_department": "cooperation_department",
    "customer_suppot.manager": "customer_support.manager",
    "customer.manager": "customer_support.manager",
    "marketing_department": "marketing_department",
    "portal_support": "portal_support",
    "public_events.conference": "public_events.conference",
}

SUSPICIOUS = {"?", "-", "ge", ""}

DEFAULT_KEEP_COLUMNS = ["filename", "text", "call_purpose", "priority", "assig_group"]
FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}__\d{2}-\d{2}-\d{2})")

RAW_NOISE_PATTERNS = [
    r"\bдобрый\s+день\b",
    r"\bдоброе\s+утро\b",
    r"\bдобрый\s+вечер\b",
    r"\bздравствуйте\b",
    r"\bдобро\s+пожаловать\b",
    r"\bвы\s+позвонили\s+в\s+компанию\b",
    r"\bвы\s+дозвонились\b",
    r"\bслужба\s+(?:поддержки|технической\s+поддержки|клиентского\s+сервиса)\b",
    r"\bтехническая\s+поддержка\b",
    r"\bудобно\s+(?:вам\s+)?(?:сейчас\s+)?говорить\b",
    r"\bчем\s+могу\s+(?:вам\s+)?помочь\b",
    r"\bслушаю\s+(?:вас\b)?",
    r"\bслушаю\b",
    r"\bкак\s+могу\s+(?:вам\s+)?помочь\b",
    r"\bгот(?:ов|ова)\s+(?:вам\s+)?помочь\b",
    r"\bя\s+(?:вас\s+)?(?:понял|поняла|слышу|слышал[а]?)\b",
    r"\bпринято\b",
    r"\bхорошо\b",
    r"\bпонятно\b",
    r"\bконечно\b",
    r"\bнет(?:[,.]?\s+нет)*\b",
    r"\bпожалуйста\b",
    r"\bподождите\s+(?:пожалуйста\s+)?(?:минуту|секунду|немного|одну\s+минуту)?\b",
    r"\bодну\s+(?:секунду|минуту)\b",
    r"\bпозвольте\s+уточнить\b",
    r"\bпроверяю\b",
    r"\bсмотрю\b",
    r"\bпередам\s+(?:ваше?\s+)?(?:обращение|вопрос|заявк[уи])\b",
    r"\bпередам\s+(?:вас|информацию|данные)\b",
    r"\bс\s+вами\s+свяжутся\b",
    r"\bс\s+вами\s+свяжется\s+(?:наш\s+)?(?:специалист|менеджер|сотрудник)\b",
    r"\bнаш\s+(?:специалист|менеджер|сотрудник)\s+(?:вам\s+)?(?:перезвонит|свяжется)\b",
    r"\bперезвоним\s+(?:вам\b)?",
    r"\bоставьте\s+(?:свои\s+)?данные\b",
    r"\bоставьте\s+(?:свой\s+)?(?:телефон|номер|email|почту)\b",
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
    r"\bваш\s+номер\s+телефона\b",
    r"\bваша?\s+(?:электронная\s+)?почта\b",
    r"\bдиктуйте\b",
    r"\bзаписываю\b",
    r"\bзаписал[а]?\b",
    r"\bмогу\s+(?:ли\s+)?(?:я\s+)?(?:ещё\s+)?чем-?(?:то|нибудь)\s+помочь\b",
    r"\bесть\s+(?:ещё\s+)?(?:какие-?(?:то|нибудь)\s+)?вопросы\b",
    r"\bесли\s+(?:у\s+вас\s+)?(?:будут\s+)?(?:ещё\s+)?вопросы\b",
    r"\bобращайтесь\b",
]

NOISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in RAW_NOISE_PATTERNS]
PUNCT_CLEANUP_RE = re.compile(r"[\s,.:;!?\-]+$")
MULTI_SPACE_RE = re.compile(r"\s{2,}")
LEADING_PUNCT_RE = re.compile(r"^[\s,.:;!?\-]+")


def normalize_col(series: pd.Series, alias_map: dict, col_name: str) -> pd.Series:
    result = series.copy()
    unknown = set()

    for i, val in series.items():
        normalized = str(val).strip().lower()
        if normalized in SUSPICIOUS:
            result[i] = ""
            continue
        mapped = alias_map.get(normalized)
        if mapped:
            result[i] = mapped
        else:
            result[i] = normalized
            unknown.add(normalized)

    if unknown:
        print(f"\n[warn] [{col_name}] Unknown values (kept as-is, review manually):")
        for v in sorted(unknown):
            count = (series.str.strip().str.lower() == v).sum()
            print(f"   {count:3d}  '{v}'")

    return result


def clean_filename(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return value
    match = FILENAME_RE.match(value.strip())
    return match.group(1) if match else value


def remove_noise(text: str) -> str:
    for pat in NOISE_PATTERNS:
        text = pat.sub(" ", text)
    text = LEADING_PUNCT_RE.sub("", text)
    text = PUNCT_CLEANUP_RE.sub("", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def run_normalize_labels(input_path: Path, output_path: Path, sep: str) -> None:
    import pandas as pd

    if not input_path.exists():
        raise FileNotFoundError(f"'{input_path}' not found.")

    df = pd.read_csv(input_path, sep=sep, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from {input_path.name}")

    for col in ["call_purpose", "priority", "assigned_group"]:
        if col not in df.columns:
            continue
        mask = df[col].str.strip().str.lower().isin(SUSPICIOUS)
        if mask.any():
            print(f"\n[warn] [{col}] Suspicious values (will be cleared):")
            for idx, row in df[mask].iterrows():
                print(f"   row {idx:4d}: '{row[col]}'  |  {str(row.get('text', ''))[:80]}")

    for col, alias_map in [
        ("priority", PRIORITY_MAP),
        ("call_purpose", CALL_PURPOSE_MAP),
        ("assigned_group", ASSIGNED_GROUP_MAP),
    ]:
        if col not in df.columns:
            print(f"\nSkipping '{col}' — column not found.")
            continue
        df[col] = normalize_col(df[col], alias_map, col)
        print(f"\n-- {col} after normalization --")
        vc = df[col][df[col] != ""].value_counts()
        for val, cnt in vc.items():
            print(f"   {cnt:3d}  {val}")
        empty = (df[col] == "").sum()
        if empty:
            print(f"   {empty:3d}  (empty / cleared)")

    df.to_csv(output_path, index=False, encoding="utf-8", sep=sep)
    print(f"\nSaved to {output_path}")


def run_filter_and_clean(
    input_path: Path,
    output_path: Path,
    sep: str,
    keep_all_cols: bool,
) -> None:
    import pandas as pd

    if not input_path.exists():
        raise FileNotFoundError(f"'{input_path}' not found.")

    df = pd.read_csv(input_path, sep=sep, dtype=str).fillna("")

    if "call_purpose" not in df.columns:
        raise ValueError(
            "Column 'call_purpose' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    n_before = len(df)
    df = df[df["call_purpose"].str.strip() != ""]
    n_after = len(df)
    n_dropped = n_before - n_after

    if "filename" in df.columns:
        df["filename"] = df["filename"].apply(clean_filename)
    else:
        print("Warning: column 'filename' not found, skipping cleanup.", file=sys.stderr)

    if not keep_all_cols:
        keep = [c for c in DEFAULT_KEEP_COLUMNS if c in df.columns]
        missing = [c for c in DEFAULT_KEEP_COLUMNS if c not in df.columns]
        if missing:
            print(f"Warning: columns not found and skipped: {missing}", file=sys.stderr)
        df = df[keep]
        print(f"  Kept columns       : {keep}")

    df.to_csv(output_path, index=False, sep=sep, encoding="utf-8")

    print(f"\n{'-' * 50}")
    print(f"Done. Output : {output_path}")
    print(f"{'-' * 50}")
    print(f"  Rows before filter : {n_before}")
    print(f"  Dropped (no target): {n_dropped}")
    print(f"  Rows after filter  : {n_after}")
    print(f"{'-' * 50}")


def run_clean_noise(
    input_path: Path,
    output_path: Path,
    min_tokens: int,
    source_column: str,
    sep: str,
    dry_run: bool,
) -> None:
    import pandas as pd
    from tqdm import tqdm

    if not input_path.exists():
        raise FileNotFoundError(f"'{input_path}' not found.")

    df = pd.read_csv(input_path, sep=sep, dtype=str).fillna("")
    if source_column not in df.columns:
        raise ValueError(
            f"Column '{source_column}' not found. Columns: {df.columns.tolist()}"
        )

    if dry_run:
        sample = df[df[source_column].str.len() > 20].sample(min(5, len(df)), random_state=42)
        for _, row in sample.iterrows():
            before = row[source_column]
            after = remove_noise(before)
            print("BEFORE:", before[:200])
            print("AFTER :", after[:200])
            print("-" * 60)
        return

    tqdm.pandas(desc="Removing noise")
    df["text_before_noise_clean"] = df[source_column]
    df[source_column] = df[source_column].progress_apply(
        lambda t: remove_noise(t) if isinstance(t, str) else t
    )
    df["too_short"] = df[source_column].apply(
        lambda t: str(len(t.split()) < min_tokens)
        if isinstance(t, str) and not t.startswith("[TRANSCRIPTION_ERROR")
        else "False"
    )

    n_short = (df["too_short"] == "True").sum()
    n_empty = (df[source_column] == "").sum()
    n_total = len(df)

    df.to_csv(output_path, index=False, encoding="utf-8", sep=sep)
    print(f"\n{'-' * 50}")
    print(f"Done. Output : {output_path}")
    print(f"{'-' * 50}")
    print(f"  Total rows         : {n_total}")
    print(f"  Too short (<{min_tokens} tok): {n_short}  <- check 'too_short' column")
    print(f"  Empty after clean  : {n_empty}")
    print(f"{'-' * 50}")
    if n_short > 0:
        print("Tip: filter before training:")
        print("     df = df[df['too_short'].str.lower() != 'true']")
