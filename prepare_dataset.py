#!/usr/bin/env python3
"""
Dataset preparation script for ML training.

Steps applied to each text:
  1. PII removal  — names, phones, emails, addresses, org names, job titles,
                    contract/ticket numbers → replaced with [PLACEHOLDER] tokens
  2. Normalization — unicode cleanup, whitespace, lowercasing, punctuation,
                     repeated chars, filler words
  3. Length filter — rows with too little text after cleaning are flagged

Usage:
    python prepare_dataset.py --input dataset.csv --output dataset_clean.csv
    python prepare_dataset.py --input dataset.csv --output dataset_clean.csv --min-tokens 5

Dependencies:
    pip install natasha pymorphy3 pandas tqdm
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Natasha (Russian NER) ────────────────────────────────────────────────────
try:
    from natasha import (
        Doc,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsNERTagger,
        Segmenter,
    )
    _NATASHA_OK = True
except ImportError:
    _NATASHA_OK = False
    print(
        "Warning: natasha not installed. NER-based PII removal (names, orgs, "
        "addresses) will be skipped.\nInstall: pip install natasha",
        file=sys.stderr,
    )

# ── Constants ────────────────────────────────────────────────────────────────

# Regex-based PII patterns (applied before NER)
PHONE_RE = re.compile(
    r"""
    (?:
        \+?7|8           # country code
    )?
    [\s\-\(]*
    \d{3}                # area code
    [\s\-\)]*
    \d{3}
    [\s\-]*
    \d{2}
    [\s\-]*
    \d{2}
    """,
    re.VERBOSE,
)

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}",
)

# Contract / ticket / order numbers: digit sequences of 5+ chars,
# often preceded by a keyword
CONTRACT_RE = re.compile(
    r"""
    (?:
        (?:договор|контракт|тикет|заявк[аи]|обращени[ея]|номер|№|#|id)\s*
    )
    [\w\-\/]{3,}
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Standalone long digit sequences (card-like, INN, SNILS, passport fragments)
LONG_DIGITS_RE = re.compile(r"\b\d{6,}\b")

# IP addresses
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# URLs
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# Russian filler / parasite words that carry no semantic value
FILLERS = {
    "ну", "вот", "так", "это", "типа", "короче", "значит", "собственно",
    "ладно", "хорошо", "понятно", "слушайте", "слушай", "скажите", "скажи",
    "просто", "буквально", "вообще", "кстати", "кажется", "наверное",
    "допустим", "предположим", "алло", "алё",
}
FILLERS_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FILLERS) + r")\b",
    re.IGNORECASE,
)

# Repeated chars (e.g. "ааааа", "!!!!")
REPEAT_CHARS_RE = re.compile(r"(.)\1{3,}")

# Multiple spaces / newlines
WHITESPACE_RE = re.compile(r"\s+")

# NER entity types to redact
NER_REDACT_TYPES = {
    "PER",   # persons
    "ORG",   # organisations / company names
    "LOC",   # locations / addresses
}

# Placeholder map
PLACEHOLDER = {
    "phone":    "[ТЕЛЕФОН]",
    "email":    "[EMAIL]",
    "contract": "[НОМЕР_ДОГОВОРА]",
    "digits":   "[НОМЕР]",
    "ip":       "[IP]",
    "url":      "[ССЫЛКА]",
    "PER":      "[ИМЯ]",
    "ORG":      "[ОРГАНИЗАЦИЯ]",
    "LOC":      "[АДРЕС]",
}


# ── Natasha pipeline (loaded once) ──────────────────────────────────────────

def _build_natasha():
    if not _NATASHA_OK:
        return None
    emb = NewsEmbedding()
    return {
        "segmenter":    Segmenter(),
        "morph_vocab":  MorphVocab(),
        "morph_tagger": NewsMorphTagger(emb),
        "ner_tagger":   NewsNERTagger(emb),
    }


def _ner_redact(text: str, nlp: dict) -> str:
    """Replace named entities (PER, ORG, LOC) with placeholders using Natasha."""
    doc = Doc(text)
    doc.segment(nlp["segmenter"])
    doc.tag_morph(nlp["morph_tagger"])
    doc.tag_ner(nlp["ner_tagger"])

    if not doc.spans:
        return text

    # Build redacted string by replacing spans from right to left
    # (so offsets stay valid)
    spans = sorted(
        [s for s in doc.spans if s.type in NER_REDACT_TYPES],
        key=lambda s: s.start,
        reverse=True,
    )
    chars = list(text)
    for span in spans:
        placeholder = PLACEHOLDER.get(span.type, "[СУЩНОСТЬ]")
        chars[span.start:span.stop] = list(placeholder)
    return "".join(chars)


# ── PII cleaning ────────────────────────────────────────────────────────────

def remove_pii_regex(text: str) -> str:
    """Apply regex-based PII substitutions."""
    text = URL_RE.sub(PLACEHOLDER["url"], text)
    text = EMAIL_RE.sub(PLACEHOLDER["email"], text)
    text = PHONE_RE.sub(PLACEHOLDER["phone"], text)
    text = IP_RE.sub(PLACEHOLDER["ip"], text)
    text = CONTRACT_RE.sub(lambda m: PLACEHOLDER["contract"], text)
    text = LONG_DIGITS_RE.sub(PLACEHOLDER["digits"], text)
    return text


# ── Text normalization ───────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Full normalization pipeline:
      - Unicode NFKC normalization
      - Strip non-printable chars
      - Lowercase
      - Remove/replace punctuation (keep sentence-meaningful ones)
      - Collapse repeated characters
      - Remove filler words
      - Collapse whitespace
    """
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove non-printable / control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    # 3. Lowercase
    text = text.lower()

    # 4. Replace punctuation that doesn't carry sentence meaning
    #    Keep: letters, digits, spaces, hyphens inside words, placeholders []
    text = re.sub(r"[\"\'«»„""\(\)\{\}\\|@#$%^&*=+<>~`]", " ", text)

    # 5. Collapse repeated characters (ааааа → аа, !!!! → !)
    text = REPEAT_CHARS_RE.sub(r"\1\1", text)

    # 6. Remove filler words
    text = FILLERS_RE.sub(" ", text)

    # 7. Collapse multiple spaces / newlines
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text


# ── Full pipeline per row ────────────────────────────────────────────────────

def process_text(text: str, nlp) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # Skip rows that already contain error markers from transcription
    if text.startswith("[TRANSCRIPTION_ERROR"):
        return text

    # Step 1: Regex PII
    text = remove_pii_regex(text)

    # Step 2: NER PII (names, orgs, addresses)
    if nlp is not None:
        try:
            text = _ner_redact(text, nlp)
        except Exception as exc:
            # NER failure is non-fatal — log and continue
            pass

    # Step 3: Normalize
    text = normalize(text)

    return text


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clean and normalize a transcription CSV for ML training."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input CSV (e.g. dataset.csv)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV (default: <input>_clean.csv)")
    parser.add_argument("--min-tokens", type=int, default=3,
                        help="Minimum token count after cleaning; "
                             "shorter rows get flagged in 'too_short' column (default: 3)")
    parser.add_argument("--no-ner", action="store_true",
                        help="Skip NER-based PII removal (faster, but misses names/orgs)")

    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else \
                  input_path.with_stem(input_path.stem + "_clean")

    if not input_path.exists():
        print(f"Error: input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading  : {input_path}")
    df = pd.read_csv(input_path, dtype=str).fillna("")

    if "text" not in df.columns:
        print("Error: CSV must contain a 'text' column.", file=sys.stderr)
        sys.exit(1)

    # Build NER pipeline once
    nlp = None
    if not args.no_ner:
        print("Loading Natasha NER model (one-time, ~2s)...")
        nlp = _build_natasha()
        if nlp:
            print("NER ready.")

    print(f"Processing {len(df)} rows...")
    tqdm.pandas(desc="Cleaning")

    df["text_original"] = df["text"]   # keep original for reference
    df["text"] = df["text"].progress_apply(lambda t: process_text(t, nlp))

    # Flag rows that are too short after cleaning
    df["too_short"] = df["text"].apply(
        lambda t: len(t.split()) < args.min_tokens
        if not t.startswith("[TRANSCRIPTION_ERROR")
        else False
    )

    n_short   = df["too_short"].sum()
    n_errors  = df["text"].str.startswith("[TRANSCRIPTION_ERROR").sum()
    n_empty   = (df["text"] == "").sum()
    n_ok      = len(df) - n_short - n_errors - n_empty

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n{'─'*50}")
    print(f"Done. Output : {output_path}")
    print(f"{'─'*50}")
    print(f"  Total rows        : {len(df)}")
    print(f"  Clean & ready     : {n_ok}")
    print(f"  Too short (<{args.min_tokens} tok): {n_short}  ← review 'too_short' column")
    print(f"  Empty after clean : {n_empty}")
    print(f"  Transcription err : {n_errors}")
    print(f"{'─'*50}")
    if n_short > 0:
        print("Tip: filter out too-short rows before training:")
        print("     df = df[~df['too_short']]")


if __name__ == "__main__":
    main()
