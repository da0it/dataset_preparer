#!/usr/bin/env python3
"""
Dataset preparation script for ML training (Russian call-center transcriptions).

Steps applied to each text:
  1. PII removal  — phones, emails, IPs, URLs, contract numbers
                    → replaced with readable [PLACEHOLDER] tokens
  2. NER redaction — person names, org names, locations via Stanza
                     + pymorphy3 dictionary fallback for names
  3. Optional lemmatization — stanza tokenize+pos+lemma
  4. Normalization — unicode cleanup, lowercasing, whitespace collapse
                     (punctuation and meaningful words are preserved)
  5. Length filter — rows with too little text after cleaning are flagged

Design principles for Russian transcription data:
  - Punctuation inside text is kept (helps with sentence structure)
  - Common Russian words (это, так, вот, хорошо...) are NOT removed —
    they carry meaning in context ("это не работает", "так и не решили")
  - Quotes are preserved — product names are often quoted («Арендата»)
  - Only provably-PII patterns are replaced, not heuristic guesses
  - Name redaction requires pymorphy3 NOM-form match to avoid false positives
    on homonyms (вера/надежда as common nouns, виктор as adjective etc.)

Usage:
    python prepare_dataset.py --input dataset.csv --output dataset_clean.csv
    python prepare_dataset.py --input dataset.csv --output dataset_clean.csv --min-tokens 5
    python prepare_dataset.py --input dataset.csv --output dataset_clean_lemma.csv --lemmatize

    # Re-run cleaning on already-cleaned file, starting from row 223 (0-based),
    # reading source text from 'text_original' column:
    python prepare_dataset.py --input dataset_clean.csv --output dataset_clean_v2.csv \\
        --source-column text_original --start-row 223

Dependencies:
    pip install stanza pymorphy3 pymorphy3-dicts-ru pandas tqdm
    python -c "import stanza; stanza.download('ru')"
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Stanza (Russian NER / lemmatization) ─────────────────────────────────────
try:
    import stanza
    from stanza import DownloadMethod
    _STANZA_IMPORT_OK = True
    _STANZA_IMPORT_ERROR = None
except Exception as e:
    _STANZA_IMPORT_OK = False
    _STANZA_IMPORT_ERROR = e
    stanza = None
    DownloadMethod = None
    print(
        f"Warning: stanza not available ({e}).\n"
        "NER-based PII removal and stanza lemmatization will be skipped.\n"
        'Install: pip install stanza && python -c "import stanza; stanza.download(\'ru\')"',
        file=sys.stderr,
    )

_nlp_stanza_ner = None
_nlp_stanza_lemma = None
_STANZA_NER_ERROR = None
_STANZA_LEMMA_ERROR = None


def _load_stanza_pipeline(processors: str):
    if not _STANZA_IMPORT_OK:
        raise RuntimeError(_STANZA_IMPORT_ERROR)
    return stanza.Pipeline(
        "ru",
        processors=processors,
        download_method=DownloadMethod.REUSE_RESOURCES,
        verbose=False,
    )


def _get_stanza_ner_pipeline():
    global _nlp_stanza_ner, _STANZA_NER_ERROR
    if _nlp_stanza_ner is None and _STANZA_NER_ERROR is None:
        try:
            _nlp_stanza_ner = _load_stanza_pipeline("tokenize,ner")
        except Exception as exc:
            _STANZA_NER_ERROR = exc
    return _nlp_stanza_ner


def _get_stanza_lemma_pipeline():
    global _nlp_stanza_lemma, _STANZA_LEMMA_ERROR
    if _nlp_stanza_lemma is None and _STANZA_LEMMA_ERROR is None:
        try:
            _nlp_stanza_lemma = _load_stanza_pipeline("tokenize,pos,lemma")
        except Exception as exc:
            _STANZA_LEMMA_ERROR = exc
    return _nlp_stanza_lemma

# ── pymorphy3 (lemmatization for name dictionary lookup) ────────────────────
try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
    _MORPH_OK = True
except ImportError:
    _morph = None
    _MORPH_OK = False
    print(
        "Warning: pymorphy3 not installed. Name lemmatization will be skipped.\n"
        "Install: pip install pymorphy3 pymorphy3-dicts-ru",
        file=sys.stderr,
    )

# ── PII regex patterns ───────────────────────────────────────────────────────

# Standard phone: +7 999 123-45-67 / 8(999)123-45-67
# Requires at least area code + 7 digits to avoid false positives on short numbers
PHONE_RE = re.compile(
    r"(?:\+?7|8)[\s\-\(]*\d{3}[\s\-\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}",
)

# Phone dictated digit by digit: "1-7-8-7-…" — min 7 digit tokens to reduce false positives
# (e.g. "версия 2 0 0 1" only has 4 tokens → won't match)
PHONE_DICTATED_DIGITS_RE = re.compile(
    r"\b(?:\d[\s\-]){6,}\d\b",
)

# Phone dictated as Russian words: один два три ... (min 7 words)
_DIGIT_WORDS = "ноль|один|два|три|четыре|пять|шесть|семь|восемь|девять"
PHONE_DICTATED_WORDS_RE = re.compile(
    r"\b(?:(?:" + _DIGIT_WORDS + r")[\s,\-]*){7,}",
    re.IGNORECASE,
)

# Standard email
EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}",
)

# Email dictated letter by letter — explicit @ marker required
EMAIL_DICTATED_RE = re.compile(
    r"(?:[а-яёa-z](?:\s+(?:русская|латинская|заглавная|маленькая))?\s+){2,}"
    r"(?:как\s+)?(?:собака|доллар|at|эт)"
    r"(?:\s+[а-яёa-z]){1,}"
    r"(?:\s+точка\s+[а-яёa-z]{2,4})?",
    re.IGNORECASE,
)

# Contract / ticket numbers — keyword must immediately precede alphanumeric ID
# Anchored tighter: keyword + optional whitespace + ID that starts with a digit or letter
# "номер версии 2.0" won't match because "версии" ≠ keyword
_CONTRACT_KEYWORDS = "|".join([
    "договор[а-я]*", "контракт[а-я]*", "тикет[а-я]*",
    r"заявк[аи]", r"обращени[ея]",
    "номер", "№", r"#\s*(?=\w)", r"\bid\b",
])
CONTRACT_RE = re.compile(
    r"(?:" + _CONTRACT_KEYWORDS + r")\s*[A-ZА-ЯЁa-zа-яё]?[\d][\w\-\/]{2,}",
    re.IGNORECASE,
)

# Standalone long digit sequences — raised threshold to 8+ digits
# 6-digit sequences hit version numbers, postal codes too often
LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")

# IP addresses
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# URLs
URL_RE = re.compile(r"https?://\S+|www\.\S+")

# Repeated chars (e.g. "ааааа", "!!!!") — collapse to max 2 repetitions
REPEAT_CHARS_RE = re.compile(r"(.)\1{3,}")

# Multiple spaces / newlines → single space
WHITESPACE_RE = re.compile(r"\s+")

# Stanza entity labels → placeholder keys
STANZA_LABEL_MAP = {
    "PER":    "PER",
    "PERSON": "PER",
    "ORG":    "ORG",
    "LOC":    "LOC",
    "GPE":    "LOC",
}

# Placeholder tokens (human-readable, survive lowercasing in downstream steps)
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


# ── Russian first names dictionary (nominative lemmas) ───────────────────────
#
# Intentionally includes homonyms (вера, надежда, виктория и т.д.):
# в транскрипциях колл-центра слово почти всегда является именем, а не
# нарицательным существительным. Ложное срабатывание (скрыть "надежда" в
# редком философском контексте) менее опасно, чем пропустить реальное имя.
#
_RUSSIAN_NAME_LEMMAS = frozenset({
    # Male
    "александр", "алексей", "андрей", "антон", "артём", "борис", "вадим",
    "валентин", "валерий", "василий", "виктор", "виталий", "владимир",
    "владислав", "вячеслав", "геннадий", "георгий", "григорий", "даниил",
    "денис", "дмитрий", "евгений", "иван", "игорь", "илья", "кирилл",
    "константин", "леонид", "максим", "михаил", "никита", "николай", "олег",
    "павел", "пётр", "роман", "руслан", "сергей", "степан", "тимур", "фёдор",
    "филипп", "юрий", "яков", "ярослав",
    # Female
    "александра", "алина", "алла", "анастасия", "анна", "валентина",
    "валерия", "вера", "виктория", "галина", "дарья", "диана", "екатерина",
    "елена", "жанна", "зинаида", "инна", "ирина", "карина", "кристина",
    "ксения", "лариса", "людмила", "маргарита", "марина", "мария", "надежда",
    "наталья", "нина", "оксана", "ольга", "полина", "светлана", "софия",
    "тамара", "татьяна", "юлия",
    # Diminutives
    "саша", "паша", "вася", "коля", "миша", "серёжа", "женя", "дима",
    "лена", "аня", "оля", "катя", "таня", "наташа", "юля", "маша", "света",
    "настя", "даша", "лёша", "вова", "макс", "рома",
})


def _is_name_lemma(word: str) -> bool:
    """
    True если лемма слова совпадает с именем из словаря.
    Проверка идёт только по словарю — без фильтрации по тегу pymorphy3,
    чтобы не пропускать имена-омонимы (вера, надежда и т.д.).
    """
    if not _MORPH_OK:
        return word.lower() in _RUSSIAN_NAME_LEMMAS
    parses = _morph.parse(word)
    if not parses:
        return False
    return parses[0].normal_form in _RUSSIAN_NAME_LEMMAS


# ── Org regex fallback ───────────────────────────────────────────────────────

# Pattern 1: юридическая форма + название (ООО Ромашка, ИП Петров)
_ORG_LEGAL_RE = re.compile(
    r"\b(?:ооо|оао|зао|пао|ип|ао|нко|фгуп|мкп|гбу|гуп)"
    r"\s+[«\"]?[А-ЯЁа-яё\w][\w\s\-«»\"]{1,40}[»\"]?",
    re.IGNORECASE,
)

# Pattern 2: триггер + название — только если название начинается с заглавной
# (снижает ложные срабатывания на обычные фразы типа "компания хочет")
_ORG_TRIGGER_KEYWORDS = "|".join([
    "компани[яи]", "компанию", "фирм[аы]", "фирму",
    "организаци[яи]", "организацию",
    "предприяти[яе]", "предприятию",
    "работодател[ьея]",
    "заказчик[аи]?", "поставщик[аи]?", "подрядчик[аи]?",
    r"холдинг(?:е|а|у|ом)?",
    r"групп[аыу]",
])
_ORG_TRIGGER_RE = re.compile(
    r"\b(?:" + _ORG_TRIGGER_KEYWORDS + r")\s+"
    r"[«\"]?([А-ЯЁ][А-ЯЁа-яё0-9\-]{1,30}"        # ← starts with CAPITAL letter
    r"(?:\s+[А-ЯЁа-яё0-9\-]{1,30}){0,3})"
    r"[»\"]?",
)


def _org_re_sub(text: str) -> str:
    text = _ORG_LEGAL_RE.sub(PLACEHOLDER["ORG"], text)
    text = _ORG_TRIGGER_RE.sub(
        lambda m: m.group(0).replace(m.group(1), PLACEHOLDER["ORG"]),
        text,
    )
    return text


# ── Stanza NER redaction ─────────────────────────────────────────────────────

def _ner_redact_stanza(text: str) -> str:
    """
    Replace named entities (PER, ORG, LOC) using stanza wikiner-ru model.
    Applied before lowercasing so capitalisation hints work properly.
    Replacements applied right-to-left to keep offsets valid.
    """
    nlp = _get_stanza_ner_pipeline()
    if nlp is None:
        return text
    doc = nlp(text)
    entities = []
    for sent in doc.sentences:
        for ent in sent.entities:
            if ent.type in STANZA_LABEL_MAP:
                entities.append((ent.start_char, ent.end_char, ent.type))

    entities.sort(key=lambda e: e[0], reverse=True)
    chars = list(text)
    for start, end, label in entities:
        ph = PLACEHOLDER.get(STANZA_LABEL_MAP[label], "[СУЩНОСТЬ]")
        chars[start:end] = list(ph)
    return "".join(chars)


# ── pymorphy3 name fallback ──────────────────────────────────────────────────

_WORD_RE = re.compile(r"[а-яёА-ЯЁ]{2,}")


def _name_lemma_redact(text: str) -> str:
    """
    Walk every Cyrillic token and replace confirmed first names with [ИМЯ].
    Requires pymorphy3 Name tag — avoids replacing homonym common nouns.
    Applied right-to-left so character positions stay valid.
    """
    matches = list(_WORD_RE.finditer(text))
    chars = list(text)
    for m in reversed(matches):
        if _is_name_lemma(m.group()):
            chars[m.start():m.end()] = list(PLACEHOLDER["PER"])
    return "".join(chars)


def _regex_fallback_redact(text: str) -> str:
    """Catch orgs / names that Stanza missed."""
    text = _org_re_sub(text)
    text = _name_lemma_redact(text)
    return text


_PLACEHOLDER_ESCAPE_MAP = {
    placeholder: f"codexplaceholder{idx}"
    for idx, placeholder in enumerate(sorted(set(PLACEHOLDER.values())))
}
_PLACEHOLDER_RESTORE_MAP = {
    marker: placeholder
    for placeholder, marker in _PLACEHOLDER_ESCAPE_MAP.items()
}


def _escape_placeholders(text: str) -> str:
    escaped = text
    for placeholder, marker in _PLACEHOLDER_ESCAPE_MAP.items():
        escaped = escaped.replace(placeholder, marker)
    return escaped


def _restore_placeholders(text: str) -> str:
    restored = text
    for marker, placeholder in _PLACEHOLDER_RESTORE_MAP.items():
        restored = restored.replace(marker, placeholder)
    return restored


def _lemmatize_stanza(text: str) -> str:
    """
    Lemmatize only token spans and keep original punctuation / spacing intact.
    Placeholder markers are temporarily escaped so stanza does not split them.
    """
    nlp = _get_stanza_lemma_pipeline()
    if nlp is None:
        return text

    escaped = _escape_placeholders(text)
    doc = nlp(escaped)
    chars = list(escaped)
    replacements = []

    for sent in doc.sentences:
        for token in sent.tokens:
            start = getattr(token, "start_char", None)
            end = getattr(token, "end_char", None)
            if start is None or end is None:
                continue
            if token.text in _PLACEHOLDER_RESTORE_MAP:
                continue

            lemmas = []
            for word in token.words:
                lemma = getattr(word, "lemma", None) or getattr(word, "text", "")
                if lemma:
                    lemmas.append(lemma)
            replacement = " ".join(lemmas).strip()
            if replacement and replacement != token.text:
                replacements.append((start, end, replacement))

    for start, end, replacement in reversed(replacements):
        chars[start:end] = list(replacement)

    return _restore_placeholders("".join(chars))


# ── PII cleaning ─────────────────────────────────────────────────────────────

def remove_pii_regex(text: str) -> str:
    text = URL_RE.sub(PLACEHOLDER["url"], text)
    text = EMAIL_RE.sub(PLACEHOLDER["email"], text)
    text = EMAIL_DICTATED_RE.sub(PLACEHOLDER["email"], text)
    text = PHONE_RE.sub(PLACEHOLDER["phone"], text)
    text = PHONE_DICTATED_DIGITS_RE.sub(PLACEHOLDER["phone"], text)
    text = PHONE_DICTATED_WORDS_RE.sub(PLACEHOLDER["phone"], text)
    text = IP_RE.sub(PLACEHOLDER["ip"], text)
    text = CONTRACT_RE.sub(PLACEHOLDER["contract"], text)
    text = LONG_DIGITS_RE.sub(PLACEHOLDER["digits"], text)
    return text


# ── Text normalization ────────────────────────────────────────────────────────
#
# What we do:
#   • Unicode NFKC — canonical form, removes zero-width chars etc.
#   • Strip Unicode control characters (category C*)
#   • Lowercase
#   • Collapse repeated characters (аааа → аа, !!!! → !!)
#   • Collapse whitespace
#
# What we deliberately do NOT do:
#   • Remove punctuation — commas, dots help with sentence boundaries
#   • Remove «» quotes — product/company names are often quoted
#   • Remove meaningful words (это, так, хорошо, понятно…) — they carry
#     context in Russian ("это не работает" ≠ "не работает")

def normalize(text: str) -> str:
    # 1. Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # 2. Strip control characters (but keep newlines → spaces below)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # 3. Lowercase
    text = text.lower()
    # 4. Remove characters that are never meaningful in transcriptions
    #    (technical symbols, not quotes/punctuation)
    text = re.sub(r"[\\|@#$%^&*=+<>~`]", " ", text)
    # 5. Collapse repeated characters: аааа → аа
    text = REPEAT_CHARS_RE.sub(r"\1\1", text)
    # 6. Collapse whitespace
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


# ── Full pipeline per row ─────────────────────────────────────────────────────

def process_text(text: str, use_ner: bool, lemmatize: bool) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    if text.startswith("[TRANSCRIPTION_ERROR"):
        return text

    # Step 1: Regex PII — before NER so placeholders don't confuse the model
    text = remove_pii_regex(text)

    # Step 2: Stanza NER — must run BEFORE lowercasing (capitalisation matters)
    if use_ner and _get_stanza_ner_pipeline() is not None:
        try:
            text = _ner_redact_stanza(text)
        except Exception as exc:
            print(f"Warning: Stanza failed on a row ({exc}), skipping NER for it.",
                  file=sys.stderr)

    # Step 3: Regex + pymorphy3 fallback for names/orgs Stanza missed
    text = _regex_fallback_redact(text)

    # Step 4: Optional lemmatization on cleaned text, before lowercasing/cleanup
    if lemmatize:
        try:
            text = _lemmatize_stanza(text)
        except Exception as exc:
            print(f"Warning: Stanza lemmatization failed on a row ({exc}), skipping it.",
                  file=sys.stderr)

    # Step 5: Normalize (lowercase + cleanup) — after NER/lemma
    text = normalize(text)

    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clean and normalize a Russian transcription CSV for ML training."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input CSV (e.g. dataset.csv)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV (default: <input>_clean.csv)")
    parser.add_argument("--min-tokens", type=int, default=3,
                        help="Minimum token count after cleaning (default: 3)")
    parser.add_argument("--no-ner", action="store_true",
                        help="Skip NER-based PII removal (faster, but misses names/orgs)")
    parser.add_argument("--lemmatize", action="store_true",
                        help="Apply stanza lemmatization to the cleaned text before saving it.")
    parser.add_argument("--source-column", default="text",
                        help="Column to read source text from (default: text). "
                             "Pass 'text_original' to re-clean from raw transcription.")
    parser.add_argument("--start-row", type=int, default=0,
                        help="0-based row index to start processing from (default: 0). "
                             "Rows before this index are copied to output as-is.")
    parser.add_argument("--sep", default=",",
                        help="CSV field separator (default: comma). Use ';' for "
                             "semicolon-delimited files.")

    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else \
                  input_path.with_stem(input_path.stem + "_clean")

    if not input_path.exists():
        print(f"Error: input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading  : {input_path}")
    df = pd.read_csv(input_path, sep=args.sep, dtype=str).fillna("")

    if "text" not in df.columns:
        print("Error: CSV must contain a 'text' column.", file=sys.stderr)
        sys.exit(1)

    src_col = args.source_column
    if src_col not in df.columns:
        print(f"Error: source column '{src_col}' not found in CSV. "
              f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    start_row = args.start_row
    if start_row < 0 or start_row > len(df):
        print(f"Error: --start-row {start_row} is out of range (0..{len(df)}).",
              file=sys.stderr)
        sys.exit(1)

    use_ner = not args.no_ner
    if use_ner:
        if _get_stanza_ner_pipeline() is not None:
            print("Stanza NER ready (ru wikiner).")
        else:
            print(f"Warning: Stanza NER not available ({_STANZA_NER_ERROR}), NER skipped.",
                  file=sys.stderr)
    if args.lemmatize:
        if _get_stanza_lemma_pipeline() is not None:
            print("Stanza lemmatizer ready (ru tokenize+pos+lemma).")
        else:
            print(
                f"Warning: Stanza lemmatizer not available ({_STANZA_LEMMA_ERROR}), "
                "lemmatization skipped.",
                file=sys.stderr,
            )
    if _MORPH_OK:
        print("pymorphy3 lemmatizer ready.")

    n_total   = len(df)
    n_skip    = start_row
    n_process = n_total - n_skip

    if n_skip > 0:
        print(f"Rows 0..{n_skip - 1} ({n_skip} rows): kept as-is.")
    print(f"Rows {n_skip}..{n_total - 1} ({n_process} rows): processing from '{src_col}'...")
    tqdm.pandas(desc="Cleaning")

    if "text_original" not in df.columns:
        df["text_original"] = ""

    df.loc[df.index[start_row:], "text_original"] = df.loc[df.index[start_row:], src_col]

    df.loc[df.index[start_row:], "text"] = (
        df.loc[df.index[start_row:], src_col]
        .progress_apply(lambda t: process_text(t, use_ner, args.lemmatize))
    )

    if "too_short" not in df.columns:
        df["too_short"] = ""

    df.loc[df.index[start_row:], "too_short"] = df.loc[df.index[start_row:], "text"].apply(
        lambda t: str(len(t.split()) < args.min_tokens)
        if not t.startswith("[TRANSCRIPTION_ERROR")
        else "False"
    )

    n_short  = (df.loc[df.index[start_row:], "too_short"] == "True").sum()
    n_errors = df.loc[df.index[start_row:], "text"].str.startswith("[TRANSCRIPTION_ERROR").sum()
    n_empty  = (df.loc[df.index[start_row:], "text"] == "").sum()
    n_ok     = n_process - n_short - n_errors - n_empty

    df.to_csv(output_path, index=False, encoding="utf-8", sep=args.sep)

    print(f"\n{'─'*50}")
    print(f"Done. Output : {output_path}")
    print(f"{'─'*50}")
    print(f"  Total rows        : {n_total}")
    print(f"  Skipped (kept)    : {n_skip}  (rows 0..{n_skip - 1})" if n_skip else
          f"  Skipped (kept)    : 0")
    print(f"  Processed         : {n_process}")
    print(f"    Clean & ready   : {n_ok}")
    print(f"    Too short (<{args.min_tokens} tok): {n_short}  ← review 'too_short' column")
    print(f"    Empty after clean: {n_empty}")
    print(f"    Transcription err: {n_errors}")
    print(f"{'─'*50}")
    if n_short > 0:
        print("Tip: filter out too-short rows before training:")
        print("     df = df[~df['too_short'].astype(str).str.lower().eq('true')]")


if __name__ == "__main__":
    main()
