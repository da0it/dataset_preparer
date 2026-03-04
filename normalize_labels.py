#!/usr/bin/env python3
"""
Normalize label columns in the dataset CSV.

- Strips whitespace, lowercases all label values
- Applies known alias mappings (typos, inconsistent naming)
- Prints rows with suspicious/unknown values for manual review
- Saves result to a new file

Usage:
    python normalize_labels.py --input dataset_clean_handcsv.csv --output dataset_norm.csv --sep ";"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Alias maps (old value → canonical value) ────────────────────────────────
# All comparisons are done after strip+lower, so no need to list case variants.

PRIORITY_MAP = {
    "high":   "high",
    "medium": "medium",
    "low":    "low",
}

CALL_PURPOSE_MAP = {
    "consulting":       "consulting",
    "cooperation":      "cooperation",
    "courses":          "courses",
    "customer_support": "customer_support",
    "general":          "general",
    "hr":               "hr",
    "license":          "license",
    "ned":              "need",           # likely typo
    "public_events":    "public_events",
    "spam":             "spam",
    "triage":           "triage",
    "portal_access":    "portal_access",
}

ASSIGNED_GROUP_MAP = {
    "education_center":         "education_center",
    "general":                  "general",
    "hr_team":                  "hr_team",
    "sales":                    "sales",
    "consulting.general":       "consulting.general",
    "consulting.greenplum":     "consulting.greenplum",
    "consulting.hadoop":        "consulting.hadoop",
    "cooperation_department":   "cooperation_department",
    "customer_suppot.manager":  "customer_support.manager",  # typo fix
    "customer.manager":         "customer_support.manager",
    "marketing_department":     "marketing_department",
    "portal_support":           "portal_support",
    "public_events.conference": "public_events.conference",
}

# Values to treat as empty/unknown
SUSPICIOUS = {"?", "-", "ge", ""}


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
            result[i] = normalized  # keep as-is but flag
            unknown.add(normalized)

    if unknown:
        print(f"\n⚠️  [{col_name}] Unknown values (kept as-is, review manually):")
        for v in sorted(unknown):
            count = (series.str.strip().str.lower() == v).sum()
            print(f"   {count:3d}  '{v}'")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Normalize label columns (call_purpose, priority, assigned_group)."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input CSV")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV (default: <input>_norm.csv)")
    parser.add_argument("--sep", default=",",
                        help="CSV separator (default: comma). Use ';' if needed.")
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else \
                  input_path.with_stem(input_path.stem + "_norm")

    if not input_path.exists():
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path, sep=args.sep, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from {input_path.name}")

    # Show suspicious rows before normalization
    for col in ["call_purpose", "priority", "assigned_group"]:
        if col not in df.columns:
            continue
        mask = df[col].str.strip().str.lower().isin(SUSPICIOUS)
        if mask.any():
            print(f"\n⚠️  [{col}] Suspicious values (will be cleared):")
            for idx, row in df[mask].iterrows():
                print(f"   row {idx:4d}: '{row[col]}'  |  {str(row.get('text', ''))[:80]}")

    # Normalize each label column
    for col, alias_map in [
        ("priority",       PRIORITY_MAP),
        ("call_purpose",   CALL_PURPOSE_MAP),
        ("assigned_group", ASSIGNED_GROUP_MAP),
    ]:
        if col not in df.columns:
            print(f"\nSkipping '{col}' — column not found.")
            continue
        df[col] = normalize_col(df[col], alias_map, col)
        print(f"\n── {col} after normalization ──")
        vc = df[col][df[col] != ""].value_counts()
        for val, cnt in vc.items():
            print(f"   {cnt:3d}  {val}")
        empty = (df[col] == "").sum()
        if empty:
            print(f"   {empty:3d}  (empty / cleared)")

    df.to_csv(output_path, index=False, encoding="utf-8", sep=args.sep)
    print(f"\n✅ Saved to {output_path}")


if __name__ == "__main__":
    main()
