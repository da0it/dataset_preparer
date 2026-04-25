#!/usr/bin/env python3
"""Benchmark multiple LLMs on support-call routing prompts via OpenAI-compatible API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".txt", ".md", ".text", ".json"}
DEFAULT_BUCKETS = ("small", "medium", "large")
DEFAULT_REQUIRED_FIELDS = ("problem", "call_purpose", "priority", "assig_group")


@dataclass
class CaseItem:
    bucket: str
    path: Path
    case_id: str
    text: str


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "item"


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _estimate_token_count(text: str) -> int:
    # Rough heuristic for OpenAI-compatible APIs when usage is missing.
    parts = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(len(parts), 1) if text.strip() else 0


def _extract_text_from_json(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("text", "transcript", "content", "body", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError("JSON file must contain one of: text, transcript, content, body, message")


def load_case_text(path: Path) -> str:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _normalize_space(_extract_text_from_json(payload))
    return _normalize_space(path.read_text(encoding="utf-8"))


def discover_cases(root: Path, buckets: list[str]) -> list[CaseItem]:
    cases: list[CaseItem] = []
    seen_ids: set[str] = set()
    for bucket in buckets:
        bucket_dir = root / bucket
        if not bucket_dir.exists():
            raise FileNotFoundError(f"Bucket directory not found: {bucket_dir}")
        for path in sorted(bucket_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            text = load_case_text(path)
            if not text:
                continue
            case_id = f"{bucket}/{path.stem}"
            if case_id in seen_ids:
                raise ValueError(f"Duplicate case id: {case_id}")
            seen_ids.add(case_id)
            cases.append(CaseItem(bucket=bucket, path=path, case_id=case_id, text=text))
    if not cases:
        raise ValueError(f"No benchmark cases found under {root}")
    return cases


def build_system_prompt(required_fields: list[str], field_descriptions: dict[str, str]) -> str:
    lines = [
        "Ты анализируешь текст обращения в техническую поддержку.",
        "Верни только JSON-объект без markdown и без пояснений.",
        "Поля JSON:",
    ]
    for field in required_fields:
        desc = field_descriptions.get(field, "заполни значением или null")
        lines.append(f'- "{field}": {desc}')
    lines.append('Если поле невозможно определить, верни null.')
    return "\n".join(lines)


def build_user_prompt(text: str) -> str:
    return f"Текст обращения:\n{text}"


def strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def extract_json_object(raw: str) -> dict[str, Any] | None:
    raw = strip_code_fence(raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {"null", "none", "n/a", "unknown"}
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def load_references(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    refs: dict[str, list[str]] = {}
    if not isinstance(payload, dict):
        raise ValueError("reference manifest must be a JSON object")
    for case_id, value in payload.items():
        if isinstance(value, str):
            refs[case_id] = [value]
        elif isinstance(value, list):
            refs[case_id] = [str(item) for item in value if str(item).strip()]
        elif isinstance(value, dict):
            for key in ("problem", "problem_keywords", "expected_problem"):
                inner = value.get(key)
                if isinstance(inner, str):
                    refs[case_id] = [inner]
                    break
                if isinstance(inner, list):
                    refs[case_id] = [str(item) for item in inner if str(item).strip()]
                    break
    return refs


def normalize_text_for_match(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("ё", "е")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def evaluate_problem_ok(problem_value: Any, expected_refs: list[str] | None) -> bool:
    if not is_present(problem_value):
        return False
    problem_norm = normalize_text_for_match(str(problem_value))
    if not expected_refs:
        return True
    return all(normalize_text_for_match(ref) in problem_norm for ref in expected_refs if ref.strip())


def format_ratio(ok: int, total: int) -> str:
    return f"{ok}/{total}"


def call_openai_chat(
    *,
    api_base: str,
    api_key: str | None,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    force_json_mode: bool,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if force_json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def request_with_fallback(**kwargs) -> tuple[dict[str, Any], bool]:
    try:
        return call_openai_chat(force_json_mode=True, **kwargs), True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {400, 404, 422} and "response_format" in body:
            return call_openai_chat(force_json_mode=False, **kwargs), False
        raise


def parse_completion(response: dict[str, Any]) -> tuple[str, int]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("API response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "\n".join(text_parts)
    if not isinstance(content, str):
        raise ValueError("API response content is not a string")

    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        completion_tokens = _estimate_token_count(content)
    return content, completion_tokens


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_model(
    *,
    model: str,
    cases: list[CaseItem],
    api_base: str,
    api_key: str | None,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: float,
    required_fields: list[str],
    problem_field: str,
    references: dict[str, list[str]],
    raw_dir: Path,
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    print(f"\n{'=' * 60}")
    print(f"Модель: {model}")
    print(f"{'=' * 60}")

    total_latency = 0.0
    total_tokens = 0
    json_ok = 0
    problem_ok = 0
    completeness_sum = 0.0
    detailed_rows: list[dict[str, Any]] = []
    bucket_latency: dict[str, list[float]] = {}
    json_mode_supported = True

    model_raw_dir = raw_dir / _safe_name(model)
    model_raw_dir.mkdir(parents=True, exist_ok=True)

    for idx, case in enumerate(cases, start=1):
        print(f"  [{idx:>2}/{len(cases)}] {case.case_id}")
        start = time.perf_counter()
        response, used_json_mode = request_with_fallback(
            api_base=api_base,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=build_user_prompt(case.text),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        elapsed = time.perf_counter() - start
        json_mode_supported = json_mode_supported and used_json_mode

        raw_text, completion_tokens = parse_completion(response)
        total_latency += elapsed
        total_tokens += completion_tokens
        bucket_latency.setdefault(case.bucket, []).append(elapsed)

        parsed = extract_json_object(raw_text)
        row_json_ok = parsed is not None
        if row_json_ok:
            json_ok += 1

        completeness = 0.0
        row_problem_ok = False
        if parsed is not None:
            filled = sum(1 for field in required_fields if is_present(parsed.get(field)))
            completeness = filled / len(required_fields) if required_fields else 1.0
            row_problem_ok = evaluate_problem_ok(parsed.get(problem_field), references.get(case.case_id))
            if row_problem_ok:
                problem_ok += 1

        completeness_sum += completeness
        tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

        raw_payload = {
            "case_id": case.case_id,
            "bucket": case.bucket,
            "model": model,
            "elapsed_sec": round(elapsed, 6),
            "completion_tokens": completion_tokens,
            "tokens_per_sec": round(tokens_per_sec, 4),
            "json_ok": row_json_ok,
            "problem_ok": row_problem_ok,
            "completeness": round(completeness, 6),
            "raw_response": raw_text,
            "parsed_json": parsed,
        }
        (model_raw_dir / f"{_safe_name(case.case_id)}.json").write_text(
            json.dumps(raw_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        detailed_rows.append({
            "model": model,
            "bucket": case.bucket,
            "case_id": case.case_id,
            "elapsed_sec": round(elapsed, 6),
            "completion_tokens": completion_tokens,
            "tokens_per_sec": round(tokens_per_sec, 4),
            "json_ok": int(row_json_ok),
            "problem_ok": int(row_problem_ok),
            "completeness": round(completeness, 6),
        })

    total_cases = len(cases)
    summary_row = {
        "model": model,
        "avg_response_sec": round(total_latency / total_cases, 3) if total_cases else 0.0,
        "tokens_per_sec": round(total_tokens / total_latency, 2) if total_latency > 0 else 0.0,
        "completeness": round(completeness_sum / total_cases, 3) if total_cases else 0.0,
        "json_ok": format_ratio(json_ok, total_cases),
        "problem_ok": format_ratio(problem_ok, total_cases),
    }
    bucket_row = {
        bucket: round(sum(times) / len(times), 3)
        for bucket, times in bucket_latency.items()
        if times
    }
    meta = {
        "json_mode_supported": json_mode_supported,
        "json_ok_count": json_ok,
        "problem_ok_count": problem_ok,
        "total_cases": total_cases,
    }
    return summary_row, bucket_row, detailed_rows, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI-compatible/Ollama LLMs on small/medium/large support-call cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_benchmark.py \\
    --input-root ./llm_cases \\
    --models qwen2.5:7b qwen2.5:14b granite3.2:8b gemma3:4b \\
    --api-base http://localhost:11434/v1 \\
    -o ./llm_benchmark_out

Optional reference manifest format:
  {
    "small/case_01": ["сбросить пароль", "не приходит письмо"],
    "large/case_05": {"problem_keywords": ["лицензия", "ядра"]}
  }
        """,
    )
    parser.add_argument("--input-root", "-i", required=True, help="Root folder with subfolders small/medium/large.")
    parser.add_argument("--models", nargs="+", required=True, help="Model ids to benchmark.")
    parser.add_argument("--api-base", default="http://localhost:11434/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("OLLAMA_API_KEY"),
                        help="API key if required. Defaults to OPENAI_API_KEY / OLLAMA_API_KEY.")
    parser.add_argument("--output", "-o", default=None, help="Output directory. Default: llm_benchmark_<timestamp>.")
    parser.add_argument("--buckets", default="small,medium,large",
                        help="Comma-separated bucket names (default: small,medium,large).")
    parser.add_argument("--required-fields", default="problem,call_purpose,priority,assig_group",
                        help="Comma-separated required JSON fields for completeness metric.")
    parser.add_argument("--problem-field", default="problem", help="Field used for Problem OK metric.")
    parser.add_argument("--reference-manifest", default=None,
                        help="Optional JSON with expected problem keywords per case.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0).")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max completion tokens (default: 256).")
    parser.add_argument("--timeout-sec", type=float, default=180.0, help="HTTP timeout per request (default: 180).")
    parser.add_argument("--system-prompt-file", default=None,
                        help="Optional custom system prompt file. If omitted, a built-in router prompt is used.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        print(f"Error: input root not found: {input_root}")
        return 1

    output_dir = Path(args.output).resolve() if args.output else Path(
        f"llm_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    buckets = [bucket.strip() for bucket in args.buckets.split(",") if bucket.strip()]
    required_fields = [field.strip() for field in args.required_fields.split(",") if field.strip()]
    references = load_references(Path(args.reference_manifest).resolve()) if args.reference_manifest else {}

    field_descriptions = {
        "problem": "кратко и по существу сформулируй основную проблему или запрос клиента",
        "call_purpose": "цель звонка/тип обращения",
        "priority": "приоритет обращения",
        "assig_group": "группа или команда, которой стоит передать обращение",
    }
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    else:
        system_prompt = build_system_prompt(required_fields, field_descriptions)

    try:
        cases = discover_cases(input_root, buckets)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Input root : {input_root}")
    print(f"Output dir : {output_dir}")
    print(f"API base   : {args.api_base}")
    print(f"Buckets    : {', '.join(buckets)}")
    print(f"Cases      : {len(cases)}")
    for bucket in buckets:
        count = sum(1 for case in cases if case.bucket == bucket)
        print(f"  {bucket:<10} {count}")
    if references:
        print(f"Problem refs: {len(references)} cases from {Path(args.reference_manifest).resolve()}")

    summary_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    meta_models: dict[str, Any] = {}

    for model in args.models:
        try:
            summary_row, bucket_row, model_details, meta = benchmark_model(
                model=model,
                cases=cases,
                api_base=args.api_base,
                api_key=args.api_key,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                required_fields=required_fields,
                problem_field=args.problem_field,
                references=references,
                raw_dir=raw_dir,
            )
        except Exception as exc:
            print(f"\n[ERROR] {model}: {exc}")
            continue

        summary_rows.append(summary_row)
        bucket_rows.append({"model": model, **{bucket: bucket_row.get(bucket, "") for bucket in buckets}})
        detailed_rows.extend(model_details)
        meta_models[model] = meta

    if not summary_rows:
        print("\nNo successful model runs.")
        return 1

    summary_rows.sort(key=lambda row: float(row["avg_response_sec"]))
    bucket_rows.sort(key=lambda row: str(row["model"]))
    detailed_rows.sort(key=lambda row: (row["model"], row["bucket"], row["case_id"]))

    summary_csv = output_dir / "llm_summary.csv"
    buckets_csv = output_dir / "llm_case_times.csv"
    detailed_csv = output_dir / "llm_detailed.csv"
    summary_json = output_dir / "llm_summary.json"

    write_csv(
        summary_csv,
        summary_rows,
        ["model", "avg_response_sec", "tokens_per_sec", "completeness", "json_ok", "problem_ok"],
    )
    write_csv(
        buckets_csv,
        bucket_rows,
        ["model", *buckets],
    )
    write_csv(
        detailed_csv,
        detailed_rows,
        ["model", "bucket", "case_id", "elapsed_sec", "completion_tokens", "tokens_per_sec", "json_ok", "problem_ok", "completeness"],
    )

    payload = {
        "input_root": str(input_root),
        "api_base": args.api_base,
        "models": args.models,
        "buckets": buckets,
        "required_fields": required_fields,
        "problem_field": args.problem_field,
        "reference_manifest": str(Path(args.reference_manifest).resolve()) if args.reference_manifest else None,
        "results": summary_rows,
        "case_times": bucket_rows,
        "meta": meta_models,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("ИТОГОВАЯ СВОДКА")
    print(f"{'=' * 72}")
    print_table(
        ["Модель", "Среднее время ответа, с", "Скорость, ток/с", "Полнота", "JSON OK", "Problem OK"],
        [
            [
                str(row["model"]),
                f'{float(row["avg_response_sec"]):.3f}',
                f'{float(row["tokens_per_sec"]):.2f}',
                f'{float(row["completeness"]):.3f}',
                str(row["json_ok"]),
                str(row["problem_ok"]),
            ]
            for row in summary_rows
        ],
    )

    print(f"\n{'=' * 72}")
    print("ПО КЕЙСАМ")
    print(f"{'=' * 72}")
    print_table(
        ["Модель", *[f"{bucket}, с" for bucket in buckets]],
        [
            [str(row["model"]), *[
                f'{float(row[bucket]):.3f}' if str(row.get(bucket, "")).strip() else ""
                for bucket in buckets
            ]]
            for row in bucket_rows
        ],
    )

    print(f"\nSaved summary : {summary_csv}")
    print(f"Saved times   : {buckets_csv}")
    print(f"Saved details : {detailed_csv}")
    print(f"Saved json    : {summary_json}")
    print(f"Saved raw     : {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
