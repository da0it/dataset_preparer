#!/usr/bin/env python3
"""
Transcribe MP3 files using WhisperX and save results to two CSVs for manual annotation.

Always produces two output files simultaneously (one transcription pass per file):
  - dataset.csv          — plain text (all segments joined into one line)
  - dataset_timed.csv    — text with timestamps: [0.0-3.5] segment | [3.5-7.2] segment ...

The output filename is derived from --output; the timed variant gets a "_timed" suffix.

Usage:
    python transcribe.py --input /path/to/mp3s
    python transcribe.py --input /path/to/mp3s --output my_dataset.csv

Columns in both CSVs:
    filename, text, is_training_sample, call_purpose, priority, assigned_group
"""

import argparse
import csv
import sys
from pathlib import Path

import whisperx

CSV_COLUMNS = ["filename", "text", "is_training_sample", "call_purpose", "priority", "assigned_group"]
DEFAULT_MODEL = "large-v2"
DEFAULT_LANGUAGE = "ru"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def get_device() -> str:
    """Detect available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def timed_output_path(base: Path) -> Path:
    """Return the '_timed' variant of a CSV path."""
    return base.with_stem(base.stem + "_timed")


def load_existing_processed(csv_path: Path) -> set[str]:
    """Return set of filenames already present in the CSV."""
    processed: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["filename"])
    return processed


def segments_to_texts(segments: list[dict]) -> tuple[str, str]:
    """
    Given a list of WhisperX segments return:
      plain_text  — all text joined with spaces
      timed_text  — segments formatted as [start-end] text, joined with " | "
    """
    plain_parts: list[str] = []
    timed_parts: list[str] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        plain_parts.append(text)
        timed_parts.append(f"[{start:.1f}-{end:.1f}] {text}")
    return " ".join(plain_parts).strip(), " | ".join(timed_parts)


def transcribe_file(audio_path: Path, model, language: str) -> tuple[str, str]:
    """Transcribe a single audio file; return (plain_text, timed_text)."""
    result = model.transcribe(str(audio_path), language=language, batch_size=16)
    segments = result.get("segments", [])
    if not segments:
        return "", ""
    return segments_to_texts(segments)


def find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def open_csv(path: Path) -> tuple:
    """Open CSV in append mode; write header if the file is new. Returns (file, writer)."""
    is_new = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    if is_new:
        writer.writeheader()
    return f, writer


def make_row(filename: str, text: str) -> dict:
    return {
        "filename": filename,
        "text": text,
        "is_training_sample": "",
        "call_purpose": "",
        "priority": "",
        "assigned_group": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe MP3 calls with WhisperX and produce two annotation CSVs: "
            "one with plain text, one with timestamps."
        )
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory containing audio files (mp3, wav, etc.)"
    )
    parser.add_argument(
        "--output", "-o", default="dataset.csv",
        help="Base output CSV path (default: dataset.csv). "
             "A second file with '_timed' suffix is always created alongside it."
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"WhisperX model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--language", "-l", default=DEFAULT_LANGUAGE,
        help=f"Language code for transcription (default: {DEFAULT_LANGUAGE})"
    )
    parser.add_argument(
        "--device", "-d", default=None,
        help="Device to use: cpu or cuda (auto-detected by default)"
    )
    parser.add_argument(
        "--compute-type", default="int8",
        help="Compute type for faster-whisper (default: int8, use float16 for GPU)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    plain_path = Path(args.output).resolve()
    timed_path = timed_output_path(plain_path)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    device = args.device or get_device()
    print(f"Using device  : {device}")
    print(f"Model         : {args.model}")
    print(f"Language      : {args.language}")
    print(f"Plain CSV     : {plain_path}")
    print(f"Timed CSV     : {timed_path}")

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"No supported audio files found in '{input_dir}'.")
        sys.exit(0)
    print(f"Found {len(audio_files)} audio file(s).")

    # Use the plain CSV as the reference for already-processed files.
    # Both files are always written together so they stay in sync.
    processed = load_existing_processed(plain_path)
    to_process = [f for f in audio_files if f.name not in processed]

    if not to_process:
        print("All files already processed. Nothing to do.")
        sys.exit(0)

    print(f"Already processed: {len(processed)} | To process: {len(to_process)}\n")

    print("Loading WhisperX model (this may take a moment)...")
    model = whisperx.load_model(
        args.model,
        device=device,
        compute_type=args.compute_type,
        language=args.language,
    )

    plain_file, plain_writer = open_csv(plain_path)
    timed_file, timed_writer = open_csv(timed_path)

    try:
        for idx, audio_path in enumerate(to_process, start=1):
            print(f"[{idx}/{len(to_process)}] {audio_path.name} ...", end=" ", flush=True)
            try:
                plain_text, timed_text = transcribe_file(audio_path, model, args.language)
                plain_writer.writerow(make_row(audio_path.name, plain_text))
                timed_writer.writerow(make_row(audio_path.name, timed_text))
                print("done")
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                err_text = f"[TRANSCRIPTION_ERROR: {exc}]"
                plain_writer.writerow(make_row(audio_path.name, err_text))
                timed_writer.writerow(make_row(audio_path.name, err_text))

            # Flush both files after each record so progress is not lost on crash
            plain_file.flush()
            timed_file.flush()
    finally:
        plain_file.close()
        timed_file.close()

    print(f"\nDone.")
    print(f"  Plain text : {plain_path}")
    print(f"  Timed text : {timed_path}")


if __name__ == "__main__":
    main()
