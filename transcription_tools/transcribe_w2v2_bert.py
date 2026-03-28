#!/usr/bin/env python3
"""Transcribe audio files with a Hugging Face Wav2Vec2-BERT CTC model."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import timedelta
from pathlib import Path

CSV_COLUMNS = ["filename", "text", "is_training_sample", "call_purpose", "priority", "assigned_group"]
DEFAULT_MODEL = "rdzotz/w2v2_bert_ru"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_existing_processed(csv_path: Path) -> set[str]:
    processed: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["filename"])
    return processed


def open_csv(path: Path) -> tuple:
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


def find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_audio(path: Path, target_sr: int = 16000):
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for Wav2Vec2-BERT transcription. "
            "Install it with: pip install librosa soundfile"
        ) from exc
    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio


def load_model(model_name: str, device: str, attn_impl: str | None):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    kwargs = {}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    print(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"Loading model: {model_name}")
    model = AutoModelForCTC.from_pretrained(model_name, **kwargs).to(device)
    model.eval()

    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return {"processor": processor, "model": model, "device": device, "dtype": dtype}


def transcribe_audio(model_bundle: dict, audio_path: Path) -> str:
    import torch

    processor = model_bundle["processor"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    audio = load_audio(audio_path, target_sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values=input_values, attention_mask=attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0]
    return " ".join(str(text).strip().split())


def unload_model(model_bundle: dict) -> None:
    try:
        import torch
        del model_bundle["model"]
        del model_bundle["processor"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with a Hugging Face Wav2Vec2-BERT CTC model."
    )
    parser.add_argument("--input", "-i", required=True, help="Directory with audio files.")
    parser.add_argument("--output", "-o", default="dataset_w2v2_bert.csv", help="Output CSV path.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", "-d", default=None, help="Device to use: cpu or cuda (auto-detected by default)")
    parser.add_argument("--attn-implementation", default=None,
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Optional Transformers attention backend.")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory '{input_dir}' does not exist.", file=sys.stderr)
        return 1

    device = get_device(args.device)
    print(f"Using device : {device}")
    print(f"Model        : {args.model}")
    print(f"Output CSV   : {output_path}")

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"No supported audio files found in '{input_dir}'.")
        return 0
    print(f"Found {len(audio_files)} audio file(s).")

    processed = load_existing_processed(output_path)
    to_process = [f for f in audio_files if f.name not in processed]
    if not to_process:
        print("All files already processed. Nothing to do.")
        return 0

    print(f"Already processed: {len(processed)} | To process: {len(to_process)}\n")

    model_bundle = load_model(args.model, device, args.attn_implementation)
    out_file, writer = open_csv(output_path)
    durations: list[float] = []

    try:
        total = len(to_process)
        for idx, audio_path in enumerate(to_process, start=1):
            print(f"[{idx}/{total}] {audio_path.name} ...", end=" ", flush=True)
            t0 = time.monotonic()
            try:
                text = transcribe_audio(model_bundle, audio_path)
                writer.writerow(make_row(audio_path.name, text))
                status = "done"
            except Exception as exc:
                print(f"\n  ERROR: {exc}", file=sys.stderr)
                writer.writerow(make_row(audio_path.name, f"[TRANSCRIPTION_ERROR: {exc}]"))
                status = "error"

            elapsed = time.monotonic() - t0
            durations.append(elapsed)
            avg = sum(durations) / len(durations)
            eta_str = str(timedelta(seconds=int(avg * (total - idx))))
            print(f"{status}  ({elapsed:.1f}s)  ETA: {eta_str}")
            out_file.flush()
    finally:
        out_file.close()
        unload_model(model_bundle)

    total_time = str(timedelta(seconds=int(sum(durations))))
    print(f"\nDone in {total_time}.")
    print(f"  Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
