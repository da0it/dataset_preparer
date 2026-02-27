#!/usr/bin/env python3
"""
Transcribe MP3 files using WhisperX and save results to three CSVs for manual annotation.

Always produces three output files simultaneously (one transcription pass per file):
  - dataset.csv            — plain text (all segments joined into one line)
  - dataset_timed.csv      — text with timestamps: [0.0-3.5] segment | [3.5-7.2] segment ...
  - dataset_speakers.csv   — text with timestamps and speaker labels:
                             [0.0-3.5] SPEAKER_00: segment | [3.5-7.2] SPEAKER_01: segment ...

The speaker CSV requires a Hugging Face token (--hf-token or HF_TOKEN env var).
If the token is not provided, only the first two files are produced.

Usage:
    python transcribe.py --input /path/to/mp3s
    python transcribe.py --input /path/to/mp3s --output my_dataset.csv
    python transcribe.py --input /path/to/mp3s --hf-token hf_xxxxxxxx

Columns in all CSVs:
    filename, text, is_training_sample, call_purpose, priority, assigned_group
"""

import argparse
import csv
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import whisperx

CSV_COLUMNS = ["filename", "text", "is_training_sample", "call_purpose", "priority", "assigned_group"]
DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "ru"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Detect available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def timed_output_path(base: Path) -> Path:
    return base.with_stem(base.stem + "_timed")


def speakers_output_path(base: Path) -> Path:
    return base.with_stem(base.stem + "_speakers")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_existing_processed(csv_path: Path) -> set[str]:
    """Return set of filenames already present in the CSV."""
    processed: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["filename"])
    return processed


def open_csv(path: Path) -> tuple:
    """Open CSV in append mode; write header if file is new. Returns (file, writer)."""
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


# ---------------------------------------------------------------------------
# Text formatters
# ---------------------------------------------------------------------------

def segments_to_plain_and_timed(segments: list[dict]) -> tuple[str, str]:
    """
    Returns:
      plain_text — all segment texts joined with spaces
      timed_text — [start-end] text | [start-end] text ...
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


def segments_to_speakers_text(segments: list[dict]) -> str:
    """
    Returns timed text with speaker labels:
      [start-end] SPEAKER_00: text | [start-end] SPEAKER_01: text ...
    Segments without a speaker label fall back to UNKNOWN.
    """
    parts: list[str] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        speaker = seg.get("speaker", "UNKNOWN")
        parts.append(f"[{start:.1f}-{end:.1f}] {speaker}: {text}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Transcription + diarization
# ---------------------------------------------------------------------------

def transcribe_file(
    audio_path: Path,
    model,
    language: str,
    diarize_pipeline,
    align_model,        # pre-loaded once before the loop, or None
    align_metadata,     # pre-loaded once before the loop, or None
    device: str,
) -> tuple[str, str, str | None]:
    """
    Transcribe and optionally diarize a single audio file.

    align_model / align_metadata must be loaded once outside the loop and
    passed in here — loading them per-file causes a memory leak that kills
    the process after several hundred files.

    Returns:
      plain_text    — all segments joined
      timed_text    — segments with timestamps
      speakers_text — segments with timestamps + speaker labels, or None if diarization skipped
    """
    import torch

    # Step 1: transcribe
    result = model.transcribe(str(audio_path), language=language, batch_size=16)
    segments = result.get("segments", [])
    if not segments:
        return "", "", ("" if diarize_pipeline else None)

    plain_text, timed_text = segments_to_plain_and_timed(segments)

    if diarize_pipeline is None:
        return plain_text, timed_text, None

    # Step 2: align — uses the pre-loaded model, no allocation per file
    if align_model is not None:
        try:
            result = whisperx.align(
                segments, align_model, align_metadata, str(audio_path), device,
                return_char_alignments=False,
            )
            segments = result.get("segments", segments)
        except Exception as align_exc:
            print(f"\n  [warn] alignment failed ({align_exc}), using unaligned segments")

    # Step 3: diarize
    try:
        audio_tensor = whisperx.load_audio(str(audio_path))
        diarize_segments = diarize_pipeline(
            {"waveform": torch.from_numpy(audio_tensor).unsqueeze(0), "sample_rate": 16000}
        )
        result = whisperx.diarize.assign_word_speakers(diarize_segments, {"segments": segments})
        segments = result.get("segments", segments)
    except Exception as diar_exc:
        print(f"\n  [warn] diarization failed ({diar_exc}), speaker labels will be UNKNOWN")

    speakers_text = segments_to_speakers_text(segments)
    return plain_text, timed_text, speakers_text


def find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe MP3 calls with WhisperX and produce up to three annotation CSVs: "
            "plain text, timed, and timed+speakers (requires HF token)."
        )
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Directory containing audio files (mp3, wav, etc.)"
    )
    parser.add_argument(
        "--output", "-o", default="dataset.csv",
        help="Base output CSV path (default: dataset.csv)"
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
    parser.add_argument(
        "--hf-token", default=None,
        help="Hugging Face token for pyannote diarization. "
             "Can also be set via HF_TOKEN environment variable."
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    plain_path = Path(args.output).resolve()
    timed_path = timed_output_path(plain_path)
    speakers_path = speakers_output_path(plain_path)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Resolve HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    diarize = hf_token is not None

    device = args.device or get_device()
    print(f"Using device  : {device}")
    print(f"Model         : {args.model}")
    print(f"Language      : {args.language}")
    print(f"Diarization   : {'enabled' if diarize else 'disabled (no HF token)'}")
    print(f"Plain CSV     : {plain_path}")
    print(f"Timed CSV     : {timed_path}")
    if diarize:
        print(f"Speakers CSV  : {speakers_path}")

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"No supported audio files found in '{input_dir}'.")
        sys.exit(0)
    print(f"Found {len(audio_files)} audio file(s).")

    processed = load_existing_processed(plain_path)
    to_process = [f for f in audio_files if f.name not in processed]

    if not to_process:
        print("All files already processed. Nothing to do.")
        sys.exit(0)

    print(f"Already processed: {len(processed)} | To process: {len(to_process)}\n")

    # Load Whisper model
    print("Loading WhisperX model (this may take a moment)...")
    model = whisperx.load_model(
        args.model,
        device=device,
        compute_type=args.compute_type,
        language=args.language,
    )

    # Load diarization pipeline + alignment model (optional, both loaded once)
    diarize_pipeline = None
    align_model = None
    align_metadata = None
    if diarize:
        print("Loading pyannote diarization model...")
        try:
            # pyannote >= 3.x uses `token`, older versions use `use_auth_token`
            try:
                diarize_pipeline = whisperx.diarize.DiarizationPipeline(
                    token=hf_token,
                    device=device,
                )
            except TypeError:
                diarize_pipeline = whisperx.diarize.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device,
                )
        except Exception as exc:
            print(f"Warning: could not load diarization pipeline: {exc}", file=sys.stderr)
            print("Continuing without diarization — only plain and timed CSVs will be produced.")
            diarize_pipeline = None
            diarize = False

        if diarize_pipeline is not None:
            print("Loading alignment model...")
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=args.language, device=device
                )
            except Exception as exc:
                print(f"Warning: could not load alignment model: {exc}", file=sys.stderr)
                print("Diarization will proceed without word-level alignment.")
                align_model = None
                align_metadata = None

    # Open output files
    plain_file, plain_writer = open_csv(plain_path)
    timed_file, timed_writer = open_csv(timed_path)
    speakers_file, speakers_writer = (open_csv(speakers_path) if diarize else (None, None))

    total = len(to_process)
    durations: list[float] = []

    try:
        for idx, audio_path in enumerate(to_process, start=1):
            print(f"[{idx}/{total}] {audio_path.name} ...", end=" ", flush=True)
            t0 = time.monotonic()
            try:
                plain_text, timed_text, speakers_text = transcribe_file(
                    audio_path, model, args.language,
                    diarize_pipeline, align_model, align_metadata, device
                )
                plain_writer.writerow(make_row(audio_path.name, plain_text))
                timed_writer.writerow(make_row(audio_path.name, timed_text))
                if speakers_writer is not None and speakers_text is not None:
                    speakers_writer.writerow(make_row(audio_path.name, speakers_text))
                status = "done"
            except Exception as exc:
                print(f"\n  ERROR: {exc}", file=sys.stderr)
                err_text = f"[TRANSCRIPTION_ERROR: {exc}]"
                plain_writer.writerow(make_row(audio_path.name, err_text))
                timed_writer.writerow(make_row(audio_path.name, err_text))
                if speakers_writer is not None:
                    speakers_writer.writerow(make_row(audio_path.name, err_text))
                status = "error"

            elapsed = time.monotonic() - t0
            durations.append(elapsed)
            avg = sum(durations) / len(durations)
            eta_str = str(timedelta(seconds=int(avg * (total - idx))))
            print(f"{status}  ({elapsed:.1f}s)  ETA: {eta_str}")

            plain_file.flush()
            timed_file.flush()
            if speakers_file:
                speakers_file.flush()

    finally:
        plain_file.close()
        timed_file.close()
        if speakers_file:
            speakers_file.close()

    total_time = str(timedelta(seconds=int(sum(durations))))
    print(f"\nDone in {total_time}.")
    print(f"  Plain text   : {plain_path}")
    print(f"  Timed text   : {timed_path}")
    if diarize:
        print(f"  With speakers: {speakers_path}")


if __name__ == "__main__":
    main()
