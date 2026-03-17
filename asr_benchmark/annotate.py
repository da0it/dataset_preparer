#!/usr/bin/env python3
"""
annotate.py — Инструмент разметки для создания ground truth ASR-бенчмарка.

Случайно семплирует короткие сегменты из MP3-записей звонков,
воспроизводит каждый и принимает транскрипцию от пользователя.
Результат сохраняется в ground_truth.json (инкрементально, crash-safe).

Usage:
    python annotate.py --input /path/to/mp3s
    python annotate.py --input /path/to/mp3s --count 50 --duration 12
    python annotate.py --input /path/to/mp3s --output my_gt.json

Управление во время разметки:
    <текст> + Enter   — сохранить транскрипцию
    r + Enter         — переслушать текущий клип
    s + Enter         — пропустить
    q + Enter         — выйти и сохранить прогресс
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub не установлен. Запустите: pip install pydub")
    print("Также нужен ffmpeg: brew install ffmpeg")
    sys.exit(1)


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ground_truth(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def get_audio_duration_ms(audio_path: Path) -> int | None:
    try:
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio)
    except Exception as e:
        print(f"[warn] Не удалось прочитать {audio_path.name}: {e}")
        return None


def extract_segment_to_wav(audio_path: Path, start_ms: int, duration_ms: int) -> Path:
    """Извлекает сегмент и сохраняет как 16kHz mono WAV во временный файл."""
    audio = AudioSegment.from_file(str(audio_path))
    segment = audio[start_ms:start_ms + duration_ms]
    segment = segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    segment.export(tmp.name, format="wav")
    return Path(tmp.name)


def play_audio(wav_path: Path) -> None:
    """Воспроизводит WAV (macOS: afplay; Linux: paplay/aplay)."""
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(wav_path)], check=False)
    else:
        for player in ["paplay", "aplay"]:
            if subprocess.run(["which", player], capture_output=True).returncode == 0:
                subprocess.run([player, str(wav_path)], check=False)
                return
        print("[warn] Аудиоплеер не найден. Установите paplay или aplay.")


# ---------------------------------------------------------------------------
# Segment sampling
# ---------------------------------------------------------------------------

def sample_segments(
    audio_files: list[Path],
    existing_ids: set,
    target_count: int,
    duration_ms: int,
    seed: int,
) -> list[dict]:
    """
    Сэмплирует сегменты равномерно по всем файлам.
    Возвращает список кандидатов размером target_count * 3 (запас).
    """
    rng = random.Random(seed)
    per_file: dict[str, list[dict]] = {}

    for audio_path in audio_files:
        file_duration_ms = get_audio_duration_ms(audio_path)
        if file_duration_ms is None:
            continue
        if file_duration_ms < duration_ms:
            print(f"[warn] {audio_path.name} слишком короткий ({file_duration_ms / 1000:.1f}s), пропускаю")
            continue

        max_start = file_duration_ms - duration_ms
        # Сколько неперекрывающихся сегментов влезает в файл
        n_slots = max(1, max_start // duration_ms)
        candidates = []
        seen_starts = set()
        attempts = 0
        while len(candidates) < n_slots * 2 and attempts < n_slots * 10:
            start_ms = rng.randint(0, max_start)
            # Избегаем слишком близких сегментов
            if any(abs(start_ms - s) < duration_ms // 2 for s in seen_starts):
                attempts += 1
                continue
            seg_id = f"{audio_path.name}@{start_ms}"
            if seg_id not in existing_ids:
                candidates.append({
                    "id": seg_id,
                    "file": audio_path.name,
                    "start_ms": start_ms,
                    "duration_ms": duration_ms,
                })
                seen_starts.add(start_ms)
            attempts += 1

        if candidates:
            per_file[audio_path.name] = candidates

    if not per_file:
        return []

    # Равномерно распределяем по файлам
    all_candidates: list[dict] = []
    file_names = list(per_file.keys())
    budget = target_count * 3

    i = 0
    while len(all_candidates) < budget:
        any_added = False
        for fn in file_names:
            if not per_file[fn]:
                continue
            all_candidates.append(per_file[fn].pop(0))
            any_added = True
            if len(all_candidates) >= budget:
                break
        if not any_added:
            break

    rng.shuffle(all_candidates)
    return all_candidates[:budget]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Разметка аудио-сегментов для ASR-бенчмарка"
    )
    parser.add_argument("--input", "-i", required=True, help="Директория с аудиофайлами")
    parser.add_argument("--output", "-o", default="ground_truth.json", help="Выходной JSON-файл")
    parser.add_argument("--count", "-n", type=int, default=50, help="Целевое количество сегментов (по умолчанию: 50)")
    parser.add_argument("--duration", "-d", type=int, default=12, help="Длина сегмента в секундах (по умолчанию: 12)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed для воспроизводимости")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    duration_ms = args.duration * 1000

    if not input_dir.exists():
        print(f"Error: '{input_dir}' не существует")
        sys.exit(1)

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"Аудиофайлы не найдены в '{input_dir}'")
        sys.exit(1)

    print(f"Найдено аудиофайлов: {len(audio_files)}")

    ground_truth = load_ground_truth(output_path)
    existing_ids = set(ground_truth.keys())
    already_done = sum(1 for v in ground_truth.values() if v.get("text") is not None)

    print(f"Уже размечено:      {already_done} сегментов")

    remaining = args.count - already_done
    if remaining <= 0:
        print(f"Цель в {args.count} сегментов уже достигнута!")
        print(f"Ground truth: {output_path}")
        return

    print(f"Осталось разметить: {remaining} сегментов")
    print("Сэмплирую кандидатов...\n")

    segments = sample_segments(audio_files, existing_ids, remaining, duration_ms, args.seed)
    if not segments:
        print("Нет доступных сегментов для разметки.")
        return

    print("Управление: введите транскрипцию + Enter | 'r' = переслушать | 's' = пропустить | 'q' = выйти\n")
    print("-" * 60)

    annotated = 0
    tmp_wav: Path | None = None

    try:
        for seg in segments:
            if annotated >= remaining:
                break

            audio_path = input_dir / seg["file"]
            if not audio_path.exists():
                print(f"[warn] Файл не найден: {seg['file']}")
                continue

            start_sec = seg["start_ms"] / 1000
            end_sec = (seg["start_ms"] + seg["duration_ms"]) / 1000

            print(
                f"[{already_done + annotated + 1}/{args.count}]  "
                f"{seg['file']}  [{start_sec:.1f}s – {end_sec:.1f}s]"
            )

            # Извлекаем сегмент
            if tmp_wav and tmp_wav.exists():
                tmp_wav.unlink()
            try:
                tmp_wav = extract_segment_to_wav(audio_path, seg["start_ms"], seg["duration_ms"])
            except Exception as e:
                print(f"  [error] Не удалось извлечь сегмент: {e}\n")
                continue

            play_audio(tmp_wav)

            while True:
                try:
                    user_input = input(">>> ").strip()
                except (KeyboardInterrupt, EOFError):
                    user_input = "q"

                if user_input == "q":
                    print("\nВыхожу и сохраняю прогресс...")
                    save_ground_truth(output_path, ground_truth)
                    total = sum(1 for v in ground_truth.values() if v.get("text") is not None)
                    print(f"Сохранено {total} аннотаций → {output_path}")
                    return
                elif user_input in ("r", ""):
                    play_audio(tmp_wav)
                    continue
                elif user_input == "s":
                    print("Пропущен.\n")
                    break
                else:
                    ground_truth[seg["id"]] = {
                        "text": user_input,
                        "file": seg["file"],
                        "start_ms": seg["start_ms"],
                        "duration_ms": seg["duration_ms"],
                    }
                    annotated += 1
                    save_ground_truth(output_path, ground_truth)
                    print()
                    break

    finally:
        if tmp_wav and tmp_wav.exists():
            tmp_wav.unlink()

    total_done = sum(1 for v in ground_truth.values() if v.get("text") is not None)
    print(f"\nГотово! Размечено: {total_done}/{args.count} сегментов")
    print(f"Ground truth сохранён: {output_path}")


if __name__ == "__main__":
    main()
