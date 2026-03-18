#!/usr/bin/env python3
"""
annotate_diarization.py — CLI-инструмент для создания RTTM ground truth.

Для каждого аудиофайла:
  1. Воспроизводит отрезок (~30 сек)
  2. Ты вводишь смену спикеров в простом формате
  3. Файл сохраняется как <filename>.rttm

Формат ввода (интервалы):
    0-5.2:A 5.5-12:B 12.5-18:A 18.2-25:B

Управление:
    <интервалы> + Enter  — сохранить и перейти к следующему файлу
    r + Enter            — переслушать
    n + Enter            — следующие 30 секунд того же файла
    p + Enter            — предыдущие 30 секунд
    d + Enter            — показать текущую разметку файла
    q + Enter            — сохранить и выйти

Usage:
    python annotate_diarization.py --input /path/to/mp3s --output /path/to/rttm
    python annotate_diarization.py --input /path/to/mp3s --output ./gt_rttm --chunk 30
    python annotate_diarization.py --input /path/to/mp3s --file call_001.mp3
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub не установлен. pip install pydub  +  brew install ffmpeg")
    sys.exit(1)

SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
SPEAKER_MAP = {}  # "A" -> "SPEAKER_00", "B" -> "SPEAKER_01", ...


# ---------------------------------------------------------------------------
# Утилиты: аудио
# ---------------------------------------------------------------------------

def load_audio(path: Path) -> AudioSegment:
    return AudioSegment.from_file(str(path)).set_frame_rate(16000).set_channels(1).set_sample_width(2)


def play_segment(audio: AudioSegment, start_ms: int, duration_ms: int) -> None:
    end_ms = min(start_ms + duration_ms, len(audio))
    chunk = audio[start_ms:end_ms]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    chunk.export(tmp_path, format="wav")
    if sys.platform == "darwin":
        subprocess.run(["afplay", tmp_path], check=False)
    else:
        for player in ["paplay", "aplay"]:
            if subprocess.run(["which", player], capture_output=True).returncode == 0:
                subprocess.run([player, tmp_path], check=False)
                return
        print("[warn] Аудиоплеер не найден. Установи paplay или aplay.")
    Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Утилиты: разбор ввода
# ---------------------------------------------------------------------------

def label_to_speaker(label: str) -> str:
    """'A' → 'SPEAKER_00', 'B' → 'SPEAKER_01', etc."""
    label = label.upper()
    if label not in SPEAKER_MAP:
        idx = len(SPEAKER_MAP)
        SPEAKER_MAP[label] = f"SPEAKER_{idx:02d}"
    return SPEAKER_MAP[label]


def parse_intervals(text: str, chunk_offset_sec: float) -> list[tuple] | None:
    """
    Разбирает строку вида "0-5.2:A 5.5-12:B" в список (start, dur, speaker).
    chunk_offset_sec добавляется к каждому временному значению.

    Возвращает None при ошибке разбора.
    """
    text = text.strip()
    if not text:
        return None

    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*[:=]\s*([A-Za-z]\w*)"
    )
    matches = pattern.findall(text)
    if not matches:
        return None

    segments = []
    for start_s, end_s, label in matches:
        start = float(start_s) + chunk_offset_sec
        end = float(end_s) + chunk_offset_sec
        if end <= start:
            print(f"  [warn] Пропущен интервал {start_s}-{end_s} (end ≤ start)")
            continue
        dur = end - start
        speaker = label_to_speaker(label)
        segments.append((start, dur, speaker))

    return segments if segments else None


# ---------------------------------------------------------------------------
# Утилиты: RTTM
# ---------------------------------------------------------------------------

def load_rttm(rttm_path: Path) -> list[tuple]:
    """Загружает существующий RTTM. Возвращает [(start, dur, speaker)]."""
    segments = []
    if not rttm_path.exists():
        return segments
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append((start, dur, speaker))
    return sorted(segments, key=lambda x: x[0])


def save_rttm(segments: list[tuple], file_id: str, rttm_path: Path) -> None:
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    segments_sorted = sorted(segments, key=lambda x: x[0])
    with open(rttm_path, "w") as f:
        for start, dur, speaker in segments_sorted:
            f.write(
                f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )


def print_rttm(segments: list[tuple]) -> None:
    if not segments:
        print("  (нет размеченных сегментов)")
        return
    for start, dur, speaker in sorted(segments, key=lambda x: x[0]):
        bar = "█" * min(int(dur * 2), 40)
        print(f"  {start:7.2f}s – {start + dur:7.2f}s  {speaker:12s}  {bar}")


# ---------------------------------------------------------------------------
# Разметка одного файла
# ---------------------------------------------------------------------------

def annotate_file(
    audio_path: Path,
    rttm_path: Path,
    chunk_sec: int,
) -> bool:
    """
    Интерактивная разметка одного аудиофайла.
    Возвращает True если файл размечен, False если пропущен/выход.
    """
    file_id = audio_path.stem

    print(f"\n{'─' * 60}")
    print(f"Файл: {audio_path.name}")

    try:
        audio = load_audio(audio_path)
    except Exception as e:
        print(f"  [error] Не удалось загрузить: {e}")
        return False

    total_sec = len(audio) / 1000
    print(f"Длина: {total_sec:.1f} сек  ({total_sec / 60:.1f} мин)")
    print(
        f"Чанки по {chunk_sec} сек — всего {int(total_sec // chunk_sec) + 1} чанков\n"
        f"Формат ввода: 0-5.2:A 5.5-12:B 12.5-{chunk_sec}:A"
    )

    # Загружаем существующую разметку (если была прервана)
    segments: list[tuple] = load_rttm(rttm_path)
    if segments:
        print(f"  Найдена существующая разметка: {len(segments)} сегментов")

    chunk_ms = chunk_sec * 1000
    chunk_idx = 0
    total_chunks = max(1, -(-len(audio) // chunk_ms))  # ceiling div

    while chunk_idx < total_chunks:
        start_ms = chunk_idx * chunk_ms
        start_sec = start_ms / 1000
        end_sec = min((chunk_idx + 1) * chunk_sec, total_sec)

        print(f"\n[Чанк {chunk_idx + 1}/{total_chunks}]  "
              f"{start_sec:.1f}s – {end_sec:.1f}s  →  воспроизведение...")
        play_segment(audio, start_ms, chunk_ms)

        while True:
            try:
                raw = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                raw = "q"

            if raw == "q":
                if segments:
                    save_rttm(segments, file_id, rttm_path)
                    print(f"\n  Сохранено {len(segments)} сегментов → {rttm_path}")
                return False  # сигнал выхода из внешнего цикла

            elif raw == "r" or raw == "":
                print("  Переслушиваю...")
                play_segment(audio, start_ms, chunk_ms)
                continue

            elif raw == "n":
                # Следующий чанк без разметки текущего
                print("  Пропускаю чанк (разметка не добавлена)")
                chunk_idx += 1
                break

            elif raw == "p":
                # Назад
                if chunk_idx > 0:
                    chunk_idx -= 1
                    print("  Возврат к предыдущему чанку")
                else:
                    print("  Уже первый чанк")
                break

            elif raw == "d":
                print("\n  Текущая разметка файла:")
                print_rttm(segments)
                continue

            elif raw == "s":
                print("  Файл пропущен.")
                return True  # продолжаем к следующему файлу без сохранения

            else:
                parsed = parse_intervals(raw, chunk_offset_sec=start_sec)
                if parsed is None:
                    print(
                        "  [!] Не удалось разобрать. Формат: 0-5.2:A 5.5-12:B\n"
                        "      Попробуй ещё раз или введи 'r' чтобы переслушать."
                    )
                    continue

                # Удаляем старые сегменты из этого чанка и добавляем новые
                segments = [
                    seg for seg in segments
                    if not (start_sec <= seg[0] < end_sec)
                ]
                segments.extend(parsed)
                save_rttm(segments, file_id, rttm_path)  # crash-safe сохранение

                print(f"  ✓ {len(parsed)} сегментов добавлено | всего: {len(segments)}")
                chunk_idx += 1
                break

    # Весь файл размечен
    print(f"\n  Разметка файла завершена ({len(segments)} сегментов)")
    print_rttm(segments)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Разметка RTTM ground truth для диаризации",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Директория с аудиофайлами")
    parser.add_argument("--output", "-o", default="gt_rttm",
                        help="Директория для сохранения .rttm файлов (default: gt_rttm/)")
    parser.add_argument("--chunk", "-c", type=int, default=30,
                        help="Длина чанка для воспроизведения в секундах (default: 30)")
    parser.add_argument("--file", "-f", default=None,
                        help="Обработать только один конкретный файл")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: '{input_dir}' не существует")
        sys.exit(1)

    if args.file:
        audio_files = [input_dir / args.file]
        audio_files = [f for f in audio_files if f.exists()]
    else:
        audio_files = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )

    if not audio_files:
        print(f"Аудиофайлы не найдены в {input_dir}")
        sys.exit(1)

    # Сколько уже размечено
    done = [f for f in audio_files if (output_dir / (f.stem + ".rttm")).exists()]
    todo = [f for f in audio_files if f not in done]

    print(f"Всего файлов:    {len(audio_files)}")
    print(f"Уже размечено:   {len(done)}")
    print(f"Осталось:        {len(todo)}")
    print(
        "\nУправление:\n"
        "  <интервалы> + Enter  — сохранить чанк\n"
        "  r                    — переслушать\n"
        "  n                    — следующий чанк (без разметки)\n"
        "  p                    — предыдущий чанк\n"
        "  d                    — показать текущую разметку\n"
        "  s                    — пропустить файл\n"
        "  q                    — сохранить и выйти"
    )

    for audio_path in todo:
        rttm_path = output_dir / (audio_path.stem + ".rttm")
        should_continue = annotate_file(audio_path, rttm_path, args.chunk)
        if not should_continue:
            print("\nВыход. Прогресс сохранён.")
            break

    # Итог
    done_now = [f for f in audio_files if (output_dir / (f.stem + ".rttm")).exists()]
    print(f"\nРазмечено: {len(done_now)}/{len(audio_files)} файлов")
    print(f"RTTM сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
