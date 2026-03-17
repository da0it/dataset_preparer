#!/usr/bin/env python3
"""
benchmark.py — Сравнение ASR-систем на русской телефонной речи.

Модели:  faster-whisper | whisperx | wav2vec2 | vosk
Метрики: WER, CER, RTF (Real-Time Factor)

Usage:
    # Запустить все модели:
    python benchmark.py --ground-truth ground_truth.json --vosk-model /path/to/vosk-model-ru

    # Только определённые модели:
    python benchmark.py --ground-truth ground_truth.json --models faster-whisper wav2vec2

    # Кастомный размер Whisper и модель wav2vec2:
    python benchmark.py --ground-truth ground_truth.json \\
        --whisper-model large-v3 \\
        --wav2vec2-model bond005/wav2vec2-large-ru-golos \\
        --vosk-model /path/to/vosk-model-ru

Vosk Russian model (скачать отдельно):
    https://alphacephei.com/vosk/models
    Рекомендуется: vosk-model-ru-0.42 (большая) или vosk-model-small-ru-0.22 (маленькая)
    wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip && unzip vosk-model-ru-0.42.zip
"""

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import torch

try:
    import tqdm
except ImportError:
    print("Error: tqdm не установлен. Запустите: pip install tqdm")
    sys.exit(1)

try:
    import jiwer
except ImportError:
    print("Error: jiwer не установлен. Запустите: pip install jiwer")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas не установлен. Запустите: pip install pandas")
    sys.exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub не установлен. Запустите: pip install pydub")
    sys.exit(1)

ALL_MODELS = ["faster-whisper", "whisperx", "wav2vec2", "vosk"]


# ---------------------------------------------------------------------------
# Нормализация текста для русского языка
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Приводит к нижнему регистру, убирает пунктуацию, нормализует пробелы.
    ё → е (стандартная нормализация для RU ASR).
    """
    text = text.lower().strip()
    text = text.replace("ё", "е")
    # Оставляем только кириллицу, латиницу, цифры, пробелы
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Подготовка WAV-кэша (16kHz mono PCM)
# ---------------------------------------------------------------------------

def prepare_wav_cache(ground_truth: dict, input_dir: Path, cache_dir: Path) -> dict[str, Path]:
    """
    Конвертирует все аудио-сегменты из ground_truth в 16kHz WAV.
    Возвращает {seg_id: wav_path}.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    seg_to_wav: dict[str, Path] = {}

    print(f"Подготовка WAV-кэша в {cache_dir} ...")
    items = list(ground_truth.items())

    for seg_id, meta in tqdm.tqdm(items, desc="Конвертация аудио"):
        safe_id = re.sub(r"[^\w\-]", "_", seg_id)
        wav_path = cache_dir / f"{safe_id}.wav"

        if not wav_path.exists():
            audio_path = input_dir / meta["file"]
            if not audio_path.exists():
                print(f"  [warn] Файл не найден: {meta['file']}, пропускаю сегмент")
                continue
            try:
                audio = AudioSegment.from_file(str(audio_path))
                segment = audio[meta["start_ms"]:meta["start_ms"] + meta["duration_ms"]]
                segment = segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                segment.export(str(wav_path), format="wav")
            except Exception as e:
                print(f"  [error] {seg_id}: {e}")
                continue

        seg_to_wav[seg_id] = wav_path

    print(f"Готово: {len(seg_to_wav)} WAV-файлов\n")
    return seg_to_wav


def get_wav_duration(wav_path: Path) -> float:
    """Возвращает длительность WAV в секундах."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Модель: faster-whisper
# ---------------------------------------------------------------------------

def load_faster_whisper(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel
    print(f"  Загрузка faster-whisper ({model_size}, {device}, {compute_type})...")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_faster_whisper(model, wav_path: Path) -> str:
    segments, _ = model.transcribe(str(wav_path), language="ru", beam_size=5)
    return " ".join(s.text for s in segments).strip()


def unload_faster_whisper(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Модель: whisperX
# ---------------------------------------------------------------------------

def load_whisperx(model_size: str, device: str, compute_type: str):
    import whisperx
    print(f"  Загрузка whisperX ({model_size}, {device}, {compute_type})...")
    return whisperx.load_model(
        model_size, device=device, compute_type=compute_type, language="ru"
    )


def transcribe_whisperx(model, wav_path: Path) -> str:
    result = model.transcribe(str(wav_path), language="ru", batch_size=8)
    return " ".join(s["text"] for s in result.get("segments", [])).strip()


def unload_whisperx(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Модель: wav2vec2
# ---------------------------------------------------------------------------

def load_wav2vec2(model_name: str, device: str):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    print(f"  Загрузка wav2vec2 ({model_name})...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def transcribe_wav2vec2(model_dict: dict, wav_path: Path) -> str:
    import librosa
    processor = model_dict["processor"]
    model = model_dict["model"]
    device = model_dict["device"]

    audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0].lower().strip()


def unload_wav2vec2(model_dict: dict):
    del model_dict["model"]
    del model_dict["processor"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Модель: vosk
# ---------------------------------------------------------------------------

def load_vosk(model_path: str):
    from vosk import Model, SetLogLevel
    SetLogLevel(-1)  # Отключаем лишние логи
    print(f"  Загрузка vosk ({model_path})...")
    return Model(model_path)


def transcribe_vosk(vosk_model, wav_path: Path) -> str:
    from vosk import KaldiRecognizer
    wf = wave.open(str(wav_path), "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    results = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            r = json.loads(rec.Result())
            if r.get("text"):
                results.append(r["text"])
    r = json.loads(rec.FinalResult())
    if r.get("text"):
        results.append(r["text"])
    wf.close()
    return " ".join(results).strip()


def unload_vosk(model):
    del model


# ---------------------------------------------------------------------------
# Основной бенчмарк
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    seg_to_wav: dict[str, Path],
    ground_truth: dict,
    args,
    device: str,
) -> dict:
    """Запускает одну модель на всех сегментах. Возвращает {seg_id: {hyp, time}}."""
    print(f"\n{'=' * 50}")
    print(f"Модель: {model_name}")
    print(f"{'=' * 50}")

    # Загрузка модели
    if model_name == "faster-whisper":
        model = load_faster_whisper(args.whisper_model, device, args.compute_type)
        transcribe_fn = lambda wav: transcribe_faster_whisper(model, wav)
        unload_fn = lambda: unload_faster_whisper(model)

    elif model_name == "whisperx":
        model = load_whisperx(args.whisper_model, device, args.compute_type)
        transcribe_fn = lambda wav: transcribe_whisperx(model, wav)
        unload_fn = lambda: unload_whisperx(model)

    elif model_name == "wav2vec2":
        model_dict = load_wav2vec2(args.wav2vec2_model, device)
        transcribe_fn = lambda wav: transcribe_wav2vec2(model_dict, wav)
        unload_fn = lambda: unload_wav2vec2(model_dict)

    elif model_name == "vosk":
        if not args.vosk_model:
            print("  [skip] --vosk-model не указан, пропускаю vosk")
            return {}
        vosk_model = load_vosk(args.vosk_model)
        transcribe_fn = lambda wav: transcribe_vosk(vosk_model, wav)
        unload_fn = lambda: unload_vosk(vosk_model)

    else:
        print(f"  [skip] Неизвестная модель: {model_name}")
        return {}

    print(f"  Модель загружена. Транскрибирую {len(seg_to_wav)} сегментов...\n")

    results = {}
    for seg_id, wav_path in tqdm.tqdm(seg_to_wav.items(), desc=model_name):
        try:
            t0 = time.monotonic()
            hypothesis = transcribe_fn(wav_path)
            elapsed = time.monotonic() - t0
            results[seg_id] = {"hypothesis": hypothesis, "time": elapsed}
        except Exception as e:
            print(f"\n  [error] {seg_id}: {e}")
            results[seg_id] = {"hypothesis": "", "time": 0.0}

    unload_fn()
    return results


# ---------------------------------------------------------------------------
# Вычисление метрик
# ---------------------------------------------------------------------------

def compute_metrics(
    model_name: str,
    model_results: dict,
    ground_truth: dict,
    seg_to_wav: dict[str, Path],
) -> dict:
    """Вычисляет WER, CER, RTF для одной модели."""
    references = []
    hypotheses = []
    total_audio_sec = 0.0
    total_inference_sec = 0.0

    for seg_id, pred in model_results.items():
        if seg_id not in ground_truth:
            continue
        ref = normalize(ground_truth[seg_id]["text"])
        hyp = normalize(pred["hypothesis"])
        # Пропускаем пустые reference (ошибки разметки)
        if not ref:
            continue
        references.append(ref)
        hypotheses.append(hyp)
        total_inference_sec += pred["time"]
        if seg_id in seg_to_wav:
            total_audio_sec += get_wav_duration(seg_to_wav[seg_id])

    if not references:
        return {"model": model_name, "WER": None, "CER": None, "RTF": None, "n_samples": 0}

    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    rtf = total_inference_sec / total_audio_sec if total_audio_sec > 0 else None

    return {
        "model": model_name,
        "WER": round(wer * 100, 2),
        "CER": round(cer * 100, 2),
        "RTF": round(rtf, 4) if rtf is not None else None,
        "n_samples": len(references),
        "total_audio_sec": round(total_audio_sec, 1),
        "total_inference_sec": round(total_inference_sec, 1),
    }


# ---------------------------------------------------------------------------
# Детальный отчёт по ошибкам
# ---------------------------------------------------------------------------

def print_error_examples(
    model_results: dict,
    ground_truth: dict,
    model_name: str,
    n: int = 5,
):
    """Печатает N примеров с наибольшим WER."""
    examples = []
    for seg_id, pred in model_results.items():
        if seg_id not in ground_truth:
            continue
        ref = normalize(ground_truth[seg_id]["text"])
        hyp = normalize(pred["hypothesis"])
        if not ref:
            continue
        try:
            w = jiwer.wer([ref], [hyp])
        except Exception:
            w = 1.0
        examples.append((w, ref, hyp, seg_id))

    examples.sort(reverse=True)
    print(f"\nТоп-{n} ошибок ({model_name}):")
    for w, ref, hyp, seg_id in examples[:n]:
        print(f"  [{w * 100:.0f}% WER] {seg_id}")
        print(f"    REF: {ref}")
        print(f"    HYP: {hyp}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Бенчмарк ASR-систем на русском языке",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ground-truth", "-g", required=True,
                        help="Путь к ground_truth.json (из annotate.py)")
    parser.add_argument("--input", "-i", required=True,
                        help="Директория с исходными аудиофайлами")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, metavar="MODEL",
                        help=f"Модели для тестирования (по умолчанию: все). "
                             f"Варианты: {', '.join(ALL_MODELS)}")
    parser.add_argument("--whisper-model", default="large-v3",
                        help="Размер модели Whisper (default: large-v3)")
    parser.add_argument("--compute-type", default="float16",
                        help="Тип вычислений для Whisper (default: float16 для GPU)")
    parser.add_argument("--wav2vec2-model", default="bond005/wav2vec2-large-ru-golos",
                        help="HuggingFace модель wav2vec2 (default: bond005/wav2vec2-large-ru-golos)")
    parser.add_argument("--vosk-model", default=None,
                        help="Путь к директории Vosk модели (обязателен для vosk)")
    parser.add_argument("--cache-dir", default="wav_cache",
                        help="Директория для WAV-кэша (default: wav_cache/)")
    parser.add_argument("--output", "-o", default="benchmark_results.csv",
                        help="Выходной CSV с результатами (default: benchmark_results.csv)")
    parser.add_argument("--error-examples", type=int, default=3,
                        help="Показывать N примеров ошибок на модель (0 = выкл)")
    parser.add_argument("--device", default=None,
                        help="Устройство: cuda / cpu (автоопределение по умолчанию)")
    args = parser.parse_args()

    # --- Проверки ---
    gt_path = Path(args.ground_truth)
    input_dir = Path(args.input).resolve()
    output_path = Path(args.output)

    if not gt_path.exists():
        print(f"Error: ground_truth файл не найден: {gt_path}")
        sys.exit(1)
    if not input_dir.exists():
        print(f"Error: директория с аудио не найдена: {input_dir}")
        sys.exit(1)

    if "vosk" in args.models and not args.vosk_model:
        print("Предупреждение: vosk выбран, но --vosk-model не указан. Vosk будет пропущен.")

    # --- Загрузка ground truth ---
    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    annotated = {k: v for k, v in ground_truth.items() if v.get("text")}
    print(f"Ground truth: {len(annotated)} размеченных сегментов")

    if not annotated:
        print("Error: нет размеченных сегментов в ground_truth.json")
        sys.exit(1)

    # --- Устройство ---
    device = args.device or get_device()
    print(f"Устройство: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- WAV-кэш ---
    cache_dir = Path(args.cache_dir)
    seg_to_wav = prepare_wav_cache(annotated, input_dir, cache_dir)

    if not seg_to_wav:
        print("Error: ни один WAV-сегмент не был подготовлен")
        sys.exit(1)

    # --- Бенчмарк ---
    all_metrics = []
    all_results: dict[str, dict] = {}

    for model_name in args.models:
        model_results = run_model(model_name, seg_to_wav, annotated, args, device)
        if not model_results:
            continue
        all_results[model_name] = model_results

        metrics = compute_metrics(model_name, model_results, annotated, seg_to_wav)
        all_metrics.append(metrics)

        print(f"\n  WER: {metrics['WER']}%  |  CER: {metrics['CER']}%  |  RTF: {metrics['RTF']}")

        if args.error_examples > 0:
            print_error_examples(model_results, annotated, model_name, args.error_examples)

    # --- Итоговая таблица ---
    if not all_metrics:
        print("\nНет результатов для отображения.")
        return

    df = pd.DataFrame(all_metrics).set_index("model")

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 60)
    print(df[["WER", "CER", "RTF", "n_samples"]].to_string())
    print()

    # Победители
    valid = df[df["WER"].notna()]
    if not valid.empty:
        best_wer = valid["WER"].idxmin()
        best_rtf = valid["RTF"].idxmin() if valid["RTF"].notna().any() else None
        print(f"Лучший WER : {best_wer} ({valid.loc[best_wer, 'WER']}%)")
        print(f"Лучший CER : {valid['CER'].idxmin()} ({valid['CER'].min()}%)")
        if best_rtf:
            print(f"Быстрейший : {best_rtf} (RTF={valid.loc[best_rtf, 'RTF']})")

    # --- Сохранение ---
    df.reset_index().to_csv(output_path, index=False)
    print(f"\nРезультаты сохранены: {output_path}")

    # Детальные результаты по каждому сегменту
    detailed_rows = []
    for model_name, model_results in all_results.items():
        for seg_id, pred in model_results.items():
            if seg_id not in annotated:
                continue
            ref = normalize(annotated[seg_id]["text"])
            hyp = normalize(pred["hypothesis"])
            seg_wer = jiwer.wer([ref], [hyp]) if ref else None
            detailed_rows.append({
                "model": model_name,
                "seg_id": seg_id,
                "file": annotated[seg_id]["file"],
                "start_ms": annotated[seg_id]["start_ms"],
                "reference": ref,
                "hypothesis": hyp,
                "wer": round(seg_wer * 100, 1) if seg_wer is not None else None,
                "inference_sec": round(pred["time"], 3),
            })

    if detailed_rows:
        detail_path = output_path.with_stem(output_path.stem + "_detailed")
        pd.DataFrame(detailed_rows).to_csv(detail_path, index=False)
        print(f"Детальные результаты: {detail_path}")


if __name__ == "__main__":
    main()
