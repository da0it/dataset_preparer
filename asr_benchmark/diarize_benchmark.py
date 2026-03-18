#!/usr/bin/env python3
"""
diarize_benchmark.py — Сравнение систем диаризации на телефонных звонках.

Системы:  pyannote 3.1 | NeMo ClusteringDiarizer | SpeechBrain (ECAPA + clustering)
Метрики:  DER, MS, FA, SC (если есть RTTM ground truth), RTF
Выход:    .rttm файлы для каждой системы + CSV с метриками

Usage:
    # Только RTF (без ground truth):
    python diarize_benchmark.py --input /path/to/audio

    # Полный бенчмарк с DER:
    python diarize_benchmark.py --input /path/to/audio --ground-truth /path/to/rttm

    # Указать число спикеров (для звонков обычно 2):
    python diarize_benchmark.py --input /path/to/audio --num-speakers 2

    # Только некоторые системы:
    python diarize_benchmark.py --input /path/to/audio --systems pyannote speechbrain

HuggingFace token (нужен для pyannote):
    export HF_TOKEN=hf_xxxxxxxx
    # Также нужно принять лицензию на https://hf.co/pyannote/speaker-diarization-3.1
"""

import argparse
import json
import os
import sys
import time
import tempfile
import wave
from pathlib import Path

import torch

try:
    from pydub import AudioSegment
except ImportError:
    print("Error: pydub не установлен. pip install pydub  +  brew install ffmpeg")
    sys.exit(1)

try:
    import tqdm
except ImportError:
    print("Error: tqdm не установлен. pip install tqdm")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Error: pandas не установлен. pip install pandas")
    sys.exit(1)

ALL_SYSTEMS = ["pyannote", "nemo", "speechbrain"]


# ---------------------------------------------------------------------------
# Аудио-утилиты
# ---------------------------------------------------------------------------

def to_wav16k(audio_path: Path, cache_dir: Path) -> Path:
    """Конвертирует MP3/любой формат в 16kHz mono WAV. Кэширует результат."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / (audio_path.stem + "_16k.wav")
    if not wav_path.exists():
        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(str(wav_path), format="wav")
    return wav_path


def get_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def prepare_wav_files(
    audio_files: list[Path], cache_dir: Path
) -> dict[str, Path]:
    """Возвращает {stem: wav_path} для всех аудиофайлов."""
    result = {}
    print(f"Конвертация {len(audio_files)} файлов в WAV 16kHz...")
    for p in tqdm.tqdm(audio_files, desc="WAV конвертация"):
        try:
            wav = to_wav16k(p, cache_dir)
            result[p.stem] = wav
        except Exception as e:
            print(f"  [warn] {p.name}: {e}")
    return result


# ---------------------------------------------------------------------------
# RTTM-утилиты
# ---------------------------------------------------------------------------

def save_rttm(segments: list[tuple], file_id: str, rttm_path: Path) -> None:
    """
    Сохраняет список (start, duration, speaker) в .rttm файл.
    segments: [(start_sec, dur_sec, speaker_label), ...]
    """
    with open(rttm_path, "w") as f:
        for start, dur, speaker in segments:
            f.write(
                f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )


def load_rttm(rttm_path: Path) -> "pyannote.core.Annotation":
    """Загружает RTTM как pyannote Annotation."""
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            ann[Segment(start, start + dur)] = speaker
    return ann


def annotation_to_segments(annotation) -> list[tuple]:
    """pyannote Annotation → список (start, dur, speaker)."""
    segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((segment.start, segment.duration, speaker))
    return segments


# ---------------------------------------------------------------------------
# Система 1: pyannote/speaker-diarization-3.1
# ---------------------------------------------------------------------------

def load_pyannote(device: str, hf_token: str | None):
    from pyannote.audio import Pipeline
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Нужен HuggingFace токен для pyannote.\n"
            "  export HF_TOKEN=hf_xxxxxxxx\n"
            "  Также примите лицензию: https://hf.co/pyannote/speaker-diarization-3.1"
        )
    print(f"  Загрузка pyannote/speaker-diarization-3.1 ({device})...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    pipeline.to(torch.device(device))
    return pipeline


def diarize_pyannote(pipeline, wav_path: Path, num_speakers: int | None) -> list[tuple]:
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    diarization = pipeline(str(wav_path), **kwargs)
    return annotation_to_segments(diarization)


def unload_pyannote(pipeline):
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Система 2: NeMo ClusteringDiarizer
# ---------------------------------------------------------------------------

def load_nemo(device: str, num_speakers: int | None, tmp_dir: Path):
    """
    Инициализирует NeMo ClusteringDiarizer с telephony-конфигом.
    tmp_dir используется для manifest и выходных файлов.
    """
    try:
        from omegaconf import OmegaConf
        from nemo.collections.asr.models import ClusteringDiarizer
    except ImportError:
        raise ImportError(
            "NeMo не установлен.\n"
            "  pip install nemo_toolkit['asr']"
        )

    cfg_dict = {
        "diarizer": {
            "manifest_filepath": str(tmp_dir / "manifest.json"),
            "out_dir": str(tmp_dir / "nemo_out"),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": "vad_multilingual_marblenet",
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.875,
                    "onset": 0.4,
                    "offset": 0.7,
                    "pad_onset": 0.05,
                    "pad_offset": -0.1,
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.4,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.0, 0.5],
                    "shift_length_in_sec": [0.75, 0.5, 0.25],
                    "multiscale_weights": [1, 1, 1],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": num_speakers is not None,
                    "num_speakers": num_speakers,
                    "max_num_speakers": 8,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False,
                },
            },
        }
    }

    cfg = OmegaConf.create(cfg_dict)
    out_dir = tmp_dir / "nemo_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ClusteringDiarizer(cfg=cfg)
    model = model.to(device)
    return {"model": model, "cfg": cfg, "tmp_dir": tmp_dir}


def diarize_nemo(nemo_dict: dict, wav_path: Path, file_id: str) -> list[tuple]:
    """Диаризует один файл через NeMo. Возвращает сегменты."""
    import json as json_mod
    from omegaconf import OmegaConf

    tmp_dir = nemo_dict["tmp_dir"]
    out_dir = tmp_dir / "nemo_out"
    model = nemo_dict["model"]
    cfg = nemo_dict["cfg"]

    # Создаём manifest для одного файла
    manifest_path = tmp_dir / "manifest.json"
    entry = {
        "audio_filepath": str(wav_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": cfg.diarizer.clustering.parameters.num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(manifest_path, "w") as f:
        f.write(json_mod.dumps(entry) + "\n")

    OmegaConf.update(cfg, "diarizer.manifest_filepath", str(manifest_path))

    model.diarize()

    # Читаем RTTM из вывода NeMo
    rttm_out = out_dir / "pred_rttms" / f"{wav_path.stem}.rttm"
    if not rttm_out.exists():
        # NeMo иногда пишет в другое место
        rttm_out = out_dir / f"{wav_path.stem}.rttm"

    if not rttm_out.exists():
        raise FileNotFoundError(f"NeMo RTTM не найден: {rttm_out}")

    ann = load_rttm(rttm_out)
    return annotation_to_segments(ann)


def unload_nemo(nemo_dict: dict):
    del nemo_dict["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Система 3: SpeechBrain ECAPA + Agglomerative Clustering
# ---------------------------------------------------------------------------

def load_speechbrain(device: str):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        raise ImportError(
            "SpeechBrain не установлен.\n"
            "  pip install speechbrain"
        )
    print(f"  Загрузка speechbrain/spkrec-ecapa-voxceleb ({device})...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )
    return {"classifier": classifier, "device": device}


def diarize_speechbrain(
    sb_dict: dict,
    wav_path: Path,
    num_speakers: int | None,
    window_sec: float = 1.5,
    step_sec: float = 0.75,
) -> list[tuple]:
    """
    Диаризует аудио через sliding-window ECAPA embeddings + AgglomerativeClustering.
    """
    import librosa
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    classifier = sb_dict["classifier"]
    device = sb_dict["device"]

    audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    duration = len(audio) / sr

    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)

    if len(audio) < window_samples:
        # Файл короче окна — один сегмент
        return [(0.0, duration, "SPEAKER_00")]

    # --- Извлечение эмбеддингов ---
    embeddings = []
    timestamps = []

    for start_s in range(0, len(audio) - window_samples + 1, step_samples):
        chunk = audio[start_s: start_s + window_samples]
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(chunk_tensor)
        embeddings.append(emb.squeeze().cpu().numpy())
        t_start = start_s / sr
        t_end = (start_s + window_samples) / sr
        timestamps.append((t_start, t_end))

    if not embeddings:
        return [(0.0, duration, "SPEAKER_00")]

    embeddings_arr = normalize(np.array(embeddings))

    # --- Кластеризация ---
    n_clusters = num_speakers or _estimate_num_speakers(embeddings_arr)
    n_clusters = max(1, min(n_clusters, len(embeddings_arr)))

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    ).fit_predict(embeddings_arr)

    # --- Схлопываем соседние окна одного спикера ---
    raw = [(t[0], t[1], f"SPEAKER_{labels[i]:02d}") for i, t in enumerate(timestamps)]
    merged = _merge_segments(raw, gap_threshold=step_sec * 1.1)
    return merged


def _estimate_num_speakers(embeddings: "np.ndarray", max_speakers: int = 6) -> int:
    """Оценивает число спикеров через eigenvalue gap в cosine-матрице."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if len(embeddings) < 4:
        return 1

    sim = cosine_similarity(embeddings)
    # Нормализуем матрицу
    sim = (sim + 1) / 2
    np.fill_diagonal(sim, 1.0)

    eigenvalues = np.sort(np.linalg.eigvalsh(sim))[::-1]
    gaps = np.diff(eigenvalues[: max_speakers + 1])
    # Максимальный gap указывает на число спикеров
    n = int(np.argmax(np.abs(gaps))) + 1
    return max(1, min(n, max_speakers))


def _merge_segments(
    segments: list[tuple], gap_threshold: float = 0.5
) -> list[tuple]:
    """Схлопывает соседние сегменты одного спикера с зазором ≤ gap_threshold."""
    if not segments:
        return []
    merged = []
    cur_start, cur_end, cur_spk = segments[0]
    for start, end, spk in segments[1:]:
        if spk == cur_spk and start - cur_end <= gap_threshold:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end - cur_start, cur_spk))
            cur_start, cur_end, cur_spk = start, end, spk
    merged.append((cur_start, cur_end - cur_start, cur_spk))
    return merged


def unload_speechbrain(sb_dict: dict):
    del sb_dict["classifier"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Метрики: DER (через pyannote.metrics)
# ---------------------------------------------------------------------------

def compute_der(reference_ann, hypothesis_ann, collar: float = 0.25) -> dict:
    """
    Вычисляет DER и его компоненты.
    collar: допуск на границу сегмента в секундах (обычно 0.25 для телефонии)
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar, skip_overlap=True)
    components = metric.compute_components(reference_ann, hypothesis_ann)
    der = metric(reference_ann, hypothesis_ann)

    total = components["total"]
    return {
        "DER": round(der * 100, 2),
        "MS":  round(components["missed detection"] / total * 100, 2) if total else 0,
        "FA":  round(components["false alarm"] / total * 100, 2) if total else 0,
        "SC":  round(components["confusion"] / total * 100, 2) if total else 0,
    }


def ann_from_segments(segments: list[tuple]) -> "pyannote.core.Annotation":
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for start, dur, speaker in segments:
        ann[Segment(start, start + dur)] = speaker
    return ann


# ---------------------------------------------------------------------------
# Основной цикл бенчмарка
# ---------------------------------------------------------------------------

def run_system(
    system_name: str,
    wav_files: dict[str, Path],
    args,
    device: str,
    rttm_out_dir: Path,
    tmp_dir: Path,
) -> dict[str, dict]:
    """
    Запускает одну систему диаризации на всех файлах.
    Возвращает {file_id: {"segments": [...], "time": float}}.
    """
    print(f"\n{'=' * 55}")
    print(f"Система: {system_name.upper()}")
    print(f"{'=' * 55}")

    # --- Загрузка модели ---
    try:
        if system_name == "pyannote":
            model = load_pyannote(device, args.hf_token)
            def infer(wav, fid):
                return diarize_pyannote(model, wav, args.num_speakers)
            def unload():
                unload_pyannote(model)

        elif system_name == "nemo":
            model = load_nemo(device, args.num_speakers, tmp_dir / "nemo")
            def infer(wav, fid):
                return diarize_nemo(model, wav, fid)
            def unload():
                unload_nemo(model)

        elif system_name == "speechbrain":
            model = load_speechbrain(device)
            def infer(wav, fid):
                return diarize_speechbrain(model, wav, args.num_speakers)
            def unload():
                unload_speechbrain(model)

        else:
            print(f"  [skip] Неизвестная система: {system_name}")
            return {}

    except Exception as e:
        print(f"  [error] Не удалось загрузить {system_name}: {e}")
        return {}

    print(f"  Модель загружена. Обрабатываю {len(wav_files)} файлов...\n")
    results = {}
    rttm_out_dir_sys = rttm_out_dir / system_name
    rttm_out_dir_sys.mkdir(parents=True, exist_ok=True)

    for file_id, wav_path in tqdm.tqdm(wav_files.items(), desc=system_name):
        try:
            t0 = time.monotonic()
            segments = infer(wav_path, file_id)
            elapsed = time.monotonic() - t0

            rttm_path = rttm_out_dir_sys / f"{file_id}.rttm"
            save_rttm(segments, file_id, rttm_path)

            results[file_id] = {
                "segments": segments,
                "time": elapsed,
                "rttm": rttm_path,
            }
        except Exception as e:
            print(f"\n  [error] {file_id}: {e}")
            results[file_id] = {"segments": [], "time": 0.0, "rttm": None}

    unload()
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Бенчмарк систем диаризации",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Директория с аудиофайлами (mp3/wav/m4a/...)")
    parser.add_argument("--ground-truth", "-g", default=None,
                        help="Директория с .rttm файлами ground truth (опционально)")
    parser.add_argument("--systems", nargs="+", default=ALL_SYSTEMS,
                        choices=ALL_SYSTEMS, metavar="SYS",
                        help=f"Системы для сравнения (default: все). "
                             f"Варианты: {', '.join(ALL_SYSTEMS)}")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Число спикеров (None = автоопределение). Для звонков обычно 2")
    parser.add_argument("--collar", type=float, default=0.25,
                        help="Collar для DER в секундах (default: 0.25)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace токен для pyannote (или задайте HF_TOKEN)")
    parser.add_argument("--output", "-o", default="diarize_results.csv",
                        help="Выходной CSV (default: diarize_results.csv)")
    parser.add_argument("--rttm-dir", default="rttm_output",
                        help="Директория для RTTM-файлов систем (default: rttm_output/)")
    parser.add_argument("--cache-dir", default="wav_cache",
                        help="Директория WAV-кэша (default: wav_cache/)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (автоопределение)")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_path = Path(args.output)
    rttm_out_dir = Path(args.rttm_dir)
    cache_dir = Path(args.cache_dir)

    if not input_dir.exists():
        print(f"Error: директория не найдена: {input_dir}")
        sys.exit(1)

    # --- Аудиофайлы ---
    EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    audio_files = sorted(p for p in input_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in EXTS)
    if not audio_files:
        print(f"Аудиофайлы не найдены в {input_dir}")
        sys.exit(1)
    print(f"Найдено файлов: {len(audio_files)}")

    # --- Конвертация в WAV 16kHz ---
    wav_files = prepare_wav_files(audio_files, cache_dir)
    if not wav_files:
        print("Error: ни один файл не конвертирован")
        sys.exit(1)

    # --- Ground truth RTTM ---
    gt_annotations: dict[str, object] = {}
    if args.ground_truth:
        gt_dir = Path(args.ground_truth)
        for rttm_path in gt_dir.glob("*.rttm"):
            try:
                gt_annotations[rttm_path.stem] = load_rttm(rttm_path)
            except Exception as e:
                print(f"[warn] Не удалось загрузить {rttm_path.name}: {e}")
        print(f"Загружено RTTM ground truth: {len(gt_annotations)} файлов")

    # --- Устройство ---
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Бенчмарк ---
    all_metrics = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="diarize_bench_"))

    for system_name in args.systems:
        sys_results = run_system(
            system_name, wav_files, args, device, rttm_out_dir, tmp_dir
        )
        if not sys_results:
            continue

        # RTF
        total_audio_sec = sum(get_duration(w) for w in wav_files.values())
        total_infer_sec = sum(r["time"] for r in sys_results.values())
        rtf = total_infer_sec / total_audio_sec if total_audio_sec > 0 else None

        # DER (если есть ground truth)
        der_vals = []
        ms_vals, fa_vals, sc_vals = [], [], []

        for file_id, result in sys_results.items():
            if file_id not in gt_annotations or not result["segments"]:
                continue
            hyp_ann = ann_from_segments(result["segments"])
            ref_ann = gt_annotations[file_id]
            try:
                d = compute_der(ref_ann, hyp_ann, collar=args.collar)
                der_vals.append(d["DER"])
                ms_vals.append(d["MS"])
                fa_vals.append(d["FA"])
                sc_vals.append(d["SC"])
            except Exception as e:
                print(f"  [warn] DER не вычислен для {file_id}: {e}")

        def mean_or_none(lst):
            return round(sum(lst) / len(lst), 2) if lst else None

        row = {
            "system": system_name,
            "n_files": len(sys_results),
            "RTF": round(rtf, 4) if rtf else None,
            "DER": mean_or_none(der_vals),
            "MS":  mean_or_none(ms_vals),
            "FA":  mean_or_none(fa_vals),
            "SC":  mean_or_none(sc_vals),
        }
        all_metrics.append(row)
        print(
            f"\n  [{system_name}]  RTF={row['RTF']}  "
            + (f"DER={row['DER']}%  MS={row['MS']}%  FA={row['FA']}%  SC={row['SC']}%"
               if row["DER"] is not None else "DER=—  (нет ground truth)")
        )

    # --- Итоговая таблица ---
    if not all_metrics:
        print("\nНет результатов.")
        return

    df = pd.DataFrame(all_metrics).set_index("system")
    print("\n" + "=" * 65)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("=" * 65)
    cols = [c for c in ["RTF", "DER", "MS", "FA", "SC", "n_files"] if c in df.columns]
    print(df[cols].to_string())
    print()

    valid = df[df["DER"].notna()] if "DER" in df.columns else pd.DataFrame()
    if not valid.empty:
        best = valid["DER"].idxmin()
        print(f"Лучший DER    : {best} ({valid.loc[best, 'DER']}%)")
    if df["RTF"].notna().any():
        best_rtf = df["RTF"].idxmin()
        print(f"Быстрейший    : {best_rtf} (RTF={df.loc[best_rtf, 'RTF']})")

    df.reset_index().to_csv(output_path, index=False)
    print(f"\nРезультаты сохранены: {output_path}")
    print(f"RTTM-файлы:           {rttm_out_dir}/")


if __name__ == "__main__":
    main()
