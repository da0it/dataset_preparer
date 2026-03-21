# Dataset Preparer

Набор утилит для:

- транскрибации звонков (`transcribe.py`)
- очистки и подготовки CSV (`prepare_dataset.py`, `dataset_cli.py`)
- обучения моделей (`train.py`, `train_advanced.py`, `ensemble.py`)
- калибровки трансформеров (`calibrate.py`)

## Основной пайплайн

```bash
python transcribe.py --input /path/to/mp3s --output dataset.csv
python prepare_dataset.py --input dataset.csv --output dataset_clean.csv
python dataset_cli.py clean-noise --input dataset_clean.csv --output dataset_ready.csv
python train_advanced.py -i dataset_ready.csv -g baseline,rubert,xlmr -t call_purpose
```

`train_advanced.py` теперь является основной точкой входа для обучения.
`train.py` оставлен только как совместимая обертка для старого набора классических моделей
и внутри делегирует запуск в `train_advanced.py -g legacy-baseline`.

## Единый CLI для датасета

Вместо нескольких мелких скриптов используй `dataset_cli.py`.

### Нормализация лейблов

```bash
python dataset_cli.py normalize-labels --input dataset.csv --output dataset_norm.csv --sep ";"
```

### Удаление шумовых фраз

```bash
python dataset_cli.py clean-noise --input dataset_clean.csv --output dataset_no_noise.csv --sep ","
```

### Фильтрация и очистка filename

```bash
python dataset_cli.py filter-ready --input dataset.csv --output dataset_ready.csv --sep ";"
```

Старые команды `normalize_labels.py`, `clean_noise.py`, `filter_and_clean.py` оставлены как совместимые обертки.

## Скрипты по ролям

Основные сценарии и готовые команды собраны в
[`docs/usage_scenarios.md`](/Users/dmitrii/dataset_preparer/docs/usage_scenarios.md).

## Структура репозитория

- `dataset_tools/` — реальные реализации подготовки и обработки CSV
- `training_tools/` — реальные реализации обучения и калибровки
- `transcription_tools/` — реализация транскрибации
- корень репозитория — совместимые entrypoint-обертки, README и shell helpers

### Транскрибация

- `transcribe.py` — WhisperX -> `dataset.csv`, `dataset_timed.csv`, `dataset_speakers.csv`

### Подготовка датасета

- `prepare_dataset.py` — PII cleanup, NER redaction, normalization
- `dataset_cli.py` — unified CLI для label normalization, noise cleanup, ready-filtering
- `inspect_classes.py` — ручная инспекция классов
- `dataset_variants.py` — подготовка `multiclass` / `binary_spam` срезов для обучения

### Обучение

- `train_advanced.py` — основной ML/DL пайплайн
- `train.py` — legacy wrapper для совместимости (`legacy-baseline`)
- `ensemble.py` — soft-voting ансамбль
- `calibrate.py` — temperature scaling для transformer-моделей

### Исследования ASR

- `asr_benchmark/*` — отдельные benchmark-скрипты, не нужны для основного train/inference pipeline

## Замечания по структуре

- Артефакты моделей и логи теперь игнорируются через `.gitignore`
- `__pycache__` и `.pyc` не должны попадать в репозиторий
- Если нужна дополнительная чистка структуры, следующий логичный шаг — вынести training и dataset tooling в отдельные подпапки (`training/`, `dataset/`, `asr/`)
