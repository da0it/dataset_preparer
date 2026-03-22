# Usage Scenarios

Единый файл с основными сценариями работы и готовыми командами.

## 1. Транскрибация аудио в CSV

```bash
python transcribe.py \
  --input /path/to/mp3_dir \
  --output dataset.csv \
  --device cuda \
  --compute-type float16
```

Результат:

- `dataset.csv`
- `dataset_timed.csv`
- `dataset_speakers.csv` при включенной diarization

## 2. Полная подготовка датасета

### 2.1 Очистка текста и PII

```bash
python prepare_dataset.py \
  --input dataset.csv \
  --output dataset_clean.csv
```

### 2.2 Удаление шумовых фраз

```bash
python dataset_cli.py clean-noise \
  --input dataset_clean.csv \
  --output dataset_no_noise.csv
```

### 2.3 Нормализация лейблов

```bash
python dataset_cli.py normalize-labels \
  --input dataset_no_noise.csv \
  --output dataset_norm.csv \
  --sep ","
```

### 2.4 Финальная фильтрация и очистка filename

```bash
python dataset_cli.py filter-ready \
  --input dataset_norm.csv \
  --output dataset_ready.csv \
  --sep ","
```

## 3. Обучение мультиклассовой модели

```bash
python train_advanced.py \
  -i multiclass.csv \
  -g baseline,rubert,xlmr \
  -t all \
  -o models_multiclass
```

## 4. Обучение бинарной spam/non_spam модели

```bash
python train_advanced.py \
  -i binary.csv \
  --dataset-variant binary_spam \
  -g baseline,rubert,xlmr \
  -t call_purpose \
  -o models_binary
```

## 5. Один запуск для мультикласса и бинарки

```bash
python train_advanced.py \
  -i multiclass.csv \
  --binary-input binary.csv \
  -g baseline,rubert,xlmr \
  -t all \
  -o models_all
```

Результаты будут разложены в:

- `models_all/multiclass/...`
- `models_all/binary_spam/...`

## 6. Legacy baseline из старого train.py

```bash
python train.py \
  -i dataset_ready.csv \
  -t call_purpose \
  -o models_legacy
```

Эквивалентно:

```bash
python train_advanced.py \
  -i dataset_ready.csv \
  -g legacy-baseline \
  -t call_purpose \
  -o models_legacy
```

## 7. Ансамбль

```bash
python ensemble.py \
  -i binary.csv \
  --dataset-variant binary_spam \
  -t call_purpose \
  -o models_ensemble
```

## 8. Калибровка трансформера

```bash
python calibrate.py \
  --model-dir models_binary/call_purpose/RuBERT \
  --input binary.csv \
  --target call_purpose \
  --sep ","
```

## 9. Подбор threshold для spam-gate

Точный подбор decision threshold по размеченному validation / diagnostic CSV.

```bash
python optimize_threshold.py \
  -i diagnostics_25_calls_svm.csv \
  -o threshold_search_svm \
  --sep "," \
  --score-col confidence \
  --label-col manual_label \
  --positive-label spam \
  --negative-label non_spam \
  --rule greater_equal \
  --optimize accuracy
```

Результат:

- `threshold_search_svm/threshold_sweep.csv` — все проверенные threshold и метрики
- `threshold_search_svm/best_threshold.json` — лучший threshold и итоговые метрики
- `threshold_search_svm/threshold_summary.txt` — короткая текстовая сводка

Для несбалансированного validation-набора обычно полезно также проверить:

```bash
python optimize_threshold.py \
  -i diagnostics_25_calls_svm.csv \
  -o threshold_search_svm_balanced \
  --sep "," \
  --score-col confidence \
  --label-col manual_label \
  --positive-label spam \
  --negative-label non_spam \
  --rule greater_equal \
  --optimize balanced_accuracy
```

## 10. Инференс SVM по реальным звонкам

```bash
python -c 'import joblib, pandas as pd; model = joblib.load("models_binary/call_purpose/TF-IDF_plus_SVM.joblib"); df = pd.read_csv("real_calls_clean_no_noise.csv", dtype=str).fillna(""); df["pred"] = model.predict(df["text"]); df["score"] = model.decision_function(df["text"]); out = df[["filename","text","pred","score"]]; out.to_csv("real_calls_pred.csv", index=False, encoding="utf-8"); print(out.to_string(index=False))'
```

## 11. Инференс ruBERT по реальным звонкам

```bash
python -c 'import joblib, pandas as pd, torch; from transformers import AutoTokenizer, AutoModelForSequenceClassification; model_dir="models_binary/call_purpose/RuBERT"; df=pd.read_csv("real_calls_clean_no_noise.csv", dtype=str).fillna(""); texts=df["text"].tolist(); le=joblib.load(model_dir + "/label_encoder.joblib"); tokenizer=AutoTokenizer.from_pretrained(model_dir); model=AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu"); device=next(model.parameters()).device; model.eval(); preds=[]; scores=[]; bs=16; \
for i in range(0, len(texts), bs): \
    batch=texts[i:i+bs]; enc=tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt"); enc={k:v.to(device) for k,v in enc.items()}; \
    with torch.no_grad(): logits=model(**enc).logits; proba=torch.softmax(logits, dim=1); idx=torch.argmax(proba, dim=1).cpu().numpy(); mx=torch.max(proba, dim=1).values.cpu().numpy(); \
    preds.extend(le.inverse_transform(idx)); scores.extend(mx.tolist()); \
df["pred"]=preds; df["score"]=scores; out=df[["filename","text","pred","score"]]; out.to_csv("real_calls_pred_rubert.csv", index=False, encoding="utf-8"); print(out.to_string(index=False))'
```

## 12. Полный practical pipeline для реальных звонков

```bash
python transcribe.py --input /path/to/mp3_dir --output real_calls.csv --device cuda --compute-type float16 && \
python prepare_dataset.py --input real_calls.csv --output real_calls_clean.csv && \
python dataset_cli.py clean-noise --input real_calls_clean.csv --output real_calls_clean_no_noise.csv && \
python -c 'import joblib, pandas as pd; model = joblib.load("models_binary/call_purpose/TF-IDF_plus_SVM.joblib"); df = pd.read_csv("real_calls_clean_no_noise.csv", dtype=str).fillna(""); df["pred"] = model.predict(df["text"]); df["score"] = model.decision_function(df["text"]); out = df[["filename","text","pred","score"]]; out.to_csv("real_calls_pred.csv", index=False, encoding="utf-8"); print(out.to_string(index=False))'
```
