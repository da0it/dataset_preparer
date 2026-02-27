pip install whisperx torch

python transcribe.py --input /path/to/mp3s
# → dataset.csv + dataset_timed.csv

python transcribe.py --input /path/to/mp3s --output calls.csv
# → calls.csv + calls_timed.csv

--input  / -i   Папка с аудиофайлами (обязательный)
--output / -o   Выходной CSV (default: dataset.csv)
--model  / -m   Модель Whisper (default: large-v2)
--language / -l Язык (default: ru)
--segments / -s Режим с временными метками
--device / -d   cpu / cuda (автоопределение)
--compute-type  int8 (default) / float16 для GPU
