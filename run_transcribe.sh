#!/bin/bash
# Wrapper script: restarts transcribe.py on crash, cleans memory before each run.
# Usage: bash run_transcribe.sh

INPUT="/home/dmitrii/messages_2020_2026/"
OUTPUT="/home/dmitrii/dataset_preparer/dataset.csv"
VENV="/home/dmitrii/dataset_preparer/venv/bin/python"
SCRIPT="/home/dmitrii/dataset_preparer/transcribe.py"
LOG="/home/dmitrii/dataset_preparer/run_transcribe.log"

# Optional: add --hf-token hf_xxx here when you have GPU
EXTRA_ARGS=""

MAX_RETRIES=999
RETRY_DELAY=10  # seconds between restarts

attempt=0

while [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    echo "========================================" | tee -a "$LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt $attempt" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"

    # Clean swap before each run
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning swap..." | tee -a "$LOG"
    sudo swapoff -a && sudo swapon -a
    sleep 2

    # Run transcription
    "$VENV" "$SCRIPT" \
        --input "$INPUT" \
        --output "$OUTPUT" \
        $EXTRA_ARGS \
        2>&1 | tee -a "$LOG"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished successfully." | tee -a "$LOG"
        exit 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Process exited with code $EXIT_CODE. Restarting in ${RETRY_DELAY}s..." | tee -a "$LOG"
    sleep $RETRY_DELAY
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max retries reached." | tee -a "$LOG"
exit 1
