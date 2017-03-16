#!/usr/bin/env sh

# Words
python3 src/launcher.py \
--phase=train \
--data-path=gwdb/word_labels_train.txt \
--data-base-dir=gwdb \
--model-dir=model_gwdb \
--log-path=log_gwdb_words_train.txt \
--no-load-model \
--batch-size=8 \
--gpu-id=0 \
--use-gru

# Lines - use model weights from words
python3 src/launcher.py \
--phase=train \
--data-path=gwdb/transcription_train.txt \
--data-base-dir=gwdb \
--model-dir=model_gwdb \
--log-path=log_gwdb_lines.txt \
--batch-size=8 \
--gpu-id=0 \
--use-gru