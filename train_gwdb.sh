#!/usr/bin/env sh

# Words
python3 src/launcher.py \
--phase=train \
--data-path=gwdb/word_labels_train.txt \
--data-base-dir=gwdb \
--model-dir=model_gwdb \
--log-path=log_gwdb_words_train.txt \
--batch-size=8 \
--num-epoch=100 \
--gpu-id=0 \
--use-gru \
--load-model


# Lines
python3 src/launcher.py \
--phase=train \
--data-path=gwdb/transcription_train.txt \
--data-base-dir=gwdb \
--model-dir=model_gwdb \
--log-path=log_gwdb_lines.txt \
--batch-size=8 \
--num-epoch=100 \
--gpu-id=0 \
--use-gru \
--augmentation=0.5 \
--load-model