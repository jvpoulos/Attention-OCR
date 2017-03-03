#!/usr/bin/env sh

python src/launcher.py \
--phase=train \
--data-path=gwdb/word_labels_train.txt \
--data-base-dir=gwdb \
--model-dir=model_gwdb \
--log-path=log_gwdb_words.txt \
--no-load-model \
--batch-size=8 \
--target-vocab-size=65