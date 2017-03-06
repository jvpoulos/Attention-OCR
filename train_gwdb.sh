#!/usr/bin/env sh

python src/launcher.py \
--phase=train \
--data-path=gwdb/word_labels_train_clean.txt \
--data-base-dir=gwdb \
--model-dir=model_demo \
--log-path=log_gwdb_words.txt \
--load-model 

