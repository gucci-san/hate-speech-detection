#!/bin/sh

python pseudo-labeller.py \
--pseudo_label_id "base" \
--run_id_list "roberta_large" "bert_large" "mdeberta_base" \
--threshold 0.65