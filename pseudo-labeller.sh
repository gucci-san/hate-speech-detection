#!/bin/sh

python pseudo-labeller.py \
--pseudo_label_id "tmp2" \
--run_id_list "roberta_large" "roberta_large_conv" \
--threshold 0.6 