#!/bin/sh
#python ensemble_optimizer.py --run_id_list \
#    "roberta-large_cat4_batch4_acc8_seed0" \
#    "roberta-large_cat4_batch4_acc8_seed1" \
#    "roberta-large_cat4_batch4_acc8_seed2" \
#    "roberta-large_cat4_batch4_acc8_seed3" \
#    "roberta-large_cat4_batch4_acc8_seed4" \
#    "roberta-large_cat4_batch4_acc8_seed5" \
#    "roberta-large_cat4_batch4_acc8_seed6" \
#    "roberta-large_cat4_batch4_acc8_seed7" \
#    "roberta-large_cat4_batch4_acc8_seed8" \
#    "roberta-large_cat4_batch4_acc8_seed9"

python ensemble_optimizer.py --run_id_list \
    "roberta-large_cat4_batch4_acc8_seed0" \
    "roberta-large_cat4_batch4_acc8_seed5" \
    "roberta-large_cat4_batch4_acc8_seed6" \
    "roberta-large_cat4_batch4_acc8_seed7" \
    "roberta-large_cat4_batch4_acc8_seed8"