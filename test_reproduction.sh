#!/bin/sh
#python bert_run_train.py --run_id "reproduction_test_A" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "mozafari" --epochs 3 --trial True --folds 3 --seed 42 
#python bert_run_train.py --run_id "reproduction_test_B" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "mozafari" --epochs 3 --trial True --folds 3 --seed 42 
python bert_run_train.py --run_id "reproduction_test_A" --model_name "cl-tohoku/bert-base-japanese" --model_custom_header "mozafari" --epochs 2 --trial True --folds 3 --seed 42 
python bert_run_train.py --run_id "reproduction_test_B" --model_name "cl-tohoku/bert-base-japanese" --model_custom_header "mozafari" --epochs 2 --trial True --folds 3 --seed 42 
python test_reproduction.py --run_id1 "reproduction_test_A" --run_id2 "reproduction_test_B"