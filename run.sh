#!/bin/sh

# single run --
#python bert_run_train.py --run_id "bert_baseline" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "roberta_baseline" --model_name "rinna/japanese-roberta-base" --epochs 10 --trial True
#python bert_run_train.py --run_id "electra_baseline" --model_name "izumi-lab/electra-base-japanese-discriminator" --epochs 10 --trial True
#python bert_run_train.py --run_id "roformer_baseline" --model_name "ganchengguang/Roformer-base-japanese" --epochs 10
#python bert_run_train.py --run_id "bert_large" --model_name "cl-tohoku/bert-large-japanese" --epochs 10

## custom_header compare --
#python bert_run_train.py --run_id "bert_conv" --model_custom_header "conv" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "bert_lstm" --model_custom_header "lstm" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "bert_cat4" --model_custom_header "concatenate-4" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10

# trial --
python bert_run_train.py --run_id "tmp" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 2 --trial True

## # prediction --
#python bert_run_test.py --run_id "bert_baseline"
#python bert_run_test.py --run_id "roberta_baseline"
#python bert_run_test.py --run_id "electra_baseline"
#python bert_run_test.py --run_id "roformer_baseline"
#python bert_run_test.py --run_id "bert_conv"
#python bert_run_test.py --run_id "bert_lstm"
#python bert_run_test.py --run_id "bert_cat4"
#python bert_run_test.py --run_id "bert_large"