#!/bin/sh

# single run --
#python bert_run_train.py --run_id "bert_baseline" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "roberta_baseline" --model_name "rinna/japanese-roberta-base" --epochs 10 --trial True
#python bert_run_train.py --run_id "electra_baseline" --model_name "izumi-lab/electra-base-japanese-discriminator" --epochs 10 --trial True
#python bert_run_train.py --run_id "roformer_baseline" --model_name "ganchengguang/Roformer-base-japanese" --epochs 10
#python bert_run_train.py --run_id "bert_large" --model_name "cl-tohoku/bert-large-japanese" --epochs 10
#python bert_run_train.py --run_id "roberta_large" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10

# roberta-largeが良かったので、パターンを試す
## ## custom_header
#python bert_run_train.py --run_id "roberta_large_maxpool" --model_name "nlp-waseda/roberta-large-japanese-seq512" --epochs 10
#python bert_run_train.py --run_id "roberta_large_lstm" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "lstm" --epochs 10
#python bert_run_train.py --run_id "roberta_large_conv" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "conv" --epochs 10

## # ## folds
#python bert_run_train.py --run_id "roberta_large_cat4_fold3" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 3 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold7" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 7 --model_custom_header "concatenate-4" --epochs 10
python bert_run_train.py --run_id "roberta_large_cat4_fold10" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 10 --model_custom_header "concatenate-4" --epochs 10

# ## learning-rate
python bert_run_train.py --run_id "roberta_large_cat4_lr2e-5" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --learning_rate 2e-5 --model_custom_header "concatenate-4" --epochs 10
python bert_run_train.py --run_id "roberta_large_cat4_lr5e-5" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --learning_rate 5e-5 --model_custom_header "concatenate-4" --epochs 10
python bert_run_train.py --run_id "roberta_large_cat4_lr1e-4" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --learning_rate 1e-4 --model_custom_header "concatenate-4" --epochs 10

## custom_header compare --
#python bert_run_train.py --run_id "bert_conv" --model_custom_header "conv" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "bert_lstm" --model_custom_header "lstm" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "bert_cat4" --model_custom_header "concatenate-4" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10

## # trial --
## python bert_run_train.py --run_id "tmp" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 2 --trial True

## # prediction --
#python bert_run_test.py --run_id "bert_baseline"
#python bert_run_test.py --run_id "roberta_baseline"
#python bert_run_test.py --run_id "electra_baseline"
#python bert_run_test.py --run_id "roformer_baseline"
#python bert_run_test.py --run_id "bert_conv"
#python bert_run_test.py --run_id "bert_lstm"
#python bert_run_test.py --run_id "bert_cat4"
#python bert_run_test.py --run_id "bert_large"
#python bert_run_test.py --run_id "roberta_large"
#python bert_run_test.py --run_id "roberta_large_maxpool"
#python bert_run_test.py --run_id "roberta_large_lstm"
#python bert_run_test.py --run_id "roberta_large_conv"