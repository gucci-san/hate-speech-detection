#!/bin/sh

## test --
#python bert_run_train.py --run_id "tmp" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 1 --trial True --folds 2

# serious test --
python bert_run_train.py --run_id "tmp" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 6 --trial True --folds 5



# single run --
#python bert_run_train.py --run_id "bert_baseline" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "roberta_baseline" --model_name "rinna/japanese-roberta-base" --epochs 10 --trial True
#python bert_run_train.py --run_id "electra_baseline" --model_name "izumi-lab/electra-base-japanese-discriminator" --epochs 10 --trial True
#python bert_run_train.py --run_id "roformer_baseline" --model_name "ganchengguang/Roformer-base-japanese" --epochs 10
#python bert_run_train.py --run_id "bert_large_cat4" --model_custom_header "concatenate-4" --model_name "cl-tohoku/bert-large-japanese" --epochs 10
#python bert_run_train.py --run_id "roberta_large" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_fold4" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10 --folds 4
#python bert_run_train.py --run_id "mdeberta_base" --use_amp False --trial True --model_name "microsoft/mdeberta-v3-base" --epochs 10
#python bert_run_train.py --run_id "gpt2_base" --model_custom_header "concatenate-4" --model_name "rinna/japanese-gpt2-medium" --epochs 10
#python bert_run_train.py --run_id "xlm_roberta_large" --model_custom_header "concatenate-4" --model_name "xlm-roberta-large" --epochs 10
#python bert_run_train.py --run_id "distilbert_base_japanese" --model_custom_header "concatenate-4" --model_name "bandainamco-mirai/distilbert-base-japanese" --epochs 10

# 本命 --
#python bert_run_train.py --run_id "roberta_large_cat4" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_corpus_check" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10 --train_data "raw+corpus_label_debug"
#python bert_run_train.py --run_id "roberta_large_cat4_corpus_check2" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10 --train_data "raw+corpus_label_debug" --learning_rate 5e-6 --min_lr 1e-7

# さすがにパラメータ多すぎてメモリ足りない --
#python bert_run_train.py --run_id "tmp" --model_name "rinna/japanese-gpt-1b" --folds 2 --trial True --train_batch_size 2 --n_accumulate 16

# roberta-largeが良かったので、パターンを試す
## ## custom_header
#python bert_run_train.py --run_id "roberta_large_maxpool" --model_name "nlp-waseda/roberta-large-japanese-seq512" --epochs 10
#python bert_run_train.py --run_id "roberta_large_lstm" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "lstm" --epochs 10
#python bert_run_train.py --run_id "roberta_large_conv" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "conv" --epochs 10

## # ## folds
#python bert_run_train.py --run_id "roberta_large_cat4_fold3" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 3 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold7" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 7 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold8" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 8 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold8_seed93" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 8 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold8_seed128" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 8 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold8_seed256" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 8 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold9" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 9 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold10" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 10 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold11" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 11 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold12" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 12 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold16" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 16 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold20" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 20 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_fold25" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 25 --model_custom_header "concatenate-4" --epochs 10

## ## learning-rate
#python bert_run_train.py --run_id "roberta_large_cat4_lr2e-5" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --learning_rate 2e-5 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_lr5e-5" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --learning_rate 5e-5 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_lr1e-4" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --learning_rate 1e-4 --model_custom_header "concatenate-4" --epochs 10

## ## train_batch_size
#python bert_run_train.py --run_id "roberta_large_cat4_batch16" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 16 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_batch64" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 64 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_batch128" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 128 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_batch256" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 128 --n_accumulate 2 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_batch512" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 128 --n_accumulate 4 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_batch8" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 8 --model_custom_header "concatenate-4" --epochs 10

## python bert_run_train.py --run_id "roberta_large_cat4_org_batch16_acc2_fold8" --train_data "raw_original_text" --train_batch_size 16 --valid_batch_size 16 --n_accumulate 2 --max_length 256 \
## --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10 --folds 8
## 
## # Batch-accumulation --
## # to -> batch64
## python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch64" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 64 --model_custom_header "concatenate-4" --epochs 10
## python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch32_acc2" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 32 --n_accumulate 2 --model_custom_header "concatenate-4" --epochs 10
## python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch16_acc4" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 16 --n_accumulate 4 --model_custom_header "concatenate-4" --epochs 10
## python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch8_acc8" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 8 --n_accumulate 8 --model_custom_header "concatenate-4" --epochs 10
## python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch4_acc16" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 4 --n_accumulate 16 --model_custom_header "concatenate-4" --epochs 10
## python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch2_acc32" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 2 --n_accumulate 32 --model_custom_header "concatenate-4" --epochs 10
## python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch1_acc64" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 1 --n_accumulate 64 --model_custom_header "concatenate-4" --epochs 10
## python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch64"
## python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch32_acc2"
## python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch16_acc4"
## python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch8_acc8"
## python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch4_acc16"
## python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch2_acc32"
## python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch1_acc64"

# to -> batch32
# python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch32" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 32 --model_custom_header "concatenate-4" --epochs 10
# python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch16_acc2" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 16 --n_accumulate 2 --model_custom_header "concatenate-4" --epochs 10
# python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch8_acc4" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 8 --n_accumulate 4 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch4_acc8" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 4 --n_accumulate 8 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch2_acc16" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 2 --n_accumulate 16 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "compare_batch_accum_roberta_large_cat4_batch1_acc32" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 1 --n_accumulate 32 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch32"
#python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch16_acc2"
#python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch8_acc4"
#python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch4_acc8"
#python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch2_acc16"
#python bert_run_test.py --run_id "compare_batch_accum_roberta_large_cat4_batch1_acc32"

# 10/27時点で最強の組み合わせを試す --
#python bert_run_train.py --run_id "roberta_large_cat4_batch4_acc8_folds8" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 8 --train_batch_size 4 --n_accumulate 8 --model_custom_header "concatenate-4" --epochs 10
#python bert_run_test.py --run_id "roberta_large_cat4_batch4_acc8_folds8"

# ヘッダにtanhを足してみる --
#python bert_run_train.py --run_id "roberta_large_cat4_batch4_acc8_tanh" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 4 --n_accumulate 8 --model_custom_header "concatenate-4" --epochs 10


# メモリエラー
#python bert_run_train.py --run_id "roberta_large_cat4_batch512" --model_name "nlp-waseda/roberta-large-japanese-seq512" --folds 5 --train_batch_size 512 --model_custom_header "concatenate-4" --epochs 10

## custom_header compare --
#python bert_run_train.py --run_id "bert_conv" --model_custom_header "conv" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "bert_lstm" --model_custom_header "lstm" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "bert_cat4" --model_custom_header "concatenate-4" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10

## roberta-large_cat4 + pseudo-label
#python bert_run_train.py --run_id "roberta_large_cat4_pseudo" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10 --train_data "raw+test_pseudo"

## roberta-largeをampで回すとどうなる？ --
#python bert_run_train.py --run_id "roberta_large_amp" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10

#python bert_run_train.py --run_id "roberta_large_msd_cat4" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10
#python bert_run_train.py --run_id "roberta_large_mixout_cat4" --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 10

# original_textを使う --
#python bert_run_train.py --run_id "tmp" --trial True --folds 2 --model_custom_header "max_pooling" --train_data "raw_original_text" --max_length 384
#python bert_run_test.py --run_id "tmp"

# original textで挙動を見ていく --
#python bert_run_train.py --run_id "bert_base_whm_org_text" --folds 5 --model_custom_header "concatenate-4" --train_data "raw_original_text" --max_length 384 --epochs 10
#python bert_run_train.py --run_id "roberta_large_cat4_org_batch8" --train_data "raw_original_text" --train_batch_size 8 --max_length 256 --model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 5

#python bert_run_train.py --run_id "roberta_large_cat4_org_batch16" --train_data "raw_original_text" --train_batch_size 16 --valid_batch_size 16 --max_length 256 \
#--model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 5
#python bert_run_train.py --run_id "roberta_large_cat4_org_batch16_acc2_fold8" --train_data "raw_original_text" --train_batch_size 16 --valid_batch_size 16 --n_accumulate 2 --max_length 256 \
#--model_name "nlp-waseda/roberta-large-japanese-seq512" --model_custom_header "concatenate-4" --epochs 5

# trial --
#python bert_run_train.py --run_id "tmp_epochs1" --folds 2 --model_custom_header "max_pooling" --epochs 2
#python bert_run_train.py --run_id "tmp_before_refactor" --folds 3 --model_custom_header "max_pooling" --epochs 5
#python bert_run_test.py --run_id "tmp_epochs1"
#python bert_run_train.py --run_id "tmp" --trial True --folds 2 --model_name "cl-tohoku/bert-large-japanese" --train_batch_size 8 --model_custom_header "max_pooling" --train_data "raw_original_text" --max_length 256
#python bert_run_test.py --run_id "tmp"

## # prediction -
#python bert_run_test.py --run_id "bert_baseline"
#python bert_run_test.py --run_id "roberta_baseline"
#python bert_run_test.py --run_id "electra_baseline"
#python bert_run_test.py --run_id "roformer_baseline"
#python bert_run_test.py --run_id "bert_conv"
#python bert_run_test.py --run_id "bert_lstm"
#python bert_run_test.py --run_id "bert_cat4"
#python bert_run_test.py --run_id "bert_large_cat4"
#python bert_run_test.py --run_id "roberta_large"
#python bert_run_test.py --run_id "roberta_large_maxpool"
#python bert_run_test.py --run_id "roberta_large_lstm"
#python bert_run_test.py --run_id "roberta_large_conv
#python bert_run_test.py --run_id "mdeberta_base_cat4"
#python bert_run_test.py --run_id "roberta_large_cat4_fold3"
#python bert_run_test.py --run_id "roberta_large_cat4_fold7"
#python bert_run_test.py --run_id "roberta_large_cat4_fold10"
#python bert_run_test.py --run_id "roberta_large_cat4_pseudo"
#
#python bert_run_test.py --run_id "roberta_large_cat4_batch16"
#python_bert_run_test.py --run_id "roberta_large_cat4_batch64"
#python bert_run_test.py --run_id "roberta_large_fold4"

#python bert_run_test.py --run_id "roberta_large_amp"
#python bert_run_test.py --run_id "roberta_large_msd_cat4"

#python bert_run_test.py --run_id "roberta_large_mixout_cat4"

#python bert_run_test.py --run_id "xlm_roberta_large"
#python bert_run_test.py --run_id "roberta_large_cat4"

#python bert_run_test.py --run_id "roberta_large_cat4_corpus_check"
#python bert_run_test.py --run_id "roberta_large_cat4_corpus_check2"
#python bert_run_test.py --run_id "roberta_large_cat4_batch128"
#python bert_run_test.py --run_id "roberta_large_cat4_batch256"
#python bert_run_test.py --run_id "roberta_large_cat4_batch512"
#python bert_run_test.py --run_id "roberta_large_cat4_batch8"
#python bert_run_test.py --run_id "roberta_large_cat4_batch64"
#python bert_run_test.py --run_id "roberta_large_cat4_fold8" 
#python bert_run_test.py --run_id "roberta_large_cat4_fold8_seed93" 
#python bert_run_test.py --run_id "roberta_large_cat4_fold8_seed128" 
#python bert_run_test.py --run_id "roberta_large_cat4_fold8_seed256" 
#python bert_run_test.py --run_id "roberta_large_cat4_fold9" 
#python bert_run_test.py --run_id "roberta_large_cat4_fold11" 
#python bert_run_test.py --run_id "roberta_large_cat4_fold10"
#python bert_run_test.py --run_id "roberta_large_cat4_fold12"
#python bert_run_test.py --run_id "roberta_large_cat4_fold16"
#python bert_run_test.py --run_id "roberta_large_cat4_fold20"
#python bert_run_test.py --run_id "roberta_large_cat4_fold25"

#python bert_run_test.py --run_id "bert_base_whm_org_text"
#python bert_run_test.py --run_id "roberta_large_cat4_org_batch8"
#python bert_run_test.py --run_id "roberta_large_cat4_org_batch16_acc2_fold8"

#python bert_run_test.py --run_id "roberta_large_cat4_org_batch16_acc2_fold8"