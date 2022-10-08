# single run --
#python bert_run_train.py --run_id "bert_baseline" --model_name "cl-tohoku/bert-base-japanese-whole-word-masking" --epochs 10
#python bert_run_train.py --run_id "roberta_baseline" --model_name "rinna/japanese-roberta-base" --epochs 10 --trial True

## # prediction --
#python bert_run_test.py --run_id "bert_baseline"
python bert_run_test.py --run_id "roberta_baseline"