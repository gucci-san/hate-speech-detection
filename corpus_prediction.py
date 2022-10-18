import hashlib
import os
import datetime

import pandas as pd
import torch.nn as nn

from transformers import AutoTokenizer, T5Tokenizer
from torch.utils.data import DataLoader
from glob import glob

from bert_utils import *
from config import *

import argparse

# 計算時のsettingはtrainで保存したjsonから読み込む --
# run_idだけ指定 --
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="roberta_large_cat4")
parser.add_argument("--df_start_index", type=int, default=0)
parser.add_argument("--df_end_index", type=int, default=1024)
args, unknown = parser.parse_known_args()
save_path = f"{input_root}corpus_label_{args.run_id}"

# # コーパス作成時のセッティングを保存 --
# if not os.path.exists(f"{save_path}"):
#     os.mkdir(f"{save_path}")

corpus_settings = pd.Series()
corpus_settings["used_model"] = args.run_id
corpus_settings["df_start_index"] = args.df_start_index
corpus_settings["df_end_index"] = args.df_end_index
corpus_settings.to_json(save_path+"/corpus_settings.json", indent=4)

# settings, fine-tuningしたモデル -- 
output_path = f"{output_root}{args.run_id}/"
settings = pd.read_json(f"{output_path}settings.json", typ="series")
model_paths = glob(f"{settings.output_path}*.pth"); model_paths.sort()

# define tokenizer --
tokenizer = define_tokenizer(settings.model_name)

# 対象とするデータの読み込み --
corpus_paths = glob(f"{input_root}*.feather")
Debug_print(corpus_paths)

df = []
for corpus_path in corpus_paths:
    _df = pd.read_feather(corpus_path)
    _df = _df.reset_index(drop=False, names="id")
    _df["id"] = corpus_path.split("/")[-1].split(".")[0] + "_" + _df["id"].astype(str)
    df.append(_df)
df = pd.concat(df)

# 全データやると一生終わらないのでバッチ指定する --
df = df.iloc[args.df_start_index:args.df_end_index, :]

# make test preds --
test_dataset = HateSpeechDataset(
    df, tokenizer=tokenizer, 
    max_length=settings.max_length, num_classes=settings.num_classes, 
    text_col="clean_text", isTrain=False
    )

# batch_size=512でGPU:19GBくらい --
test_loader = DataLoader(test_dataset, batch_size=512, num_workers=2, shuffle=False, pin_memory=True)

preds_list = []
for fold in range(0, settings.folds):
    softmax = nn.Softmax()
    model_id = "model"
    preds = inference(settings.model_name, settings.num_classes, settings.model_custom_header, settings.dropout, model_paths[fold], test_loader, device)
    
    # preds : BERT -> fc, 確率にするためにsoftmaxに通す必要がある --
    preds_list.append(softmax(torch.Tensor(preds)).numpy())

final_preds = np.mean(np.array(preds_list), axis=0)
df[f"{model_id}_pred"] = np.argmax(final_preds, axis=1)
for _class in range(0, settings.num_classes):
    df.loc[:, f"{model_id}_oof_class_{_class}"] = final_preds[:, _class]


# コーパスのラベル付けの結果を集計していく --
log_df = pd.DataFrame(corpus_settings).T
log_df["pred_0"] = df["model_pred"].value_counts()[0]
log_df["pred_1"] = df.shape[0] - df["model_pred"].value_counts()[0]

text = datetime.datetime.now().strftime(format="%Y%m%d-%H%m%S")
hash = hashlib.md5(text.encode("utf-8")).hexdigest()
log_df["hash"] = hash

if not os.path.exists(f"{save_path}/corpus_info.csv"):
    log_df.to_csv(f"{save_path}/corpus_info.csv", index=False)
else:
    log_df.to_csv(f"{save_path}/corpus_info.csv", index=False, mode="a", header=None)

df.reset_index(drop=True).to_feather(f"{save_path}/corpus_labeled_{hash}.feather")