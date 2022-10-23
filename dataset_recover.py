import pandas as pd
import numpy as np

import os

from tqdm import tqdm
from colorama import Fore
y_ = Fore.YELLOW; sr_ = Fore.RESET
from config import *


def Debug_print(x):
    print(f"{y_}{x}{sr_}")
    return None


def recover_original_text(df, corpus_df):

    original_text_list = []
    nofinds_data_index_list = []
    duplicated_data_index_list = []
    for i in tqdm(range(df.shape[0]), total=df.shape[0]):
        first_sentence = df.loc[i, "first_sentence"]
        text_picked = [x for x in corpus_df["text"].values.tolist() if first_sentence in x]

        if len(text_picked) == 0: # データセットはある程度正規化されているので、データ元からヒットしないケースがある --
            nofinds_data_index_list.append(i)
            original_text_list.append("")
            continue

        elif len(text_picked) > 1: # 1つの文に対して2つ以上のレスアンカーがついていた場合, 2つ以上ヒットする --
            duplicated_data_index_list.append(i)  # 分離不能な場合があるので諦めてindexだけメモっときます --
            original_text_list.append(repr(text_picked[0]))

        else:  # 1つだけヒット
            original_text_list.append(repr(text_picked[0]))

    df["original_text"] = original_text_list
    df = df.drop(["first_sentence"], axis=1)

    return df, duplicated_data_index_list, nofinds_data_index_list




# ##########################################
#
#               データセット読み込み
#
# ##########################################
train_df = pd.read_csv(data_path+"train.csv").head(100)
test_df = pd.read_csv(data_path+"test.csv").head(100)

output_path = f"{input_root}/dataset_with_original_text"

if not os.path.exists(output_path):
    os.mkdir(output_path)


# ##########################################
#
#               原文tsvの読み込み
#
# ##########################################
corpus_df_list = []
for source in ["newsplus", "news4vip", "livejupiter"]:
    corpus_df = pd.DataFrame(columns=["id", "source", "text"])
    tsv_line_list = []
    with open(input_root+f"corpus/{source}.tsv", encoding="utf-8") as f:
        for i, l in tqdm(enumerate(f)):
            tsv_line_list.append(l)
    corpus_df["text"] = tsv_line_list
    corpus_df["source"] = source
    corpus_df = corpus_df.reset_index(drop=False)
    corpus_df["id"] = corpus_df["source"] + "_" + corpus_df["index"].astype(str)
    corpus_df = corpus_df.drop(["index"], axis=1)
    corpus_df_list.append(corpus_df)

corpus_df = pd.concat(corpus_df_list)
corpus_df = corpus_df.reset_index(drop=True)


# 実行 --
train_df["first_sentence"] = train_df["text"].map(lambda x: x.split("\n")[0])
train_df, train_duplicated_index_list, train_nofinds_index_list = recover_original_text(train_df, corpus_df)

test_df["first_sentence"] = test_df["text"].map(lambda x: x.split("\n")[0])
test_df, test_duplicated_index_list, test_nofinds_index_list = recover_original_text(test_df, corpus_df)


# 保存 --
train_df.to_feather(output_path+"/train_with_original_text.feather") 
test_df.to_feather(output_path+"/test_with_original_text.feather")

pd.Series(train_duplicated_index_list).to_json(output_path+"/train_duplicated_index.json", indent=4)
pd.Series(test_duplicated_index_list).to_json(output_path+"/test_duplicated_index.json", indent=4)
pd.Series(train_nofinds_index_list).to_json(output_path+"/train_nofinds_index.json", indent=4)
pd.Series(test_nofinds_index_list).to_json(output_path+"/test_nofinds_index.json", indent=4)