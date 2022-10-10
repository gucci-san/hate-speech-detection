import os

import pandas as pd
import numpy as np

from config import *
from bert_utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pseudo_label_id", type=str, default="tmp", required=True)
parser.add_argument("--run_id_list", nargs="*", type=str, default=None)
parser.add_argument("--threshold", type=float, default=0.5)
args, unknown = parser.parse_known_args()

pseudo_label_id = args.pseudo_label_id
run_id_list = args.run_id_list
threshold = args.threshold

settings = pd.Series(dtype=object)
settings["pseudo_label_id"] = pseudo_label_id
settings["run_id_list"] = run_id_list
settings["threshold"] = threshold
settings["save_path"] = f"{input_root}pseudo_label_{settings.pseudo_label_id}/"

if not os.path.exists(f"{settings.save_path}"):
    os.mkdir(f"{settings.save_path}")
settings.to_json(f"{settings.save_path}settings.json", indent=4)

train_original_cols = pd.read_csv(f"{data_path}train.csv").columns

# dfにtest_predがthreshold以上のレコードを保存してfeather化する --
# ## run_id_listに含めたsingle-modelの予測結果の平均とthresholdを比較 --
df = pd.DataFrame()
for i, run_id in enumerate(run_id_list):
    test_df = pd.read_feather(f"{output_root}{run_id}/test_df.feather")
    df[f"{run_id}_oof_class_1"] = test_df["model_oof_class_1"]

df = pd.concat([test_df.loc[:, ["index", "id", "source", "text", "clean_text"]], df], axis=1)
df[label_name] = df.loc[:, df.columns.str.contains("oof")].mean(axis=1)
df = df.sort_values(label_name, ascending=False)
df = df.loc[(df[label_name] > threshold), train_original_cols].reset_index(drop=True)
df.to_feather(f"{settings.save_path}test_pseudo_labeled.feather")