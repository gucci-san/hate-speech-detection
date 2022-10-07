# ====================================================== 
#                                      
#   おーぷん2ちゃんねる対話コーパスの前処理スクリプト   
#　　https://github.com/1never/open2ch-dialogue-corpus
#
# ======================================================

import pandas as pd
import numpy as np

from glob import glob
from config import *
from bert_utils import clean_text
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default=None)
args, unknown = parser.parse_known_args()

if args.source not in ["newsplus", "livejupiter", "news4vip"]:
    assert False, ("Define corpus tsv from {newsplus, livejupiter, news4vip}")

def tsv_text_cleaner(text: str) -> str:
    t = text.replace("__BR__", "").replace(" ", "").replace("　", "")
    return t.split("\t")

tsv_line_list = []
with open(data_path+f"corpus/{args.source}.tsv", encoding="utf-8") as f:
    for l in f:
        tsv_line_list += tsv_text_cleaner(l)
df = pd.DataFrame({
    "raw_text": tsv_line_list,
    "source": args.source
})
df["clean_text"] = df["raw_text"].parallel_map(lambda x: clean_text(x))
df = df.drop(["raw_text"], axis=1)
df.to_feather(f"{input_root}{args.source}.feather")