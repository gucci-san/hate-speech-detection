import pandas as pd

from glob import glob

paths = glob("./input/corpus_label_roberta_large_cat4/*.feather")
dfs = []
for path in paths:
    df = pd.read_feather(path)
    dfs.append(df)

df = pd.concat(dfs)

df.reset_index(drop=True).to_feather("./input/corpus_label_roberta_large_cat4/corpus_labeled.feather")