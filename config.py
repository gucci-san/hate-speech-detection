import torch

# SEED = 42
# SEED = 93
SEED = 256

data_path = "/mnt/sdb/NISHIKA_DATA/hate-speech-detection/"
input_root = "./input/"
output_root = "./output/"
# output_root = "/mnt/sdb/NISHIKA_DATA/hate-speech-detection/output/"
experiment_root = "./experiment/"

id_name = "id"
label_name = "label"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# mecab --
dic_neologd = "/var/lib/mecab/dic/mecab-ipadic-neologd"

# plot color --
pal, color = ["#016CC9", "#DEB078"], ["#8DBAE2", "#EDD3B3"]
