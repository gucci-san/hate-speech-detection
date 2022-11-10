import os
import torch

# original competition data path --
data_path = "/mnt/sda1/NISHIKA_DATA/hate-speech-detection/"

# processed input data path --
input_root = "./input/"
if not os.path.exists(input_root):
    os.mkdir(input_root)

# computational result path --
output_root = "./output/"
#output_root = "/mnt/sdb/NISHIKA_DATA/hate-speech-detection/output/"
#output_root = "/mnt/sdc/NISHIKA_DATA/hate-speech-detection/output/"
if not os.path.exists(output_root):
    os.mkdir(output_root)

# experiment-result table path --
experiment_root = "./experiment/"
if not os.path.exists(experiment_root):
    os.mkdir(experiment_root)

# unique-id column name of data records --
id_name = "id"

# classification target --
label_name = "label"

# torch device --
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# mecab dictionary path --
#dic_neologd = "/var/lib/mecab/dic/mecab-ipadic-neologd"
dic_neologd = "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"

# plot color --
pal, color = ["#016CC9", "#DEB078"], ["#8DBAE2", "#EDD3B3"]
