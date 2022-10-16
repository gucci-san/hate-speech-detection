import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from transformers import AdamW

from colorama import Fore; r_=Fore.RED; sr_=Fore.RESET
from glob import glob
from config import *
from bert_utils import *

from mixout import *

import argparse




# ====================================== #
#                                        #
#    -- Define Settings and Constants -- #
#                                        #
# ====================================== #
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("--model_fold", type=int, default=0)
parser.add_argument("--train_id", type=str, default="ttmp")
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--save_mode", type=str, default="save_as_new")
args, unknown = parser.parse_known_args()

# settings, fine-tuningしたモデルを読み込み -- 
output_path = f"{output_root}{args.run_id}/"
settings = pd.read_json(f"{output_path}settings.json", typ="series")
checkpoint_path = f"{settings.output_path}checkpoint-fold{args.model_fold}.pth"

# 学習セッティングの保存 --
train_settings = pd.Series()
train_settings["model_fold"] = args.model_fold
train_settings["train_id"] = args.train_id
train_settings["epochs"] = args.epochs
train_settings["train_batch_size"] = args.train_batch_size
train_settings["save_mode"] = args.save_mode
train_settings["output_path"] = f"{settings.output_path}retrain_{train_settings.train_id}/"
train_settings["from"] = checkpoint_path
train_settings["to"] = train_settings.output_path+"/checkpoint-fold0.pth"
train_settings["seed"] = SEED

# save_mode == "overwrite"でretrainしたcheckpointを上書き --
# ## 最初にbert_run_train.pyで作成したやつは触らない想定 --
if not os.path.exists(train_settings.output_path):
    os.mkdir(train_settings.output_path)
else:
    if train_settings.save_mode == "overwrite":
        pass
    else:
        assert False, (f"{r_}*** ... train_id {train_settings.train_id} alreadly exists ... ***{sr_}")


# 計算時点でのpyファイル, settingsを保存 --
os.system(f"cp ./bert_restart_train.py {train_settings.output_path}")
os.system(f"cp ./bert_restart_train.sh {train_settings.output_path}")
train_settings.to_json(f"{train_settings.output_path}train_settings.json", indent=4)




# ====================================== #
#                                        #
#    --     Prepare for training   --    #
#                                        #
# ====================================== #
# load data + preprocess --
#train_df = pd.read_feather("input/corpus_label_tmp/corpus_labeled.feather")
#train_df["label"] = train_df["model_pred"]
# raw --
train_df = pd.read_csv(data_path+"train.csv")
train_df["clean_text"] = train_df["text"].map(lambda x: clean_text(x))
valid_df = pd.read_csv(data_path+"train.csv")
valid_df["clean_text"] = valid_df["text"].map(lambda x: clean_text(x))

# コーパスデータを0, 正規の学習データを1としたfoldを組めば関数そのまま使える --
train_df["kfold"] = -1
valid_df["kfold"] = 0
train_df = pd.concat([train_df, valid_df])

# define tokenizer --
tokenizer = define_tokenizer(settings.model_name)

# define log file --
log = open(train_settings.output_path + "/retrain.log", "w", buffering=1)
Write_log(log, f"use:{train_df.shape} ... with valid:{valid_df.shape}")
Write_log(log, f"   {train_df[label_name].value_counts()}")
Write_log(log, "***************** TRAINING ********************")




# ====================================== #
#                                        #
#    --          Training          --    #
#                                        #
# ====================================== #
fold = 0
Write_log(log, f"\n================== Hold-Out : (corpus) ---> (train_data) ==================")

# Create DataLoader --
train_loader, valid_loader = prepare_loaders(
    df=train_df,
    tokenizer=tokenizer,
    fold=fold,
    trn_batch_size=train_settings.train_batch_size,
    val_batch_size=settings.valid_batch_size,
    max_length=settings.max_length,
    num_classes=settings.num_classes,
    text_col="clean_text"
)
Debug_print(len(train_loader))
Debug_print(len(valid_loader))

# Model construct --
model = HateSpeechModel(
    model_name=settings.model_name,
    num_classes=settings.num_classes,
    custom_header=settings.model_custom_header,
    dropout=settings.dropout,
    )
if settings.mixout:
    model = replace_mixout(model)  # mixout --

# Define Optimizer and Scheduler --
optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
scheduler = fetch_scheduler(optimizer=optimizer, scheduler=settings.scheduler_name)

# load state --
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
Write_log(log, f"... ... Start from : epoch:{epoch}, loss:{loss}")

model.to(device)
model, history = run_training(
    model, train_loader, valid_loader, optimizer, scheduler, 
    settings.n_accumulate, device, settings.use_amp, 
    train_settings.epochs, fold, train_settings.output_path, log,
    save_checkpoint=True
)

del model, history, train_loader, valid_loader
_ = gc.collect()




# ====================================== #
#                                        #
#    --         Validate           --    #
#                                        #
# ====================================== #
model_paths = glob(f"{train_settings.output_path}*.pth"); model_paths.sort()

fold_f1 = []
fold_acc = []
print(f"{y_} ====== Hold-Out : Validation ======{sr_}")

#model_id = model_paths[fold].split("/")[3].split(".")[0].split("-")[0]
model_id = "model"

# Create DataLoader --
train_loader, valid_loader = prepare_loaders(
    df=train_df,
    tokenizer=tokenizer,
    fold=fold,
    trn_batch_size=settings.train_batch_size,
    val_batch_size=settings.valid_batch_size,
    max_length=settings.max_length,
    num_classes=settings.num_classes,
    text_col="clean_text"
)

valid = train_df[train_df.kfold == fold]
out = inference(
    settings.model_name, settings.num_classes,
    settings.model_custom_header, settings.dropout,
    model_paths[fold], valid_loader, device)

valid_preds = np.argmax(out, axis=1)

fold_f1.append(f1_score(valid[label_name].values, valid_preds))
fold_acc.append(accuracy_score(valid[label_name].values, valid_preds))

# log validatation result --
Write_log(log, "\n++++++++++++++++++++++++++++++++++++++++\n")
Write_log(log, f">> valid_metric : f1 = {np.mean(fold_f1):.4f} ... acc = {np.mean(fold_acc):.4f}")