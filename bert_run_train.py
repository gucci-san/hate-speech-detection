import pandas as pd
import torch
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from transformers import AutoTokenizer, AdamW

from colorama import Fore; r_=Fore.RED; sr_=Fore.RESET
from glob import glob
from config import *
from bert_utils import *

import argparse


# ====================================== #
#                                        #
#    -- Define Settings and Constants -- #
#                                        #
# ====================================== #
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("--num_classes", type=int, default=2)

parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--valid_batch_size", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=64)

parser.add_argument("--model_name", type=str, default=r"cl-tohoku/bert-base-japanese-whole-word-masking")
parser.add_argument("--max_length", type=int, default=76)
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--num_hidden_layers", type=int, default=24)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--scheduler_name", type=str, default="CosineAnnealingLR")
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--T_max", type=int, default=500)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--n_accumulate", type=int, default=1)

args, unknown = parser.parse_known_args()

settings = pd.Series(dtype=object)
# project settings --
settings["run_id"] = args.run_id
settings["num_classes"] = args.num_classes
settings["output_path"] = f"{output_root}{settings.run_id}/"
# training settings --
settings["epochs"] = args.epochs
settings["folds"] = args.folds
settings["train_batch_size"] = args.train_batch_size
settings["valid_batch_size"] = args.valid_batch_size
settings["test_batch_size"] = args.test_batch_size
# bert settings --
settings["model_name"] = args.model_name
settings["max_length"] = args.max_length
settings["hidden_size"] = args.hidden_size
settings["num_hidden_layers"] = args.num_hidden_layers
settings["dropout"] = args.dropout
# optimizer settings --
settings["learning_rate"] = args.learning_rate
settings["scheduler_name"] = args.scheduler_name
settings["min_lr"] = args.min_lr
settings["T_max"] = args.T_max
settings["weight_decay"] = args.weight_decay
settings["n_accumulate"] = args.n_accumulate

# run_idが重複したらlogが消えてしまうので、プログラムごと止めるようにする --
if not os.path.exists(settings.output_path):
    os.mkdir(settings.output_path)
else:
    assert False, (f"{r_}*** ... run_id {args.run_id} alreadly exists ... ***{sr_}")

os.system(f"cp ./*py {settings.output_path}")
settings.to_json(f"{settings.output_path}settings.json", indent=4)


# ====================================== #
#                                        #
#    --     Prepare for training   --    #
#                                        #
# ====================================== #

# load data --
train = pd.read_csv(data_path+"train.csv")
test = pd.read_csv(data_path+"test.csv")
df = pd.concat([train, test]).reset_index(drop=True)
train_shape = train.shape[0]
del train, test; _ = gc.collect()

# preprocess --
df["clean_text"] = df["text"].map(lambda x: clean_text(x))
train_df = df.loc[:train_shape-1, :]
test_df = df.loc[train_shape:, :]

# make folds --
skf = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=SEED)
split = skf.split(train_df, train_df[label_name])

for fold, (_, val_index) in enumerate(skf.split(X=train_df, y=train_df[label_name])):
    train_df.loc[val_index, "kfold"] = int(fold)
train_df["kfold"] = train_df["kfold"].astype(int)

# define tokenizer --
tokenizer = AutoTokenizer.from_pretrained(
    settings.model_name,
    mecab_kwargs={"mecab_dic":None, "mecab_option": f"-d {dic_neologd}"}
)

# define log file --
log = open(settings.output_path + "/train.log", "w", buffering=1)
Write_log(log, "***************** TRAINING ********************")


# ====================================== #
#                                        #
#    --          Training          --    #
#                                        #
# ====================================== #
for fold in range(0, settings.folds):
    
    #print(f"{y_} ====== Fold: {fold} ======{sr_}")
    Write_log(log, f"\n================== Fold: {fold} ==================")

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

    # Model construct --
    model = HateSpeechModel(model_name=settings.model_name, num_classes=settings.num_classes)
    model.to(device)

    # Define Optimizer and Scheduler --
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = fetch_scheduler(optimizer=optimizer, scheduler=settings.scheduler_name)

    model, history = run_training(
        model, train_loader, valid_loader, optimizer, scheduler, settings.n_accumulate, device, settings.epochs, fold, settings.output_path, log
    )

    del model, history, train_loader, valid_loader
    _ = gc.collect()



# ====================================== #
#                                        #
#    --         Validate           --    #
#                                        #
# ====================================== #
model_paths = glob(f"{settings.output_path}*.pth"); model_paths.sort()
model_paths

fold_f1 = []
fold_acc = []
for fold in range(0, settings.folds):
    print(f"{y_} ====== Fold: {fold} ======{sr_}")

    model_id = model_paths[fold].split("/")[3].split(".")[0].split("-")[0]
    
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
    out = inference(settings.model_name, settings.num_classes, model_paths[fold], valid_loader, device)

    for _class in range(0, settings.num_classes):
        valid[f"{model_id}_oof_class{_class}"] = out[:, _class]
        train_df.loc[valid.index.tolist(), f"{model_id}_oof_class_{_class}"] = valid[f"{model_id}_oof_class{_class}"]

    valid_preds = np.argmax(out, axis=1)

    fold_f1.append(f1_score(valid[label_name].values, valid_preds))
    fold_acc.append(accuracy_score(valid[label_name].values, valid_preds))

    train_df.loc[valid.index.tolist(), f"{model_id}_pred"] = valid_preds

# save oof --
train_df.reset_index(drop=False).to_feather(f"{settings.output_path}train_df.feather")
test_df.reset_index(drop=False).to_feather(f"{settings.output_path}test_df.feather")

# log validatation result --
Write_log(log, "\n++++++++++++++++++++++++++++++++++++++++\n")
Write_log(log, f">> mean_valid_metric : f1 = {np.mean(fold_f1):.4f} ... acc = {np.mean(fold_acc):.4f}")
Write_log(log, f">>  all_valid_metric : f1 = {f1_score(train_df.label, train_df.model_pred):.4f} ... acc = {accuracy_score(train_df.label, train_df.model_pred):.4f} ")

# experiment manage --
mean_valid_metric = np.mean(fold_f1)
all_valid_metric = f1_score(train_df.label, train_df.model_pred)
log_df = pd.DataFrame(settings).T
log_df["all_valid_metric"] = all_valid_metric
log_df["mean_valid_metric"] = mean_valid_metric
Write_exp_management(output_root, log_df)

