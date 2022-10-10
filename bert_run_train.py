import pandas as pd
import torch
import os

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from transformers import AutoTokenizer, T5Tokenizer, BertTokenizer, AdamW

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
parser.add_argument("--train_data", type=str, default="raw")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--valid_batch_size", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--model_name", type=str, default=r"cl-tohoku/bert-base-japanese-whole-word-masking")
parser.add_argument("--model_custom_header", type=str, default="max_pooling")
parser.add_argument("--max_length", type=int, default=76)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--scheduler_name", type=str, default="CosineAnnealingLR")
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--T_max", type=int, default=500)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--n_accumulate", type=int, default=1)
parser.add_argument("--remark", type=str, default=None)
parser.add_argument("--trial", type=bool, default=False)
args, unknown = parser.parse_known_args()


settings = pd.Series(dtype=object)
# project settings --
settings["run_id"] = args.run_id
settings["num_classes"] = args.num_classes
settings["output_path"] = f"{output_root}{settings.run_id}/"
# training settings --
settings["train_data"] = args.train_data
settings["epochs"] = args.epochs
settings["folds"] = args.folds
settings["train_batch_size"] = args.train_batch_size
settings["valid_batch_size"] = args.valid_batch_size
settings["test_batch_size"] = args.test_batch_size
# bert settings --
settings["model_name"] = args.model_name
settings["model_custom_header"] = args.model_custom_header
settings["max_length"] = args.max_length
settings["dropout"] = args.dropout
# optimizer settings --
settings["learning_rate"] = args.learning_rate
settings["scheduler_name"] = args.scheduler_name
settings["min_lr"] = args.min_lr
settings["T_max"] = args.T_max
settings["weight_decay"] = args.weight_decay
settings["n_accumulate"] = args.n_accumulate
# experiment remarks --
settings["remark"] = args.remark
settings["seed"] = SEED


# run_idが重複したらlogが消えてしまうので、プログラムごと止めるようにする --
if not os.path.exists(settings.output_path):
    os.mkdir(settings.output_path)
else:
    if args.trial:
        assert True
    else:
        assert False, (f"{r_}*** ... run_id {args.run_id} alreadly exists ... ***{sr_}")


# 計算時点でのpyファイル, settingsを保存 --
os.system(f"cp ./*py {settings.output_path}")
settings.to_json(f"{settings.output_path}settings.json", indent=4)




# ====================================== #
#                                        #
#    --     Prepare for training   --    #
#                                        #
# ====================================== #
# load data --
# ## -> df = pd.concat([train, ..., test]);
# ## -> train_shape = train.shape[0]
if settings.train_data == "raw":
    train = pd.read_csv(data_path+"train.csv")
    test = pd.read_csv(data_path+"test.csv")
    df = pd.concat([train, test]).reset_index(drop=True)
    train_shape = train.shape[0]
    del train, test; _ = gc.collect()

elif settings.train_data == "raw+test_pseudo":
    train = pd.read_csv(data_path+"train.csv")
    test_pseudo = pd.read_feather(f"{input_root}pseudo_label_base/test_pseudo_labeled.feather")
    test_pseudo[label_name] = 1 # hard label --
    train = pd.concat([train, test_pseudo]).reset_index(drop=True)
    test = pd.read_csv(data_path+"test.csv")
    df = pd.concat([train, test]).reset_index(drop=True)
    train_shape = train.shape[0]
    del train, test_pseudo, test; _ = gc.collect()

# preprocess --
df["clean_text"] = df["text"].map(lambda x: clean_text(x))
if settings.model_name in ["nlp-waseda/roberta-large-japanese-seq512"]:
    df["clean_text"] = df["clean_text"].parallel_map(lambda x: juman_parse(x))
train_df = df.loc[:train_shape-1, :]
test_df = df.loc[train_shape:, :]

# make folds --
skf = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=SEED)
split = skf.split(train_df, train_df[label_name])
for fold, (_, val_index) in enumerate(skf.split(X=train_df, y=train_df[label_name])):
    train_df.loc[val_index, "kfold"] = int(fold)
train_df["kfold"] = train_df["kfold"].astype(int)

# define tokenizer --
tokenizer = define_tokenizer(settings.model_name)

# define log file --
log = open(settings.output_path + "/train.log", "w", buffering=1)
Write_log(log, f"train:{train_df.shape}, test:{test_df.shape}\n")
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
    model = HateSpeechModel(model_name=settings.model_name, num_classes=settings.num_classes, custom_header=settings.model_custom_header)
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
    out = inference(settings.model_name, settings.num_classes, settings.model_custom_header, model_paths[fold], valid_loader, device)

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
log_df = pd.DataFrame()
log_df["Single_Public_LB"] = [None]  # VSCodeでLBみて手書きする想定 --
log_df["all_valid_metric"] = [np.round(all_valid_metric, 6)]
log_df["mean_valid_metric"] = [np.round(mean_valid_metric, 6)]
log_df = pd.concat([log_df, pd.DataFrame(settings).T], axis=1)

Write_exp_management(output_root, log_df)
