import pandas as pd
import os

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

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
args, unknown = parser.parse_known_args()

# settings, fine-tuningしたモデル, モデル作成時に前処理したtest_dfを読み込み -- 
output_path = f"{output_root}{args.run_id}/"
settings = pd.read_json(f"{output_path}settings.json", typ="series")
#model_paths = glob(f"{settings.output_path}*.pth"); model_paths.sort()
checkpoint_path = f"{settings.output_path}checkpoint-fold{args.model_fold}.pth"


print(checkpoint_path)
exit()

test_df = pd.read_feather(f"{settings.output_path}test_df.feather")



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
os.system(f"cp ./*sh {settings.output_path}")
settings.to_json(f"{settings.output_path}settings.json", indent=4)




# ====================================== #
#                                        #
#    --     Prepare for training   --    #
#                                        #
# ====================================== #
# load data --
df, train_shape = prepare_dataframe(train_data=settings.train_data)

# preprocess --
train_df, test_df = preprocess_text(df, train_shape, settings.model_name)

# make folds --
skf = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=SEED)
split = skf.split(train_df, train_df[label_name])
train_df = make_folds(split, train_df, label_name=label_name)

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
    model = HateSpeechModel(
        model_name=settings.model_name,
        num_classes=settings.num_classes,
        custom_header=settings.model_custom_header,
        dropout=settings.dropout,
        )
    if settings.mixout:
        model = replace_mixout(model)  # mixout --
    model.to(device)

    # Define Optimizer and Scheduler --
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = fetch_scheduler(optimizer=optimizer, scheduler=settings.scheduler_name)

    model, history = run_training(
        model, train_loader, valid_loader, 
        optimizer, scheduler, settings.n_accumulate, device, settings.use_amp, 
        settings.epochs, fold, settings.output_path, log
    )

    del model, history, train_loader, valid_loader
    _ = gc.collect()




# ====================================== #
#                                        #
#    --         Validate           --    #
#                                        #
# ====================================== #
model_paths = glob(f"{settings.output_path}*.pth"); model_paths.sort()

fold_f1 = []
fold_acc = []
for fold in range(0, settings.folds):
    print(f"{y_} ====== Fold: {fold} ======{sr_}")

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

Write_exp_management(experiment_root, log_df)
