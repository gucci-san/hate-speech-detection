import os, gc, random, time, copy, math
from tracemalloc import start
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

import re
import demoji
import neologdn

from pyknp import Juman, BList, KNP

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid juman warnings --

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.cuda.amp import autocast
from cosine_lr import CosineLRScheduler

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    T5Tokenizer,
    BertTokenizer,
)

from tqdm import tqdm
from collections import defaultdict
from colorama import Fore

b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN
sr_ = Fore.RESET
from config import *


def Debug_print(x):
    print(f"{y_}{x}{sr_}")
    return None


def Write_exp_management(exp_manage_path, log_df):
    if not os.path.exists(f"{exp_manage_path}experiment_log.csv"):
        log_df.to_csv(f"{exp_manage_path}experiment_log.csv", index=False)
    else:
        log_df.to_csv(
            f"{exp_manage_path}experiment_log.csv", index=False, mode="a", header=None
        )


def Write_log(logFile, text, isPrint=True):
    if logFile is None:
        print(text)
        return None
    else:
        if isPrint:
            print(text)
        logFile.write(text)
        logFile.write("\n")
        return None


def seed_everything(seed=42):
    print(f"\n****** random-seed fixed : {seed} ******\n\n")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def nchars(s, n):
    """
    ????????? s ????????????????????? n ?????????????????????????????????????????????????????????
    https://qiita.com/norioc/items/324d9210ae2db29d08e6
    """
    assert n > 0
    reg = re.compile("(.)\\1{%d,}" % (n - 1))  # ????????????????????? n ????????????????????????
    while True:
        m = reg.search(s)
        if not m:
            break
        yield m.group(0)
        s = s[m.end() :]


def clean_text(text: str) -> str:
    """
    ????????????????????????????????????
    ??????????????????????????????????????????????????????????????????????????????????????????
    ??????????????????????????????????????? --

    <??????>
    * https://note.com/narudesu/n/na35de30a583a

    """
    # ????????????????????????????????????????????? --
    text = text.replace("(????`)y???", "")
    text = text.replace("( ????`)y???", "")
    text = text.replace("<(_ _)>", "")
    text = text.replace("m(_ _)m", "")

    # ????????????????????? --
    text = text.replace("\n", "").replace("\r", "")

    # ??????-?????????????????? --
    text = neologdn.normalize(text)

    # re.sub?????????????????????????????????????????? --
    # ## 3???????????? --
    text = text.replace("?????????", "")
    text = text.replace("...", "")
    text = text.replace("???", "")

    # ## ^^ --
    text = text.replace("^", "")

    # URL?????? --
    text = re.sub(r"http?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)
    text = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", text)

    # ?????????????????? --
    text = demoji.replace(string=text, repl="")

    # ??????????????????(??????, ??????) --
    text = re.sub(
        r'[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~????????????????????????????????????????????????????????????????????????????????????]',
        "",
        text,
    )
    text = re.sub(
        "[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", "", text
    )

    return text


def prepare_dataframe(train_data):
    if train_data == "raw":
        train = pd.read_csv(data_path + "train.csv")
        test = pd.read_csv(data_path + "test.csv")

    elif train_data == "raw+test_pseudo":
        train = pd.read_csv(data_path + "train.csv")
        test_pseudo = pd.read_feather(
            f"{input_root}pseudo_label_base/test_pseudo_labeled.feather"
        )
        test_pseudo[label_name] = 1  # hard label --
        train = pd.concat([train, test_pseudo]).reset_index(drop=True)
        test = pd.read_csv(data_path + "test.csv")

    elif train_data == "raw+corpus_label_debug":
        train = pd.read_csv(data_path + "train.csv")
        corpus_labeled = pd.read_feather(
            "input/corpus_label_roberta_large_cat4/corpus_labeled.feather"
        ).rename(columns={"clean_text": "text", "model_pred": label_name})
        corpus_labeled = corpus_labeled.drop(
            ["model_oof_class_0", "model_oof_class_1"], axis=1
        )
        train = pd.concat([train, corpus_labeled])
        test = pd.read_csv(data_path + "test.csv")

    elif train_data == "raw_original_text":
        train = pd.read_feather(
            f"{input_root}dataset_with_original_text/train_with_original_text.feather"
        )
        test = pd.read_feather(
            f"{input_root}dataset_with_original_text/test_with_original_text.feather"
        )

    else:
        Debug_print(f"NOT implemented : train_data=={train_data}")
        assert False

    df = pd.concat([train, test]).reset_index(drop=True)
    train_shape = train.shape[0]
    return df, train_shape


def juman_parse(text):
    words = ""
    jumanpp = Juman()
    result = jumanpp.analysis(text)
    for mrp in result.mrph_list():
        words += mrp.midasi + " "

    return words[:-1]  # last " " omit by [:-1] --


def original_text_preprocess(text, use_juman=False):
    """
    ????????????2ch????????????????????????bert???????????????????????????????????????
    tokenizer.encode_plus????????????????????????????????????/?????????special_token?????????????????????
    """
    text = text.replace(" __BR__ ", "???")
    text = text.replace("__BR__", "???")  # __BR__??????????????????1????????????????????? --
    text = clean_text(text)
    if use_juman:
        text = juman_parse(text)
        text = text.replace("\\t", "[SEP]")
    else:
        text = text.replace("\t", "[SEP]")
    return text


def preprocess_text(df, train_shape, model_name, train_data="raw"):
    if train_data == "raw_original_text":
        Debug_print("... ... Process -> raw_original_text")
        if model_name in ["nlp-waseda/roberta-large-japanese-seq512"]:
            df["clean_text"] = df["original_text"].parallel_map(
                lambda x: original_text_preprocess(x, use_juman=True)
            )
        else:
            df["clean_text"] = df["original_text"].parallel_map(
                lambda x: original_text_preprocess(x)
            )
    else:
        df["clean_text"] = df["text"].map(lambda x: clean_text(x))
        if model_name in ["nlp-waseda/roberta-large-japanese-seq512"]:
            df["clean_text"] = df["clean_text"].parallel_map(lambda x: juman_parse(x))

    train_df = df.loc[: train_shape - 1, :]
    test_df = df.loc[train_shape:, :]

    return train_df, test_df


def make_folds(split, train_df, label_name, fold_colname="kfold"):
    for fold, (_, val_index) in enumerate(split):
        train_df.loc[val_index, fold_colname] = int(fold)
    train_df[fold_colname] = train_df[fold_colname].astype(int)

    return train_df


def define_tokenizer(model_name: str):
    if model_name in [
        "rinna/japanese-roberta-base",
        "rinna/japanese-gpt-1b",
        "rinna/japanese-gpt2-medium",
    ]:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.do_lower_case = True
    elif model_name in ["ganchengguang/Roformer-base-japanese"]:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_name in [
        "nlp-waseda/roberta-large-japanese-seq512",
        "xlm-roberta-large",
    ]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            mecab_kwargs={"mecab_dic": None, "mecab_option": f"-d {dic_neologd}"},
        )
    return tokenizer


class HateSpeechDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_length,
        num_classes,
        text_col="text",
        isTrain=True,
        label_name=label_name,
    ):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df[text_col].values
        self.num_classes = num_classes
        if isTrain:
            self.target = df[label_name].values
        else:
            self.target = np.zeros(df.shape[0])
        self.isTrain = isTrain

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        text = self.text[index]
        inputs_text = self.tokenizer.encode_plus(
            text,
            truncation=True,  # ????????????????????? --
            add_special_tokens=True,  # [CLS][SEP]??????????????? --
            max_length=self.max_len,
            padding="max_length",
        )

        if self.isTrain:
            target = int(self.target[index])
            onehot_t = np.zeros(self.num_classes, dtype=np.float32)
            onehot_t[target] = 1.0
            return {
                "input_ids": torch.tensor(inputs_text["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    inputs_text["attention_mask"], dtype=torch.long
                ),
                "target": torch.tensor(onehot_t, dtype=torch.float),
            }

        else:
            return {
                "input_ids": torch.tensor(inputs_text["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    inputs_text["attention_mask"], dtype=torch.long
                ),
            }


class BertClassificationMaxPoolingHeader(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(BertClassificationMaxPoolingHeader, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # max pooling --
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, base_output):
        out = base_output["hidden_states"][-1].max(axis=1)[0]
        out = self.fc(out)
        return out


class BertClassificationConvolutionHeader(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(BertClassificationConvolutionHeader, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # conv1d --
        self.cnn1 = nn.Conv1d(self.hidden_size, 256, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(256, self.num_classes, kernel_size=2, padding=1)

    def forward(self, base_output):
        last_hidden_state = base_output["hidden_states"][-1].permute(0, 2, 1)
        cnn_embeddings = F.relu(self.cnn1(last_hidden_state))
        cnn_embeddings = self.cnn2(cnn_embeddings)
        outputs = cnn_embeddings.max(axis=2)[0]
        return outputs


class BertClassificationLSTMHeader(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(BertClassificationLSTMHeader, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # lstm --
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, base_output):
        last_hidden_state = base_output["hidden_states"][-1]
        out = self.lstm(last_hidden_state, None)[0]
        out = out[:, -1, :]  # lstm????????????????????????????????????, [batch_size, hidden_size] --
        outputs = self.fc(out)
        return outputs


class BertClassificationConcatenateHeader(nn.Module):
    def __init__(self, hidden_size, num_classes, use_layer_num=4):
        super(BertClassificationConcatenateHeader, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_layer_num = use_layer_num

        # concatenate --
        self.fc = nn.Linear(self.hidden_size * self.use_layer_num, self.num_classes)

    def forward(self, base_output):
        out = torch.cat(
            [
                base_output["hidden_states"][-1 * i][:, 0, :]
                for i in range(1, self.use_layer_num + 1)
            ],
            dim=1,
        )
        outputs = self.fc(out)
        return outputs


class BertClassificationMozafariHeader(nn.Module):
    """
    Mozafari et al., 2019???(d)???????????????Module
    AutoModel???output["hidden_states"]?????????????????????
    ????????????????????????(https://github.com/ZeroxTM/BERT-CNN-Fine-Tuning-For-Hate-Speech-Detection-in-Online-Social-Media/blob/main/Model.py)
    ?????????????????????Embedding layer?????????????????????????????????????????????????????????????????????????????????????????????????????????????????? --

    BatchNorm2d???????????????????????????
    https://qiita.com/cfiken/items/b477c7878828ebdb0387

    """

    def __init__(self, hidden_size, hidden_layer_num, max_length, num_classes):
        super(BertClassificationMozafariHeader, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.max_length = max_length
        self.num_classes = num_classes

        self.conv = nn.Conv2d(
            in_channels=self.hidden_layer_num,
            out_channels=self.hidden_layer_num,
            kernel_size=(3, self.hidden_size),
            padding=1,
        )  # [batch, hidden_layer, max_length, hidden_size] -> [batch, hidden_layer, max_length, 1]
        self.batchnorm = nn.BatchNorm2d(num_features=self.hidden_layer_num)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()  # [batch, a, b, c...] -> [batch, a*b*c...]
        self.fc = nn.Linear(
            self.hidden_layer_num * (self.max_length - 2), self.num_classes
        )

    def forward(self, base_output):
        out = torch.transpose(
            torch.cat(
                tuple([t.unsqueeze(0) for t in base_output["hidden_states"]][1:]), 0
            ),
            0,
            1,
        )  # -> [batch, hidden_layer_num, max_length, hidden_size] --
        out = self.relu(self.batchnorm(self.conv(out)))
        out = self.pool(out)  # "...the maximum value for each transformer encoder..."
        outputs = self.fc(self.flatten(out))

        return outputs


def torch_init_params_by_name(model, name):
    """name?????????named_parameter????????????????????????"""
    init_params = [
        (param_name, params)
        for (param_name, params) in model.named_parameters()
        if name in param_name
    ]
    for param in init_params:
        print(f"{g_}... {param[0]} initialized ... {sr_}")
        nn.init.normal_(param[1], mean=0, std=0.02)


def torch_freeze_params_by_name(model, name):
    """name?????????named_parameter???freeze(required_grad=False)????????????"""
    freeze_params = [
        (param_name, params)
        for (param_name, params) in model.named_parameters()
        if name in param_name
    ]
    for param in freeze_params:
        print(f"{b_}... {param[0]} freezed ... {sr_}")
        param[1].requires_grad = False


class HateSpeechModel(nn.Module):
    def __init__(
        self,
        model_name,
        max_length,
        num_classes,
        custom_header="max_pooling",
        dropout=0.2,
        n_msd=None,
    ):
        super(HateSpeechModel, self).__init__()
        self.cfg = AutoConfig.from_pretrained(model_name)
        self.max_length = max_length
        self.num_classes = num_classes
        self.l1 = AutoModel.from_pretrained(
            model_name, output_attentions=True, output_hidden_states=True
        )

        if custom_header == "max_pooling":
            self.l2 = BertClassificationMaxPoolingHeader(
                self.cfg.hidden_size, self.num_classes
            )
        elif custom_header == "conv":
            self.l2 = BertClassificationConvolutionHeader(
                self.cfg.hidden_size, self.num_classes
            )
        elif custom_header == "lstm":
            self.l2 = BertClassificationLSTMHeader(
                self.cfg.hidden_size, self.num_classes
            )
        elif custom_header in ["concatenate", "concatenate-4"]:
            self.l2 = BertClassificationConcatenateHeader(
                self.cfg.hidden_size, self.num_classes, use_layer_num=4
            )
        elif custom_header == "mozafari":
            self.l2 = BertClassificationMozafariHeader(
                self.cfg.hidden_size,
                self.cfg.num_hidden_layers,
                self.max_length,
                self.num_classes,
            )
        else:
            assert (
                False
            ), f"custom header == {custom_header} not defined (or implemented)"

        self.l3 = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        out = self.l1(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        out = self.l2(out)
        out = self.l3(out)
        return out


def prepare_loaders(
    df,
    fold,
    tokenizer,
    trn_batch_size,
    val_batch_size,
    max_length,
    num_classes,
    seed,
    text_col="text",
    label_name=label_name,
):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    train_dataset = HateSpeechDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        num_classes=num_classes,
        text_col=text_col,
        label_name=label_name,
    )
    valid_dataset = HateSpeechDataset(
        valid_df,
        tokenizer=tokenizer,
        max_length=max_length,
        num_classes=num_classes,
        text_col=text_col,
        label_name=label_name,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=trn_batch_size,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, valid_loader


def criterion(outputs, targets):
    # loss_f = nn.BCELoss()
    # loss_f = nn.BCEWithLogitsLoss()
    # loss_f = nn.CrossEntropyLoss()
    loss_f = nn.NLLLoss()

    # return loss_f(outputs, targets)
    return loss_f(outputs, torch.argmax(targets, axis=1))


def fetch_scheduler(scheduler, optimizer, T_max=500, eta_min=1e-7):
    if scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif scheduler == "CosineAnnealingWithWarmUp":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=T_max,
            lr_min=eta_min,
            warmup_t=int(0.1 * T_max),
            warmup_lr_init=eta_min,
            warmup_prefix=True,
        )
    else:
        print(f"{y_}*** Scheduler : None ***{sr_}")
        scheduler = None
    return scheduler


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    dataloader,
    device,
    scaler,
    use_amp,
    epoch,
    n_accumulate,
):
    model.train()

    dataset_size = 0
    running_loss = 0.0
    train_loss_list = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.float)

        batch_size = input_ids.size(0)

        with autocast(enabled=use_amp):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            loss = loss / np.float(n_accumulate)
            scaler.scale(loss).backward()

            if (step + 1) % n_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None:
                    if "torch" in str(scheduler.__class__):
                        scheduler.step()
                    else:
                        scheduler.step(step + 1)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size
            train_loss_list.append(epoch_loss)

            bar.set_postfix(
                Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
            )

    # clean GPU memory --
    del input_ids, attention_mask, targets
    gc.collect()
    torch.cuda.empty_cache()

    return train_loss_list


@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    valid_loss_list = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.float)

        batch_size = input_ids.size(0)

        outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, targets)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        valid_loss_list.append(epoch_loss)

        bar.set_postfix(
            Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
        )

    # clean GPU memory --
    del input_ids, attention_mask, targets
    gc.collect()
    torch.cuda.empty_cache()

    return valid_loss_list


def run_training(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    n_accumulate,
    device,
    scaler,
    use_amp,
    num_epochs,
    fold,
    output_path,
    log=None,
    save_checkpoint=False,
    load_checkpoint=None,
):

    if torch.cuda.is_available():
        Write_log(log, f"[INFO] Using GPU : {torch.cuda.get_device_name()}\n")

    # ------------------------------------------------------------
    start_time = time.time()

    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        random.setstate(checkpoint["random"])
        np.random.set_state(checkpoint["np_random"])
        torch.set_rng_state(checkpoint["torch"])
        torch.random.set_rng_state(checkpoint["torch_random"])
        torch.cuda.set_rng_state(checkpoint["cuda_random"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_all"])
        start_epoch, loss, best_epoch_loss = (
            checkpoint["epoch"],
            checkpoint["loss"],
            checkpoint["best_epoch_loss"],
        )
        start_epoch += 1  # ??????????????????epoch==1???????????????2?????????????????????????????? --
    else:
        start_epoch, loss, best_epoch_loss = 1, None, np.inf

    best_model_wts = copy.deepcopy(model.state_dict())
    history = defaultdict(list)
    for epoch in range(start_epoch, num_epochs + start_epoch):
        gc.collect()

        train_epoch_loss_list = train_one_epoch(
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
            n_accumulate=n_accumulate,
        )

        valid_epoch_loss_list = valid_one_epoch(
            model,
            optimizer,
            dataloader=valid_loader,
            device=device,
            epoch=epoch,
        )
        valid_epoch_loss = valid_epoch_loss_list[-1]

        history["Train Loss"].append(train_epoch_loss_list)
        history["Valid Loss"].append(valid_epoch_loss_list)

        if valid_epoch_loss <= best_epoch_loss:
            Write_log(
                log,
                f"epoch{epoch}: Valid Loss Improved : {best_epoch_loss:.6f} ---> {valid_epoch_loss:.6f}",
            )

            best_epoch_loss = valid_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            if save_checkpoint:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": valid_epoch_loss,
                        "best_epoch_loss": best_epoch_loss,
                        "scaler_state_dict": scaler.state_dict(),
                        "random": random.getstate(),
                        "np_random": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "torch_random": torch.random.get_rng_state(),
                        "cuda_random": torch.cuda.get_rng_state(),
                        "cuda_random_all": torch.cuda.get_rng_state_all(),
                    },
                    f"{output_path}checkpoint-fold{fold}.pth",
                )
                Write_log(log, f"Checkpoint Saved")
            else:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                    },
                    f"{output_path}model-fold{fold}.pth",
                )
                Write_log(log, f"Model Saved")
            print()

    end_time = time.time()
    # --------------------------------------------------------------

    time_elapsed = end_time - start_time
    Write_log(
        log,
        "Training Complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        ),
    )
    Write_log(log, "Best Loss: {:.4f}".format(best_epoch_loss))

    model.load_state_dict(best_model_wts)
    history["Train Loss"] = sum(history["Train Loss"], [])
    history["Valid Loss"] = sum(history["Valid Loss"], [])

    return model, history


@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()  # model???train????????????to(device)????????????????????? --

    preds = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)

        outputs = model(input_ids, attention_mask).cpu().detach()
        preds.append(outputs.numpy())

    preds = np.concatenate(preds)
    gc.collect()

    return preds


def inference(
    model_name,
    max_length,
    num_classes,
    custom_header,
    dropout,
    model_paths,
    dataloader,
    device,
):
    final_preds = []

    for i, path in enumerate([model_paths]):
        model = HateSpeechModel(
            model_name=model_name,
            max_length=max_length,
            num_classes=num_classes,
            custom_header=custom_header,
            dropout=dropout,
        )
        model.to(device)
        checkpoint = torch.load(model_paths)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"{y_}Getting predictions for model : {path} {sr_}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds


def state_dict_divider(state_dict, div=2, target=["weight", "bias"]):
    """state_dict??????????????????????????????"""
    for key in state_dict.keys():
        if key.split(".")[-1] in target:
            state_dict[key] /= float(div)

    return state_dict


def model_parameter_ensembler(model_paths: list, target=["weight", "bias"]):
    """.pth???????????????????????????????????????????????????"""

    for i, path in enumerate(model_paths):
        if i == 0:
            avg_state_dict = torch.load(path)["model_state_dict"]
            print(f"... load {path} to set average-root ... ... ")
        else:
            print(f"... ... key add : {path}")
            for key in tqdm(avg_state_dict.keys(), total=len(avg_state_dict.keys())):
                if key.split(".")[-1] in target:
                    state_dict = torch.load(path)["model_state_dict"]
                    avg_state_dict[key] += state_dict[key]
    avg_state_dict = state_dict_divider(
        avg_state_dict, div=len(model_paths), target=target
    )
    return avg_state_dict
