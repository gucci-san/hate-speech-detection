from faulthandler import is_enabled
import os, gc, random, time, copy
import warnings; warnings.simplefilter("ignore")
import numpy as np

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
from torch.cuda.amp import GradScaler, autocast

from transformers import (
    AutoModel, RobertaForMaskedLM, RoFormerModel,
    AutoTokenizer, T5Tokenizer, BertTokenizer,
    AutoModelForMaskedLM, AutoModelForCausalLM,
)

from tqdm import tqdm
from collections import defaultdict
from colorama import Fore
b_ = Fore.BLUE; y_ = Fore.YELLOW; g_ = Fore.GREEN; sr_ = Fore.RESET
from config import *


def Debug_print(x):
    print(f"{y_}{x}{sr_}")
    return None


def Write_exp_management(exp_manage_path, log_df):
    if not os.path.exists(f"{exp_manage_path}experiment_log.csv"):
        log_df.to_csv(f"{exp_manage_path}experiment_log.csv", index=False)
    else:
        log_df.to_csv(f"{exp_manage_path}experiment_log.csv", index=False, mode="a", header=None)


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
    print(f"\n****** SEED fixed : {seed} ******\n\n")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(SEED)


def clean_text(text: str) -> str:
    """
    日本語から記号とかを削除
    タスク・与えられたデータによって適宜見ながらやるしかなさそう
    絵文字が鬼門という印象あり --

    <参考>
    * https://note.com/narudesu/n/na35de30a583a
    
    """
    # 改行コード削除 --
    text = text.replace("\n", "").replace("\r", "")

    # 半角-全角の正規化 --
    text = neologdn.normalize(text)

    # URL削除 --
    text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)    

    # 絵文字の削除 --
    text = demoji.replace(string=text, repl="")

    # 記号系の削除(半角, 全角) --
    text = re.sub(r'[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]', '', text)
    text = re.sub("[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", '', text)

    return text


def juman_parse(text):
    words = ""
    jumanpp = Juman()
    result = jumanpp.analysis(text)
    for mrp in result.mrph_list():
        words += (mrp.midasi + " ")
    return words[:-1]  # last " " omit by [:-1] -- 


def define_tokenizer(model_name: str):
    if model_name in ["rinna/japanese-roberta-base", "rinna/japanese-gpt-1b", "rinna/japanese-gpt2-medium"]:
        tokenizer = T5Tokenizer.from_pretrained(
            model_name
        )
        tokenizer.do_lower_case = True
    elif model_name in ["ganchengguang/Roformer-base-japanese"]:
        tokenizer = BertTokenizer.from_pretrained(
            model_name
        )
    elif model_name in ["nlp-waseda/roberta-large-japanese-seq512"]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            mecab_kwargs={"mecab_dic":None, "mecab_option": f"-d {dic_neologd}"}
        )
    return tokenizer


class HateSpeechDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, num_classes, text_col="text", isTrain=True, label_name=label_name):
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
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length"
        )
        
        if self.isTrain:
            target = int(self.target[index])
            onehot_t = np.zeros(self.num_classes, dtype=np.float32)
            onehot_t[target] = 1.0
            return {
                "input_ids": torch.tensor(inputs_text["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(inputs_text["attention_mask"], dtype=torch.long),
                "target": torch.tensor(onehot_t, dtype=torch.float)
            }
        
        else:
            return {
                "input_ids": torch.tensor(inputs_text["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(inputs_text["attention_mask"], dtype=torch.long),
            }


class HateSpeechModel(nn.Module):
    def __init__(self, model_name, num_classes, custom_header=None):
        super(HateSpeechModel, self).__init__()
        if model_name in ["rinna/japanese-roberta-base"]:
            self.model = RobertaForMaskedLM.from_pretrained(
                model_name,
                output_attentions=True, output_hidden_states=True,
                )
            self.hidden_size = 768
        elif model_name in ["ganchengguang/Roformer-base-japanese"]:
            self.model = RoFormerModel.from_pretrained(
                model_name,
                output_attentions=True, output_hidden_states=True,
                )
            self.hidden_size = 768
        elif model_name in ["cl-tohoku/bert-large-japanese"]:
            self.model = AutoModel.from_pretrained(
                model_name,
                output_attentions=True, output_hidden_states=True,
                )
            self.hidden_size = 1024
        elif model_name in ["nlp-waseda/roberta-large-japanese-seq512"]:
            self.model = AutoModel.from_pretrained(
                model_name,
                output_attentions=True, output_hidden_states=True,
                )
            self.hidden_size = 1024
        elif model_name in ["rinna/japanese-gpt-1b"]:
            self.model = AutoModel.from_pretrained(
                model_name,
                output_attentions=True, output_hidden_states=True
            )
            self.hidden_size = 2048
        elif model_name in ["rinna/japanese-gpt2-medium"]:
            #self.model = AutoModelForCausalLM.from_pretrained(
            self.model = AutoModel.from_pretrained(
                model_name,
                output_attentions=True, output_hidden_states=True,
                )
            self.hidden_size = 1024
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                output_attentions=True, output_hidden_states=True,
                )
            self.hidden_size = 768

        self.model_name = model_name
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.hidden_size, num_classes)
        self.softmax = nn.Softmax()

        if custom_header == "conv":
            self.cnn1 = nn.Conv1d(self.hidden_size, 256, kernel_size=2, padding=1)
            self.cnn2 = nn.Conv1d(256, num_classes, kernel_size=2, padding=1)
        elif custom_header == "lstm":
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        elif custom_header == "concatenate-4":
            self.fc_4 = nn.Linear(self.hidden_size*4, num_classes)

        self.custom_header = custom_header
        print(f"{y_}{self.custom_header}{sr_}")

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # 最終的に[batch_size, hidden_size]になるようにcustom_headerを作っていく --
        # https://www.ai-shift.co.jp/techblog/2145 --
        if self.custom_header == "max_pooling":
            out = out["hidden_states"][-1].max(axis=1)[0]  # last_hidden_state + max_pooling --
            out = self.dropout(out)
            outputs = self.fc(out)
            #outputs = self.softmax(outputs)

        elif self.custom_header == "conv":
            last_hidden_state = out["hidden_states"][-1].permute(0, 2, 1)
            cnn_embeddings = F.relu(self.cnn1(last_hidden_state))
            cnn_embeddings = self.cnn2(cnn_embeddings)
            outputs = cnn_embeddings.max(axis=2)[0]
            #outputs = self.softmax(outputs)

        elif self.custom_header == "lstm":
            last_hidden_state = out["hidden_states"][-1]
            out = self.lstm(last_hidden_state, None)[0]
            out = out[:, -1, :]  # lstmの時間方向の最終層を抜く, [batch_size, hidden_size] --
            outputs = self.fc(out)
            #outputs = self.softmax(outputs)

        elif self.custom_header == "concatenate-4":
            out = torch.cat([out["hidden_states"][-1*i][:, 0, :] for i in range(1, 4+1)], dim=1)
            outputs = self.fc_4(out)
            #outputs = self.softmax(outputs)

        return outputs


def prepare_loaders(df, fold, tokenizer, trn_batch_size, val_batch_size, max_length, num_classes, text_col="text", label_name=label_name):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = HateSpeechDataset(train_df, tokenizer=tokenizer, max_length=max_length, num_classes=num_classes, text_col=text_col, label_name=label_name)
    valid_dataset = HateSpeechDataset(valid_df, tokenizer=tokenizer, max_length=max_length, num_classes=num_classes, text_col=text_col, label_name=label_name)

    train_loader = DataLoader(
        train_dataset, batch_size=trn_batch_size, num_workers=2, shuffle=True, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=val_batch_size, num_workers=2, shuffle=False, pin_memory=True
    )
    return train_loader, valid_loader


def criterion(outputs, targets):
    #loss_f = nn.BCELoss()
    loss_f = nn.BCEWithLogitsLoss()
    return loss_f(outputs, targets)


def fetch_scheduler(scheduler, optimizer, T_max=500, eta_min=1e-7):
    if scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    else:
        print(f"*** *** NOT implemented *** *** ")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    return scheduler


def train_one_epoch(model, optimizer, scheduler, dataloader, device, use_amp, epoch, n_accumulate):
    model.train()

    scaler = GradScaler(enabled=use_amp)    

    dataset_size = 0
    running_loss = 0.0

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
            #loss.backward()

            if (step+1) % n_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

#        if (step+1) % n_accumulate == 0:
#            optimizer.step()
#            optimizer.zero_grad()
#
#            if scheduler is not None:
#                scheduler.step()

            running_loss += (loss.item()*batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"])
    
    gc.collect()
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.float)
        
        batch_size = input_ids.size(0)

        outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, targets)

        running_loss += (loss.item()*batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"])
    
    gc.collect()
    return epoch_loss


def run_training(model, train_loader, valid_loader, optimizer, scheduler, n_accumulate, device, use_amp, num_epochs, fold, output_path, log=None):

    if torch.cuda.is_available():
        Write_log(log, f"[INFO] Using GPU : {torch.cuda.get_device_name()}\n")

    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs+1):
        gc.collect()

        train_epoch_loss = train_one_epoch(
            model, optimizer, scheduler,
            dataloader=train_loader,
            device=device, use_amp=use_amp, epoch=epoch,
            n_accumulate=n_accumulate
        )

        valid_epoch_loss = valid_one_epoch(
            model, optimizer, 
            dataloader=valid_loader,
            device=device, epoch=epoch,
        )

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(valid_epoch_loss)

        if valid_epoch_loss <= best_epoch_loss:
            Write_log(log, f"Valid Loss Improved : {best_epoch_loss:.6f} ---> {valid_epoch_loss:.6f}")
            
            best_epoch_loss = valid_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # 提出ファイルを圧縮前合計25GBにしないといけないので、いらないもの保存できない --
            torch.save({
                "model_state_dict": model.state_dict(),
            }, f"{output_path}model-fold{fold}.pth")

            # cf.) 途中再開したい場合はmodel.state_dict()以外も必要なものアリ --
            ## torch.save({
            ##     "epoch": epoch,
            ##     "model_state_dict": model.state_dict(),
            ##     "optimizer_state_dict": optimizer.state_dict(),
            ##     "loss": valid_epoch_loss,
            ## }, f"{output_path}model-fold{fold}.pth")
            
            Write_log(log, f"Model Saved"); print()


    end_time = time.time()
    time_elapsed = end_time - start_time
    Write_log(log, "Training Complete in {:.0f}h {:.0f}m {:.0f}s".format(
        time_elapsed//3600, (time_elapsed%3600)//60, (time_elapsed%3600)%60
    ))
    Write_log(log, "Best Loss: {:.4f}".format(best_epoch_loss))

    model.load_state_dict(best_model_wts)

    return model, history


@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()  # modelはtrainの時点でto(device)されている前提 --

    preds = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)

        outputs = model(input_ids, attention_mask)

        preds.append(outputs.cpu().detach().numpy())

    preds = np.concatenate(preds)
    gc.collect()

    return preds


def inference(model_name, num_classes, custom_header, model_paths, dataloader, device):
    final_preds = []

    for i, path in enumerate([model_paths]):
        model = HateSpeechModel(model_name=model_name, num_classes=num_classes, custom_header=custom_header)
        model.to(device)
        checkpoint = torch.load(model_paths)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Getting predictions for model : {path}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds