import os, gc, random, time, copy
import warnings; warnings.simplefilter("ignore")
import numpy as np

import re
import demoji
import neologdn

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModel 
)

from tqdm import tqdm
from collections import defaultdict
from colorama import Fore
b_ = Fore.BLUE; y_ = Fore.YELLOW; g_ = Fore.GREEN; sr_ = Fore.RESET
from config import *


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


class HateSpeechDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, num_classes, text_col="text"):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df[text_col].values
        self.target = df[label_name].values
        self.num_classes = num_classes

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
        target = int(self.target[index])

        onehot_t = np.zeros(self.num_classes, dtype=np.float32)
        onehot_t[target] = 1.0

        return {
            "input_ids": torch.tensor(inputs_text["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs_text["attention_mask"], dtype=torch.long),
            "target": torch.tensor(onehot_t, dtype=torch.float)
        }


class HateSpeechModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(HateSpeechModel, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True,
            )
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        out = self.dropout(out[1])
        outputs = self.fc(out)
        outputs = self.sigmoid(outputs)

        return outputs.squeeze()


def prepare_loaders(df, fold, tokenizer, trn_batch_size, val_batch_size, max_length, num_classes, text_col="text"):
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = HateSpeechDataset(train_df, tokenizer=tokenizer, max_length=max_length, num_classes=num_classes, text_col=text_col)
    valid_dataset = HateSpeechDataset(valid_df, tokenizer=tokenizer, max_length=max_length, num_classes=num_classes, text_col=text_col)

    train_loader = DataLoader(
        train_dataset, batch_size=trn_batch_size, num_workers=2, shuffle=True, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=val_batch_size, num_workers=2, shuffle=False, pin_memory=True
    )
    return train_loader, valid_loader


def criterion(outputs, targets):
    loss_f = nn.BCELoss()
    return loss_f(outputs, targets)


def fetch_scheduler(scheduler, optimizer, T_max=500, eta_min=1e-7):
    if scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    else:
        print(f"*** *** NOT implemented *** *** ")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    return scheduler


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, n_accumulate):
    model.train()

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
        loss = loss / np.float(n_accumulate)
        loss.backward()

        if (step+1) % n_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

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


def run_training(model, train_loader, valid_loader, optimizer, scheduler, n_accumulate, device, num_epochs, fold, output_path):

    if torch.cuda.is_available():
        print(f"[INFO] Using GPU : {torch.cuda.get_device_name()}\n")

    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs+1):
        gc.collect()

        train_epoch_loss = train_one_epoch(
            model, optimizer, scheduler,
            dataloader=train_loader,
            device=device, epoch=epoch,
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
            print(f"{b_}Valid Loss Improved : {best_epoch_loss:.6f} ---> {valid_epoch_loss:.6f}")
            best_epoch_loss = valid_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            #torch.save(model.state_dict(), f"{output_path}model-state-dict-fold{fold}.bin")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": valid_epoch_loss,
            }, f"{output_path}model-fold{fold}.pth")  # 途中再開したい場合はmodel.state_dict()以外も必要 --
            print(f"Model Saved{sr_}"); print()

    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Training Complete in {:.0f}h {:.0f}m {:.0f}s".format(
        time_elapsed//3600, (time_elapsed%3600)//60, (time_elapsed%3600)%60
    ))
    print("Best Loss: {:.4f}".format(best_epoch_loss))

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


def inference(model_name, num_classes, model_paths, dataloader, device):
    final_preds = []

    for i, path in enumerate([model_paths]):
        model = HateSpeechModel(model_name=model_name, num_classes=num_classes)
        model.to(device)
        checkpoint = torch.load(model_paths)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Getting predictions for model : {path}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)


    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds