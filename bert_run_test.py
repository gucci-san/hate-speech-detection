import pandas as pd

from transformers import AutoTokenizer, T5Tokenizer
from torch.utils.data import DataLoader
from glob import glob

from bert_utils import *
from config import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="tmp")

args, unknown = parser.parse_known_args()

output_path = f"./output/{args.run_id}/"
settings = pd.read_json(f"{output_path}settings.json", typ="series")

model_paths = glob(f"{settings.output_path}*.pth"); model_paths.sort()
model_paths

test_df = pd.read_feather(f"{settings.output_path}test_df.feather")

# define tokenizer --
if settings.model_name in ["rinna/japanese-roberta-base"]:
    tokenizer = T5Tokenizer.from_pretrained(
        settings.model_name
    )
    tokenizer.do_lower_case = True
else:
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name,
        mecab_kwargs={"mecab_dic":None, "mecab_option": f"-d {dic_neologd}"}
    )

# make test preds --
test_dataset = HateSpeechDataset(
    test_df, tokenizer=tokenizer, 
    max_length=settings.max_length, num_classes=settings.num_classes, 
    text_col="clean_text", isTrain=False
    )
test_loader = DataLoader(test_dataset, batch_size=settings.test_batch_size, num_workers=2, shuffle=False, pin_memory=True)

preds_list = []
for fold in range(0, settings.folds):
    model_id = model_paths[fold].split("/")[3].split(".")[0].split("-")[0]
    preds = inference(settings.model_name, settings.num_classes, model_paths[fold], test_loader, device)
        
    for _class in range(0, settings.num_classes):
        test_df.loc[:, f"{model_id}_oof_class_{_class}"] = preds[:, _class]

    preds_list.append(preds)

final_preds = np.mean(np.array(preds_list), axis=0)
test_df[f"{model_id}_pred"] = np.argmax(final_preds, axis=1)


# make submission file --
submission = pd.read_csv(f"{data_path}sample_submission.csv")
submission = pd.merge(submission, test_df.loc[:, ["id", f"{model_id}_pred"]], how="left", on="id")
submission = submission.drop(["label"], axis=1).rename(columns={f"{model_id}_pred": "label"})
submission.to_csv(f"{settings.output_path}sub_{settings.run_id}.csv", index=False)

