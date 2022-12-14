import pandas as pd

from torch.utils.data import DataLoader
from glob import glob

from bert_utils import *
from config import *

import argparse

# 計算時のsettingはtrainで保存したjsonから読み込む --
# run_idだけ指定 --
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, default="tmp")
parser.add_argument(
    "--single_pred",
    type=str,
    default=None,
    help="if wants prediction with single-model, define the absolute path of model.",
)
args, unknown = parser.parse_known_args()

save_hash = None

# settings, モデル作成時に前処理したtest_dfを読み込み --
output_path = f"{output_root}{args.run_id}/"
settings = pd.read_json(f"{output_path}settings.json", typ="series")
test_df = pd.read_feather(f"{settings.output_path}test_df.feather")

# define tokenizer --
tokenizer = define_tokenizer(settings.model_name)

# make test preds --
test_dataset = HateSpeechDataset(
    test_df,
    tokenizer=tokenizer,
    max_length=settings.max_length,
    num_classes=settings.num_classes,
    text_col="clean_text",
    isTrain=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=settings.test_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
)

# #################################################### #
#                                                      #
#                   -- test pred --                    #
#                                                      #
# #################################################### #
# run_idのディレクトリにあるpthを全て読み込んで、result-meanをするのが標準実装 --
# single_predが指定されればそれを使う --
if args.single_pred is not None:
    model_paths = [args.single_pred]
    use_model_num = len(model_paths)
    save_hash = "PRED_DEFINED_MODEL"
else:
    model_paths = glob(f"{settings.output_path}*.pth")
    model_paths.sort()
    use_model_num = settings.folds

preds_list = []
for fold in range(0, use_model_num):
    model_id = "model"
    preds = inference(
        settings.model_name,
        settings.max_length,
        settings.num_classes,
        settings.model_custom_header,
        settings.dropout,
        model_paths[fold],
        test_loader,
        device,
    )
    preds_list.append(preds)

final_preds = np.mean(np.array(preds_list), axis=0)
test_df[f"{model_id}_pred"] = np.argmax(final_preds, axis=1)
for _class in range(0, settings.num_classes):
    test_df.loc[:, f"{model_id}_oof_class_{_class}"] = final_preds[:, _class]

# update test_df.feather
test_df.to_feather(f"{settings.output_path}test_df_{save_hash}.feather")

# make submission file --
submission = pd.read_csv(f"{data_path}sample_submission.csv")
submission = pd.merge(
    submission, test_df.loc[:, ["id", f"{model_id}_pred"]], how="left", on="id"
)
submission = submission.drop(["label"], axis=1).rename(
    columns={f"{model_id}_pred": "label"}
)

submission.to_csv(
    f"{settings.output_path}sub_{settings.run_id}_{save_hash}.csv", index=False
)
