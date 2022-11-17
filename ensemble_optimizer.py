#!/usr/bin/env python
# coding: utf-8
import os
import datetime
import optuna
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from config import *
from bert_utils import *

seed_everything(seed=42)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run_id_list", type=str, nargs="*", required=True)
parser.add_argument("--how", type=str, default="nelder-mead")
args, unknown = parser.parse_known_args()

# define run hash --
run_hash = datetime.datetime.now().strftime("%m_%d_%H%M%S")
if not os.path.exists(f"./output/ensemble_{run_hash}"):
    os.mkdir(f"./output/ensemble_{run_hash}")

run_id_list = args.run_id_list
assert len(run_id_list) > 0, f" ... run_id_list must be len() > 1 ..."

# ###########################################
#
#   -- weights & threshold optimization --
#
# ###########################################

# Nelder-Mead --
if args.how == "nelder-mead":
    result_df = pd.DataFrame(columns=([label_name] + run_id_list))
    for i, run_id in enumerate(run_id_list):
        t = pd.read_feather(f"{output_root}{run_id}/train_df.feather")
        if i == 0:
            result_df[label_name] = t[label_name]
        result_df[f"{run_id}"] = t.loc[:, "model_oof_class_1"].rename(f"{run_id}")

    def objective(x, result_df, run_id_list):
        ensemble_pred = np.zeros(result_df.shape[0])

        for i, run_id in enumerate(run_id_list):
            ensemble_pred += x[i]*result_df[f"{run_id}"]
        ensemble_pred = pd.Series(ensemble_pred).map(lambda v: 1 if v > x[-1] else 0)

        return -f1_score(ensemble_pred, result_df[label_name])

    def constraints(x):
        return (np.sum(x[:-1]) - 1)

    optimization_result = minimize(
        objective,
        x0 = [1.0/len(run_id_list)]*len(run_id_list) + [0.5],
        args = (result_df, run_id_list),
        method="Nelder-Mead",
        #bounds=[[-np.inf, np.inf]]*len(run_id_list) + [0, 1],
        constraints={"type": "eq", "fun": constraints},
    )

    print("\n\n {} \n\n".format(optimization_result["message"]))
    weights_threshold = optimization_result["x"]
    print(dict(optimization_result))


# optuna --
elif args.how == "optuna":
    
    def objective_ensemble(run_id_list):
        result_df = pd.DataFrame(columns=([label_name] + run_id_list))
        for i, run_id in enumerate(run_id_list):
            t = pd.read_feather(f"{output_root}{run_id}/train_df.feather")
            if i == 0:
                result_df[label_name] = t[label_name]
            result_df[f"{run_id}"] = t.loc[:, "model_oof_class_1"].rename(f"{run_id}")

            def objective(trial):
                x = []
                for run_id in run_id_list:
                    x.append(trial.suggest_float(f"{run_id}", 0, 1))
                x.append(trial.suggest_float("threshold", 0, 1))

                ensemble_pred = np.zeros(result_df.shape[0])
                for i, run_id in enumerate(run_id_list):
                    ensemble_pred += x[i]*result_df[f"{run_id}"]
                ensemble_pred = pd.Series(ensemble_pred).map(lambda v: 1 if v > x[-1] else 0)

                # constraints --
                c0 = (x[0] + x[1]) - 0.9999
                c1 = 1.0001 - (x[0] + x[1])
                trial.set_user_attr("constraint", (c0, c1))

                return f1_score(ensemble_pred, result_df[label_name])

            def constraints(trial):
                return trial.user_attrs["constraint"]

        return objective, constraints

    objective, constraints = objective_ensemble(run_id_list)

    sampler = optuna.samplers.NSGAIISampler(seed=42, constraints_func=constraints)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        )
    study.optimize(
        objective,
        n_trials=4000
    )

    weights_threshold = []
    for k, v in study.best_params.items():
        weights_threshold.append(v)
    optimization_result = study.best_trial

    print("Optuna Result : \n")
    print("weights   : ---> {}".format(weights_threshold[:-1]))
    print("threshold : ---> {}".format(weights_threshold[-1]))


# ###########################################
#
#        -- make submission file --
#
# ###########################################
test_df = pd.DataFrame(columns=(["id"] + run_id_list))
for i, run_id in enumerate(run_id_list):
    t = pd.read_feather(f"{output_root}{run_id}/test_df.feather")
    if i == 0:
        test_df["id"] = t["id"]
    test_df[f"{run_id}"] = t.loc[:, "model_oof_class_1"]

final_preds = np.zeros(test_df.shape[0])
for i, run_id in enumerate(run_id_list):
    final_preds += weights_threshold[i]*test_df[f"{run_id}"]
final_preds = pd.Series(final_preds).map(lambda v: 1 if v > weights_threshold[-1] else 0)

final_preds = pd.DataFrame({
    "id": test_df["id"],
    "final_preds": final_preds
})
sub = pd.read_csv(f"{data_path}sample_submission.csv")
sub = pd.merge(
    sub, final_preds, how="left", on="id"
).drop("label", axis=1).rename(columns={"final_preds": "label"})

# keep settings and optimization result --
settings = pd.Series()
settings["run_id_list"] = run_id_list
settings["weights"] = weights_threshold[0:-1]
settings["threshold"] = weights_threshold[-1]
settings["optimization_result"] = dict(optimization_result) if args.how=="nelder-mead" else optimization_result

# save --
settings.to_json(f"./output/ensemble_{run_hash}/settings.json", indent=4)
sub.to_csv(f"./output/ensemble_{run_hash}/sub_ens_{run_hash}.csv", index=False)
