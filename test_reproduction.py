import pandas as pd
import numpy as np
import torch
from config import *

from glob import glob
from tqdm import tqdm
from typing import OrderedDict
from colorama import Fore

g_ = Fore.GREEN
y_ = Fore.YELLOW
r_ = Fore.RED
sr_ = Fore.RESET

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_id1", type=str, required=True)
parser.add_argument("--run_id2", type=str, required=True)
args, unknown = parser.parse_known_args()


def torch_parameter_compare(m1_path: str, m2_path: str) -> list:
    """2つのpthのパスを受け取り、state_dictの各キーごとにtensorの値が一致するかどうかを確認する関数"""
    print(f"Compare with")
    print(f"    --> {m1_path}")
    print(f"    --> {m2_path}")

    # pthのロード --
    m1 = torch.load(m1_path)["model_state_dict"]
    m2 = torch.load(m2_path)["model_state_dict"]

    # そもそもkeyが完全一致しなければ計算しない, modelのstructureを確認してほしい --
    assert (
        len(set(m1.keys()) - set(m2.keys())) == 0
    ), "... m1.keys() vs m2.keys() dosen't match ..."

    # 各キーごとにtensorを比較する --
    unmatched_keys = []
    for key in m1.keys():
        key_matches = torch.eq(m1[key], m2[key]).all().item()
        if not key_matches:
            print(f"{g_}{key} --> {r_}{key_matches}{sr_}")
            unmatched_keys.append(key)

    if len(unmatched_keys) == 0:
        print(f"{g_}        --> All keys matched ... {sr_}")
    else:
        print(f"{y_}        --> Some keys unmatched : {unmatched_keys} {sr_}")

    return unmatched_keys


def main():
    path1 = f"{output_root}{args.run_id1}/*.pth"
    path2 = f"{output_root}{args.run_id2}/*.pth"
    model_paths1 = glob(path1)
    model_paths2 = glob(path2)
    model_paths1.sort()
    model_paths2.sort()

    assert len(model_paths1) == len(
        model_paths2
    ), f"Two run_id contains different models (pth files), {len(model_paths1)} and {len(model_paths2)}"

    for fold in range(0, len(model_paths1)):
        _ = torch_parameter_compare(model_paths1[fold], model_paths2[fold])

    print()
    print(f"{g_} ##################################{sr_}")
    print(f"{g_} #                                #{sr_}")
    print(f"{g_} #   -- All folds pth matched --  #{sr_}")
    print(f"{g_} #                                #{sr_}")
    print(f"{g_} ##################################{sr_}")
    print()


if __name__ == "__main__":
    main()
