import os
from typing import Iterable, Literal
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils import read_env

env_path = Path(__file__).parent.parent / "env_vars.txt"
env = read_env(env_path)
os.environ.update(env)
input_dim = int(os.environ["INPUT_DIM"])


def tag_zs(zs: float) -> list:
    if zs >= 1.5:
        return [0, 0, 0, 0, 1]
    elif 0.2 <= zs < 1.5:
        return [0, 0, 0, 1, 0]
    elif -0.2 < zs < 0.2:
        return [0, 0, 1, 0, 0]
    elif -1.5 < zs <= -0.2:
        return [0, 1, 0, 0, 0]
    else:
        return [1, 0, 0, 0, 0]


def trans_class_num(cls: list):
    return cls.index(max(cls))


def cal_zscore(pcg: list):
    pcg = np.where(np.isinf(pcg), 0, pcg)
    zscores = [
        (pcg[i] - np.mean(pcg[1:])) / np.std(pcg[1:]) for i in range(1, len(pcg))
    ]
    zscores.insert(0, 0)
    return zscores


def mark_zscore(zscores: list):
    return list(map(tag_zs, zscores))


def read_daily_data(code: str) -> pd.DataFrame:
    file_path = Path(__file__).parent / f"{code}.csv"
    fu_dat = pd.read_csv(file_path)
    return fu_dat


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    pcg = list(data["close"].values)
    data["pcg_zscore"] = cal_zscore(pcg)
    return data


def data_to_zscore(data: pd.DataFrame) -> pd.DataFrame:
    features = data.drop(columns=["change1", "ts_code"])
    returns = features.iloc[:, 1:22].astype(float).apply(np.log).diff()
    indicaters = features.iloc[:, 22:].astype(float)
    for i in range(1, 22):
        features.iloc[:, i] = cal_zscore(returns.iloc[:, i - 1].values)
    for i in range(22, len(features.columns)):
        features.iloc[:, i] = cal_zscore(indicaters.iloc[:, i - 22].values)
    return features.iloc[1:, :].reset_index(drop=True)


def make_seqs(seq_len: int, data: Iterable) -> torch.Tensor:
    num_samp = len(data)
    return torch.stack([data[i : i + seq_len] for i in range(num_samp - seq_len + 1)])


def make_seq_dataset(data: pd.DataFrame, seq_len: int) -> TensorDataset:
    x_dat = torch.tensor(data.iloc[:-1, :].to_numpy(), dtype=torch.float32)
    marked_y = mark_zscore(data.iloc[1:]["pcg_zscore"].values)
    y_dat = torch.tensor(marked_y, dtype=torch.float32)
    x = make_seqs(seq_len, x_dat)
    y = make_seqs(seq_len, y_dat)[:, -1, :]
    dataset = TensorDataset(x, y)
    return dataset


def split_train_dataset(dataset: TensorDataset, split_index: int) -> TensorDataset:
    train_dataset = dataset[:split_index]
    x, y = train_dataset
    seq_len = x.size(1)
    input_dim = x.size(2)
    x = x.view(x.size(0), -1).numpy()
    y = y.numpy()
    ros = SMOTE(k_neighbors=2)
    x, y = ros.fit_resample(x, y)
    x = torch.tensor(x, dtype=torch.float32).view(-1, seq_len, input_dim)
    y = torch.tensor(y, dtype=torch.float32)
    train_dataset = TensorDataset(x, y)
    return train_dataset


def split_test_dataset(dataset: TensorDataset, split_index: int) -> TensorDataset:
    test_dataset = dataset[split_index:]
    test_dataset = TensorDataset(*test_dataset)
    return test_dataset


def make_dataloader(dataset: TensorDataset, batch_size: int, shuffle: bool = True):
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader


def make_vgg_train_data(
    code: str, split_date: int, seq_len: int, batch_size, shuffle: bool = True
) -> DataLoader:
    fu_dat = read_daily_data(code)
    split_index = fu_dat.index[fu_dat["trade_date"] == split_date][0] - len(
        fu_dat.index
    )
    features = data_to_zscore(fu_dat)
    marked_dat = add_target(features)
    marked_dat.drop(columns=["trade_date"], inplace=True)
    dataset = make_seq_dataset(marked_dat, seq_len)
    train_dataset = split_train_dataset(dataset, split_index)
    train_dataloader = make_dataloader(train_dataset, batch_size, shuffle)
    return train_dataloader


def make_vgg_test_data(
    code: str, split_date: int, seq_len: int, batch_size: int
) -> DataLoader:
    fu_dat = read_daily_data(code)
    split_index = fu_dat.index[fu_dat["trade_date"] == split_date][0] - len(
        fu_dat.index
    )
    features = data_to_zscore(fu_dat)
    marked_dat = add_target(features)
    marked_dat.drop(columns=["trade_date"], inplace=True)
    dataset = make_seq_dataset(marked_dat, seq_len)
    train_dataset = split_test_dataset(dataset, split_index)
    train_dataloader = make_dataloader(train_dataset, batch_size, False)
    return train_dataloader


def get_test_data(code: str, split_date: int, seq_len: int):
    data = read_daily_data(code)
    data = data_to_zscore(data)
    split_index = data.index[data["trade_date"] == split_date][0] - len(data.index)
    data.drop(columns=["trade_date"], inplace=True)
    data = data.iloc[split_index - seq_len + 1 :]
    return data


def make_gbdt_data(dataset: TensorDataset) -> tuple[pd.DataFrame]:
    feats, targets = dataset.tensors
    feats = feats.view(feats.size(0), -1).numpy()
    x_dat = pd.DataFrame(feats)
    y_dat = pd.DataFrame(targets.unsqueeze(1).numpy().tolist())
    y_dat = y_dat.iloc[:, -1].apply(trans_class_num).reset_index(drop=True)
    return x_dat, y_dat


def gbdt_train_data(
    code: str, split_date: int, seq_len: int, shuffle: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = make_vgg_train_data(code, split_date, seq_len, 64, shuffle).dataset
    x_train_dat, y_train_dat = make_gbdt_data(dataset)
    return x_train_dat, y_train_dat


def gbdt_test_data(
    code: str, split_date: int, seq_len: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = make_vgg_test_data(code, split_date, seq_len, 64).dataset
    x_test_dat, y_test_dat = make_gbdt_data(dataset)
    return x_test_dat, y_test_dat


def split_gbdt_data(
    code: str, split_date: int, seq_len: int, shuffle: bool = True
) -> tuple[pd.DataFrame, ...]:
    x_train_dat, y_train_dat = gbdt_train_data(code, split_date, seq_len, 64, shuffle)
    x_test_dat, y_test_dat = gbdt_test_data(code, split_date, seq_len, 64)
    return x_train_dat, y_train_dat, x_test_dat, y_test_dat


if __name__ == "__main__":
    split_date = int(os.environ["split_date".upper()])
    code = "IF.CFX"
    fu_dat = read_daily_data(code)
    split_index = fu_dat.index[fu_dat["trade_date"] == split_date][0] - len(
        fu_dat.index
    )
    features = data_to_zscore(fu_dat)
    marked_dat = add_target(features)
    marked_dat.drop(columns=["trade_date"], inplace=True)
    dataset = make_seq_dataset(marked_dat, 50)
    train_dataset = split_test_dataset(dataset, split_index)
    train_dataloader = make_dataloader(train_dataset, 64, False)
    gbdt = make_gbdt_data(train_dataset)
    x, y = gbdt
    print(x, "\n", y)
    breakpoint()
