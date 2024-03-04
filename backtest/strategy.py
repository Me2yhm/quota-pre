import os
import sys
import math
from pathlib import Path


import torch
from torch.nn.functional import normalize
import pandas as pd
import pandas as np
import matplotlib.pyplot as plt
import lightgbm as lgb

from backtest.schema import futureAccount
from data.lstm_datloader import make_data, make_seqs
from model.vgg_lstm import VGG_LSTM
from gbdt import split_data


class tradeSignal:
    HARD_BUY: float = 0.5
    LITTLE_BUY: float = 0.2
    FLAT: float = 0.0
    LITTLE_SELL: float = -0.2
    HARD_SELL: float = -0.5


def read_data(code: str) -> pd.DataFrame:
    data_path = Path(__file__).parent.parent / f"data/{code}_test_data.csv"
    if os.path.exists(data_path):
        test_data = pd.read_csv(data_path)
    else:
        _, test_data = make_data(code)
    return test_data


def read_orin_data(code: str) -> pd.DataFrame:
    file_path = Path(__file__).parent.parent / f"data/{code}.csv"
    fu_dat = pd.read_csv(file_path)
    features = fu_dat.drop(columns=["change1", "ts_code"])
    data = features.iloc[:-1, :]
    test_data = (
        data[data["trade_date"] >= 20220913]
        .drop(columns=["trade_date"])
        .reset_index(drop=True)
    )
    return test_data


def make_vgg_data(code: str, seq_len: int) -> torch.Tensor:
    test_data = read_data(code)
    features = torch.tensor(test_data.iloc[:, :-1].to_numpy(), dtype=torch.float32)
    features = make_seqs(seq_len, features)
    return features


def make_gbdt_data(code: str, seq_len: int):
    _, _, test_data, _ = split_data(code, seq_len, False)
    return test_data.ffill()


def unilize(signals: torch.Tensor) -> torch.Tensor:
    normal = (signals - signals.min()) / (signals.max() - signals.min())
    return normal / normal.sum()


def execut_signal(
    code: str,
    account: futureAccount,
    weight: torch.Tensor,
    signals: torch.Tensor,
    price: float,
):
    volumes_rate = (unilize(signals) * weight).sum().item()
    # arg = signals.argmax().item()
    # volumes_rate = weight[arg].item()
    # # print(volumes_rate)
    account.order_to(code, volumes_rate, price)


def generate_signal(data, model):
    signals = model(data)
    return signals


def vgg_lstm_strategy(code: str, seq_len: int):
    pre_times = 0
    signals = None
    portfolio_values = []
    model_path = Path(__file__).parent.parent / "vgg_lstm_model.pth"
    model = VGG_LSTM(5, 20, 50, 100)
    model.load_state_dict(torch.load(model_path))
    has_siganl = False
    weight = torch.tensor([-0.5, -0.2, 0.0, 0.2, 0.5], dtype=torch.float32)
    account = futureAccount(current_date="2022-09-13", base=10000000, pool={})
    data = read_orin_data(code)
    test_data = make_vgg_data(code, seq_len)
    for i in range(len(data)):
        account.update_date(1)
        price = data.loc[i, ["close"]].item()
        if has_siganl:
            execut_signal(code, account, weight, signals, price)
            has_siganl = False
        account.update_price({code: price})
        portfolio_values.append(account.portfolio_value)
        if (i + 1) >= seq_len and i <= len(data) - seq_len:
            signals = generate_signal(test_data[pre_times].unsqueeze(0), model)
            pre_times += 1
            has_siganl = True
    return [v / portfolio_values[0] for v in portfolio_values]


def gbdt_strategy(code: str, seq_len: int):
    pre_times = 0
    signals = None
    portfolio_values = []
    model_path = Path(__file__).parent.parent / "model.txt"
    model = lgb.Booster(model_file=model_path).predict
    has_siganl = False
    weight = torch.tensor([-0.5, -0.2, 0.0, 0.2, 0.5], dtype=torch.float32)
    account = futureAccount(current_date="2022-09-13", base=10000000, pool={})
    data = read_orin_data(code)
    test_data = make_gbdt_data(code, seq_len)
    for i in range(len(data)):
        account.update_date(1)
        price = data.loc[i, ["close"]].item()
        if has_siganl:
            execut_signal(code, account, weight, signals, price)
            has_siganl = False
        account.update_price({code: price})
        portfolio_values.append(account.portfolio_value)
        if (i + 1) >= seq_len and i <= len(data) - seq_len:
            s = generate_signal([test_data.iloc[pre_times].to_numpy()], model).squeeze()
            signals = torch.tensor(s, dtype=torch.float32)
            pre_times += 1
            has_siganl = True
    return [v / portfolio_values[0] for v in portfolio_values]


def random_strategy(code: str, seq_len: int):
    pre_times = 0
    signals = None
    portfolio_values = []
    has_siganl = False
    weight = torch.tensor([-0.5, -0.2, 0.0, 0.2, 0.5], dtype=torch.float32)
    account = futureAccount(current_date="2022-09-13", base=10000000, pool={})
    data = read_orin_data(code)
    for i in range(len(data)):
        account.update_date(1)
        price = data.loc[i, ["close"]].item()
        if has_siganl:
            execut_signal(code, account, weight, signals, price)
            has_siganl = False
        account.update_price({code: price})
        account.calculate_portfolio_value()
        portfolio_values.append(account.portfolio_value)
        if (i + 1) >= seq_len and i <= len(data) - seq_len:
            signals = torch.randn(5)
            pre_times += 1
            has_siganl = True
    return [v / portfolio_values[0] for v in portfolio_values]


def bench_mark(code: str) -> pd.Series:
    data = read_orin_data(code)
    return data["close"] / data["close"][0]


if __name__ == "__main__":
    gbdt_result = gbdt_strategy("IC.CFX", 50)
    vgg_lstm_result = vgg_lstm_strategy("IC.CFX", 50)
    random_result = random_strategy("IC.CFX", 50)
    bench_result = list(bench_mark("IC.CFX").values)
    # print(pd.DataFrame(result).iloc[:, 0].values)
    df = pd.DataFrame(
        {
            "lstm": vgg_lstm_result,
            "gbdt": gbdt_result,
            "random": random_result,
            "bench": bench_result,
        }
    )
    df.dropna().plot()
    plt.show()