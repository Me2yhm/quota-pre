import os
from pathlib import Path
from typing import Callable
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

import sys

sys.path.append(str(Path(__file__).parent.parent))

from backtest.schema import BREED, futureAccount
from data.lstm_datloader import (
    cal_zscore,
    data_to_zscore,
    get_labled_data,
    make_data,
    make_data_loader,
    make_seqs,
)
from model.vgg_lstm import VGG_LSTM
from gbdt import split_data, train_gbdt
from train_model import mk_vgg_lstm_model, update_vgg_lstm
from utils import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    is_trading_day,
    read_env,
)

root_path = Path(__file__).parent
env_path = root_path.parent / "env_vars.txt"
env = read_env(env_path)
os.environ.update(env)
input_dim = int(os.environ["INPUT_DIM"])
num_class = int(os.environ["CLASS_NUM"])
seq_len = int(os.environ["SEQ_LEN"])
hidden_dim = int(os.environ["HIDDEN_DIM"])
code = os.environ["CODE"]
if_agg = bool(int(os.environ["IF_AGG"]))
split_date = int(os.environ["SPLIT_DATE"])

breed_IH = BREED("IH.CFX", 300, True)
breed_IF = BREED("IF.CFX", 300, True)
breed_IC = BREED("IC.CFX", 200, True)
breed_IM = BREED("IM.CFX", 200, True)

breed_dic = {bree.breed: bree for bree in [breed_IH, breed_IC, breed_IF, breed_IM]}


class tradeSignal:
    HARD_BUY: float = 0.5
    LITTLE_BUY: float = 0.3
    FLAT: float = 0.0
    LITTLE_SELL: float = -0.3
    HARD_SELL: float = -0.5


def read_data(code: str) -> pd.DataFrame:
    data_path = root_path.parent / f"data/{code}_test_data.csv"
    if os.path.exists(data_path):
        test_data = pd.read_csv(data_path)
    else:
        _, test_data = make_data(code)
    return test_data


def read_orin_data(code: str) -> pd.DataFrame:
    file_path = root_path.parent / f"data/{code}.csv"
    fu_dat = pd.read_csv(file_path)
    data = fu_dat.iloc[:, :]
    test_data = data[data["trade_date"] >= split_date].reset_index(drop=True)
    return test_data


def make_pre_data(test_data: pd.DataFrame) -> torch.Tensor:
    features = test_data.drop(columns=["change1", "ts_code"]).reset_index(drop=True)
    returns = features.iloc[:, 1:22].astype(float).apply(np.log).diff()
    indicaters = features.iloc[:, 22:].astype(float)
    for i in range(1, 22):
        features.iloc[:, i] = cal_zscore(returns.iloc[:, i - 1].values)
    for i in range(22, 53):
        features.iloc[:, i] = cal_zscore(indicaters.iloc[:, i - 26].values)
    # test_data = features[features["trade_date"] >= split_date].reset_index(drop=True)
    features = test_data.iloc[:, 1:]
    features = torch.tensor(features.values, dtype=torch.float32)
    return features


def make_vgg_data(code: str, seq_len: int) -> torch.Tensor:
    file_path = root_path.parent / f"data/{code}.csv"
    test_data = pd.read_csv(file_path)
    features = make_pre_data(test_data)
    features = make_seqs(seq_len, features)
    return features


def make_gbdt_data(code: str, seq_len: int):
    _, _, test_data, _ = split_data(code, seq_len, False)
    return test_data.ffill()


def unilize(signals: torch.Tensor) -> torch.Tensor:
    normal = (signals - signals.min()) / (signals.max() - signals.min())
    return normal / normal.sum()


def execut_signal(
    breed: BREED,
    account: futureAccount,
    weight: torch.Tensor,
    signals: torch.Tensor,
    price: float,
):
    volumes_rate = (unilize(signals) * weight).sum().item()
    # volume_rate 大于0和小于0时 order_to_rate的处理逻辑有所区别
    volumes_rate = volumes_rate if volumes_rate >= 0 else -(1 + volumes_rate)
    # print(
    #     f"-----rate:{volumes_rate},price:{price},pools:{account.pools},return:{account.pfo_return},fu:{account.fu_pools}-----"
    # )
    account.order_to_rate(breed, volumes_rate, price)
    # print(
    #     f"-----pools:{account.pools},return:{account.pfo_return},fu:{account.fu_pools}-----"
    # )


def generate_signal(data, model):
    signals = model(data)
    return signals


def vgg_update_model(code: str, seq_len: int, batch_size: int) -> Callable:
    update_fuc = partial(mk_vgg_lstm_model, code, batch_size, seq_len)
    return update_fuc


def gbdt_update_model(code: str, seq_len: int) -> Callable:
    model = partial(train_gbdt, code, seq_len)

    def update_fuc(split_date: int):
        return model(split_date).predict

    return update_fuc


def lstm_updata_fuc(orin_data: pd.DataFrame, seq_len: int, batch_size: int):
    data = data_to_zscore(orin_data).drop(columns=["trade_date"]).reset_index(drop=True)
    x, y = get_labled_data(data, seq_len, resample=True)
    dataloader = make_data_loader(x, y, batch_size)
    return dataloader


class strategy:
    model: Callable
    weight: torch.Tensor
    has_signal: bool
    pre_times: int
    code: str
    seq_len: int
    orin_data: pd.DataFrame
    test_data: pd.DataFrame
    account: futureAccount
    portfolio_values: list
    start_date: str
    end_date: str

    def __init__(
        self,
        code: str,
        seq_len: int,
        test_data: pd.DataFrame,
        model: Callable | None = None,
        update: bool = False,
    ) -> None:
        self.breed = breed_dic[code]
        self.model = model
        self.signals = None
        self.has_signal = False
        self.update = update
        self.seq_len = seq_len
        self.test_data = test_data
        self.orin_data = read_orin_data(code)
        self.win_times = []
        self.odds = {"win": [], "loss": []}
        self.portfolio_values = []
        self.pfo_returns = []
        self.weight = torch.tensor([-0.5, -0.2, 0.0, 0.2, 0.5], dtype=torch.float32)
        self.account = futureAccount(
            base_currency=10000000, current_date=f"{split_date}", fu_pools={}
        )
        self.start_date = self.account.current_date
        file_path = root_path.parent / f"data/{code}.csv"
        fu_data = pd.read_csv(file_path)
        self.split_index = fu_data.index[fu_data["trade_date"] == split_date][0] - len(
            fu_data.index
        )
        self.fu_dat = fu_data

    def excute_stratgy(
        self,
        signal_gerater: Callable,
        update_fuc: Callable | None = None,
        data_fuc: Callable | None = None,
    ):
        for i in range(len(self.orin_data)):
            self.account.update_date(1)
            while not is_trading_day(self.account.current_date):
                self.account.update_date(1)
            price = self.orin_data.loc[i, ["close"]].item()
            self.daily_settle(price)
            self.portfolio_values.append(self.account.portfolio_value)
            self.pfo_returns.append(self.account.pfo_return / self.account.base)
            self.signals = signal_gerater(
                self.test_data[i + self.split_index], self.model
            )
            if i - (self.seq_len) > 0 and self.update:
                if (i - (self.seq_len)) % 30 == 0:
                    j = i + self.split_index
                    data = data_fuc(
                        self.fu_dat.iloc[
                            self.split_index - 2 * self.seq_len : j + 1
                        ].reset_index(drop=True)
                    )
                    self.update_model(update_fuc, data)
            self.has_signal = True
            if self.has_signal:
                execut_signal(
                    self.breed, self.account, self.weight, self.signals, price
                )
                self.has_signal = False
        self.end_date = self.account.current_date

    def daily_settle(self, current_price: float):
        if self.account.fu_pools:
            old_return = self.account.pfo_return
            self.account.order_to_rate(self.breed, 0, current_price)
            returns = self.account.pfo_return
            intrest = returns - old_return
            if intrest > 0:
                self.win_times.append(1)
                self.odds["win"].append(intrest)
            else:
                self.win_times.append(0)
                self.odds["loss"].append(abs(intrest))

    def update_model(self, update_fuc: Callable, data):
        update_fuc(self.model, data)
        pass


def lstm_sig_gener(data, model) -> torch.Tensor:
    print("prediction----", generate_signal(data.unsqueeze(0), model))
    return generate_signal(data.unsqueeze(0), model)


def roll_date(date: str):
    date_format = "%Y%m%d"
    old_date = datetime.strptime(date, date_format)
    new_date = old_date + timedelta(days=-1)
    while not is_trading_day(new_date.strftime(date_format)):
        new_date = new_date + timedelta(days=-1)
    return new_date.strftime(date_format)


def vgg_lstm_strategy(code: str, seq_len: int, if_agg: bool = False):
    model_path = root_path.parent / f"vgg_lstm_model_{code}.pth"
    if if_agg:
        model_path = root_path.parent / "vgg_lstm_model_agg.pth"
    model = VGG_LSTM(num_class, input_dim, seq_len, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    update_fuc = partial(update_vgg_lstm, code=code)
    data_fuc = partial(lstm_updata_fuc, seq_len=seq_len, batch_size=64)
    test_data = make_vgg_data(code, seq_len)
    executer = strategy(code, seq_len, test_data, model, update=True)
    executer.excute_stratgy(lstm_sig_gener, update_fuc, data_fuc)
    portfolio_values = executer.portfolio_values
    win_times = sum(executer.win_times)
    lose_times = len(executer.win_times) - win_times
    win_return = sum(list(executer.odds["win"]))
    loss = sum(list(executer.odds["loss"]))
    win_rate = win_times / len(executer.win_times)
    odds = (win_return / win_times) / (loss / lose_times)
    breakpoint()
    return [v / portfolio_values[0] for v in portfolio_values], win_rate, odds


def gbdt_sig_gener(data, model) -> torch.Tensor:
    s = generate_signal([data.to_numpy()], model).squeeze()
    signals = torch.tensor(s, dtype=torch.float32)
    return signals


def random_gener(data, model) -> torch.Tensor:
    return torch.randn(5)


def gbdt_strategy(code: str, seq_len: int):
    model_path = root_path.parent / f"{code}_gbdt_model.txt"
    model = lgb.Booster(model_file=model_path).predict
    update_fuc = gbdt_update_model(code, seq_len)
    test_data = make_gbdt_data(code, seq_len).iloc
    executer = strategy(code, seq_len, test_data, model)
    executer.excute_stratgy(gbdt_sig_gener, update_fuc)
    portfolio_values = executer.portfolio_values
    win_times = sum(executer.win_times)
    lose_times = len(executer.win_times) - win_times
    win_return = sum(list(executer.odds["win"]))
    loss = sum(list(executer.odds["loss"]))
    win_rate = win_times / len(executer.win_times)
    odds = (win_return / win_times) / (loss / lose_times)
    return [v / portfolio_values[0] for v in portfolio_values], win_rate, odds


def random_strategy(code: str, seq_len: int):
    test_data = make_gbdt_data(code, seq_len).iloc
    executer = strategy(code, seq_len, test_data)
    executer.excute_stratgy(random_gener)
    portfolio_values = executer.portfolio_values
    win_times = sum(executer.win_times)
    lose_times = len(executer.win_times) - win_times
    win_return = sum(list(executer.odds["win"]))
    loss = sum(list(executer.odds["loss"]))
    win_rate = win_times / len(executer.win_times)
    odds = (win_return / win_times) / (loss / lose_times)
    return [v / portfolio_values[0] for v in portfolio_values], win_rate, odds


def bench_mark(code: str) -> pd.Series:
    data = read_orin_data(code)
    return data["close"] / data["close"][0]


if __name__ == "__main__":
    if if_agg:
        print("----------------use agg model----------------")
    test_data = read_orin_data(code)
    test_date = test_data["trade_date"].apply(
        lambda x: str(x)[2:4] + "-" + str(x)[4:6] + "-" + str(x)[6:]
    )
    vgg_lstm_result, lstm_wrate, lstm_odds = vgg_lstm_strategy(code, seq_len, if_agg)
    gbdt_result, gbdt_wrate, gbdt_odds = gbdt_strategy(code, seq_len)
    random_result, rand_wrate, rand_odds = random_strategy(code, seq_len)
    bench_result = list(bench_mark(code).values)
    sharps = list(
        map(
            calculate_sharpe_ratio,
            [vgg_lstm_result, gbdt_result, random_result],
        )
    )
    drowdowns = list(
        map(calculate_max_drawdown, [vgg_lstm_result, gbdt_result, random_result])
    )
    returns = pd.DataFrame(
        {
            "lstm": vgg_lstm_result,
            "gbdt": gbdt_result,
            "random": random_result,
            "bench": bench_result,
        },
        index=test_date,
    )
    rates = pd.DataFrame(
        {
            "strategy": ["lstm", "gbdt", "random"],
            "win_rates": [lstm_wrate, gbdt_wrate, rand_wrate],
            "odds": [lstm_odds, gbdt_odds, rand_odds],
            "sharp": sharps,
            "Max_drawdown": drowdowns,
        }
    )
    print(f"----------strategy subject {code}-------------")
    print(rates)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rc("font", size=12)
    returns.dropna().plot()
    plt.xlabel("回测时间")
    plt.ylabel("净值")
    plt.show()
