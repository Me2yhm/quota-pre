import os
from typing import Literal
from pathlib import Path
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from data.lstm_datloader import (
    gbdt_test_data,
    gbdt_train_data,
)
from utils import read_env

num_round = 10
env_path = Path(__file__).parent / "env_vars.txt"
os.environ.update(read_env(env_path))
windows = int(os.environ["SEQ_LEN"])
code = os.environ["code"]
split_date = int(os.environ["SPLIT_DATE"])


def train_gbdt(
    code: str, seq_len: int, split_date: int = 20220913, shuffle: bool = True
):
    x_train, y_train = gbdt_train_data(code, split_date, seq_len, shuffle)

    # 模型训练
    train_data = lgb.Dataset(x_train, label=y_train)
    params = {
        "num_leaves": 31,
        "num_trees": 100,
        "metric": "multi_error",
        "objective": "multiclass",
        "num_class": 5,
    }
    bst = lgb.train(params, train_data, num_round)
    bst.save_model(f"{code}_gbdt_model.txt")
    return bst


def test_gbdt(code: str, seq_len: int, split_date: int = 20220913):
    x_test, y_test = gbdt_test_data(code, split_date, seq_len)
    # 模型效果评估
    model_path = Path(__file__).parent / f"{code}_gbdt_model.txt"
    bst = lgb.Booster(model_file=model_path)
    y_pred = bst.predict(x_test)
    y_pred = pd.Series(map(lambda x: x.argmax(), y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    y_test.index = range(0, len(y_test.index))
    y_pred = y_pred
    print(sum(y_test == y_pred) / len(y_pred))
    print(accuracy)


if __name__ == "__main__":
    train_gbdt(code, windows, split_date=split_date)
    test_gbdt(code, windows, split_date=split_date)
