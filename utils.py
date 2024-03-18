import os
from pathlib import Path
from datetime import datetime

import numpy as np
from chinese_calendar import is_holiday


def write_env(env_file_path, environ):
    with open(env_file_path, "+w") as file:
        for key, value in environ.items():
            file.write(f"{key}={value}\n")


def read_env(env_file_path):
    environ = {}
    if os.path.exists(env_file_path):
        with open(env_file_path, "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                environ[key] = value
    return environ


def is_trading_day(today: str):
    weekday = datetime.strptime(today, "%Y%m%d").weekday()
    today = datetime.strptime(today, "%Y%m%d").date()
    if weekday >= 5:
        return False
    if is_holiday(today):
        return False
    return True


def annualized_return(returns: float, holding_period_years: float = 1 / 360):
    """
    计算年化收益率（连续复利）

    parameter:
        returns: 期末净值/期初净值
        holding_period_years: 持有期（年）
    """
    # annual_return = np.log(returns) / holding_period_years
    annual_return = returns ** (1 / holding_period_years) - 1
    return annual_return


def calculate_sharpe_ratio(result: list[float], risk_free_rate: float = 0.02):
    """
    计算夏普比率

    parameter:
        returns: 日收益率序列
        risk_free_rate: 无风险利率，默认为0

    """
    returns = np.asarray(result)
    excess_return = annualized_return(returns[-1], len(returns) / 256) - risk_free_rate
    returns = returns[1:] / returns[:-1] - 1
    # returns = list(map(annualized_return, returns))
    # returns = [
    #     annualized_return(returns[i], (i + 1) / 360) for i in range(len(returns))
    # ]
    # returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate
    sharpe_ratio = (excess_return) / (np.std(excess_returns, ddof=1) * np.sqrt(256))
    return sharpe_ratio


def calculate_max_drawdown(returns):
    """
    计算最大回撤

    parameter:
        returns: 收益率序列

    """
    max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
    return max_drawdown


if __name__ == "__main__":
    path = Path(__file__).parent
    batch_size = 64
    input_dim = 63
    hidden_dim = 100
    seq_len = 50
    num_layers = 1
    class_num = 5
    code = "IC.CFX"
    if_agg = 0
    environ = {
        "BATCH_SIZE": str(batch_size),
        "INPUT_DIM": str(input_dim),
        "HIDDEN_DIM": str(hidden_dim),
        "SEQ_LEN": str(seq_len),
        "CLASS_NUM": str(class_num),
        "NUM_LAYERS": str(num_layers),
        "CODE": code,
        "IF_AGG": if_agg,
    }
    env_file_path = path / "env_vars.txt"
    write_env(env_file_path, environ)
