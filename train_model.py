import os
from math import sqrt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from data.lstm_datloader import make_vgg_train_data
from model.vgg_lstm import VGG_LSTM
from utils import read_env

path = Path(__file__).parent
env_path = path / "env_vars.txt"
os.environ.update(read_env(env_path))
batch_size = int(os.environ["BATCH_SIZE"])
input_dim = int(os.environ["INPUT_DIM"])
hidden_dim = int(os.environ["HIDDEN_DIM"])
seq_len = int(os.environ["SEQ_LEN"])
num_layers = int(os.environ["NUM_LAYERS"])
class_num = int(os.environ["CLASS_NUM"])
code = os.environ["CODE"]
codes = ["IH.CFX", "IF.CFX", "IC.CFX"]
if_agg = bool(int(os.environ["IF_AGG"]))
split_date = int(os.environ["SPLIT_DATE"])


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.distance = torch.tensor([-0.5, -0.34, 0, 0.34, 0.5], dtype=float)
        self.weight = torch.tensor(
            [
                [-0.6, 0, 0, -0.8, 0],
                [0, -sqrt(2) / 2, 0.5, -0.5, 0],
                [0, 0, 1, 0, 0],
                [0, sqrt(2) / 2, 0.5, 0.5, 0],
                [0, 0, 0, 0.8, 0.6],
            ],
            dtype=torch.float32,
        )

    def forward(self, output, target):
        """
        通过预测向量和真实向量的夹角来度量损失程度
        为了使5类真实类别向量的夹角不均匀,加了一个线性变换,即self.weight
        """
        assert (
            output.size() == target.size()
        ), "the size of output should mathch the target"
        err_multi = torch.zeros_like(output, dtype=torch.float32)
        true_target = torch.zeros_like(target, dtype=torch.float32)
        for i in range(target.size(0)):
            err_multi[i] = torch.mv(self.weight, output[i])
            true_target[i] = torch.mv(self.weight, target[i])
        loss = self.cosin(err_multi, true_target).sum() / target.size(0)
        return loss

    def cosin(self, output, targets):
        cosin = (output * targets).sum(dim=1, keepdim=True) / (
            torch.norm(output, dim=1, keepdim=True)
            * torch.norm(targets, dim=1, keepdim=True)
        )
        return torch.exp(-cosin)


def train_vgg_lstm(
    model: torch.nn.Module,
    data: DataLoader,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    code: str,
    epochs=100,
):
    losses = []
    accs = []
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(
                y_pred.to(dtype=torch.float), y_actual.to(dtype=torch.float)
            )
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        for x, y_actual in data:
            # Get the error rate for the whole batch:
            y_pred = model(x)
            loss = criterion(
                y_pred.to(dtype=torch.float), y_actual.to(dtype=torch.float)
            ).item()
            pred = torch.zeros_like(y_pred)
            batch_num = y_pred.size(0)
            for i in range(batch_num):
                max_ind = y_pred[i].argmax()
                pred[i][max_ind] = 1
            corrects = (pred.data == y_actual.data).all(dim=1).sum()
            acc = corrects / batch_num * 100
            accs.append(acc)
            losses.append(loss)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print(
                "epoch {:4d}: loss={:.5f}, Accracy={:.5f}%".format(
                    t,
                    sum(losses) / len(losses),
                    sum(accs) / len(accs),
                )
            )
            torch.save(model.state_dict(), f"vgg_lstm_model_{code}.pth")
    torch.save(model.state_dict(), f"vgg_lstm_model_{code}.pth")
    return model


def update_vgg_lstm(
    model: torch.nn.Module,
    data: DataLoader,
    code: str,
    epochs=100,
    if_save: bool = False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = CustomLoss()
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(
                y_pred.to(dtype=torch.float), y_actual.to(dtype=torch.float)
            )
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if if_save:
                torch.save(model.state_dict(), path / f"vgg_lstm_model_{code}.pth")
    if if_save:
        torch.save(model.state_dict(), path / f"vgg_lstm_model_{code}.pth")
    return model


def mk_vgg_lstm_model(
    code: str, batch_size: int, seq_len: int, split_date: int = 20220913
):
    data = make_vgg_train_data(code, split_date, seq_len, batch_size)
    model = VGG_LSTM(class_num, input_dim, seq_len, hidden_dim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = CustomLoss()
    return train_vgg_lstm(model, data, optimizer, criterion, code, 100)


def agg_data_train(batch_size: int, seq_len: int, split_date: int = 20220913):
    from torch.utils.data import DataLoader, ConcatDataset

    print("----------train agg model----------")
    data1 = make_vgg_train_data(code[0], split_date, seq_len, batch_size).dataset
    data2 = make_vgg_train_data(code[1], split_date, seq_len, batch_size).dataset
    data3 = make_vgg_train_data(code[2], split_date, seq_len, batch_size).dataset
    combined_dataset = ConcatDataset([data1, data2, data3])
    data = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    model = VGG_LSTM(class_num, input_dim, seq_len, hidden_dim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = CustomLoss()
    return train_vgg_lstm(model, data, optimizer, criterion, "agg", 100)


if __name__ == "__main__":
    print(f"----------train subject {code}----------")
    if if_agg:
        agg_data_train(batch_size, seq_len, split_date=split_date)
    else:
        model = mk_vgg_lstm_model(code, batch_size, seq_len, split_date=split_date)
