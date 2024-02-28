import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vgg import vgg

conv_arch = ((1, 64), (1, 128))


class VGG_LSTM(nn.Module):
    def __init__(
        self, num_classes, input_dim, seq_len, lstm_hidden_dim=128, lstm_num_layers=1
    ):
        super(VGG_LSTM, self).__init__()

        # 加载预训练的VGG16模型
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.vgg = vgg(conv_arch, input_dim, seq_len)

        # LSTM参数
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )

        # 全连接层
        self.fc = nn.Linear(self.lstm_hidden_dim, num_classes)

    def forward(self, x):
        # 提取图像特征
        with torch.no_grad():
            x = torch.unsqueeze(x, 1)
            features = self.vgg(x)

        # 将特征转换为LSTM需要的格式
        features = features.view(
            features.size(0), -1, self.input_dim
        )  # (batch_size, sequence_length, feature_size)

        # LSTM层的前向传播
        lstm_out, _ = self.lstm(features)

        # 取LSTM输出的最后一个时间步
        lstm_out = lstm_out[:, -1, :]

        # 全连接层的前向传播
        output = self.fc(lstm_out)

        return output


if __name__ == "__main__":
    # 实例化CNN-LSTM模型
    num_classes = 10  # 分类任务的类别数
    cnn_lstm_model = VGG_LSTM(num_classes, 20, 50)

    # 打印模型结构
    print(cnn_lstm_model)