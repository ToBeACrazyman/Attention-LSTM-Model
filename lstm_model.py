import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Define GRU NETWORK structure
class GRU_bi(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU_bi, self).__init__()
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = r_out
        return out


# Define GRU NETWORK structure
class GRU_bi_toLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU_bi_toLinear, self).__init__()
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = r_out[:, -1, :]
        return out


# Define GRU NETWORK structure
class GRU_uni(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU_uni, self).__init__()
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, x):
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = r_out
        return out

    # Define GRU NETWORK structure


class GRU_uni_toLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU_uni_toLinear, self).__init__()
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, x):
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = r_out[:, -1, :]
        return out


class LSTM_bi(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_bi, self).__init__()
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = r_out
        return out

    # Define GRU NETWORK structure


class LSTM_bi_toLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_bi_toLinear, self).__init__()
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = r_out[:, -1, :]
        return out


class LSTM_uni(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_uni, self).__init__()
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, x):
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = r_out
        return out

    # Define GRU NETWORK structure


class LSTM_uni_toLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_uni_toLinear, self).__init__()
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, x):
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = r_out[:, -1, :]
        return out


class var_attention_layer(nn.Module):
    def __init__(self, input_size):
        super(var_attention_layer, self).__init__()
        self.atten = nn.Sequential(nn.Tanh(), nn.Linear(input_size, input_size), nn.Softmax(2))
        self.wei = None

    def forward(self, x):
        self.wei = self.atten(x).mean(0).mean(0)  # 求注意力矩阵
        x = torch.mul(x, self.wei)  # 注意力矩阵与特征矩阵相乘
        return x


# Define GRU NETWORK structure
class lstm_bi_attention(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size, num_layers, pre_length):
        super(lstm_bi_attention, self).__init__()

        self.atten = nn.Sequential(nn.Tanh(), nn.Linear(input_size, input_size), nn.Softmax(2))

        self.rnn_bi = nn.LSTM(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

        self.rnn_bi_to_linear = nn.LSTM(
            input_size=linear_size,
            hidden_size=int(linear_size / 2),
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(int(linear_size), int(linear_size / 2)),
            nn.ReLU(),
            nn.Linear(int(linear_size / 2), int(linear_size / 4)),
            nn.ReLU(),
            nn.Linear(int(linear_size / 4), pre_length)
        )

    def forward(self, x):
        wei = self.atten(x).mean(0).mean(0)  # 求注意力矩阵
        x = torch.mul(x, wei)  # 注意力矩阵与特征矩阵相乘
        r_out, hn = self.rnn_bi(x, None)  # None 表示 hidden state 会用全0的 state
        r_out, hn = self.rnn_bi_to_linear(r_out, None)  # None 表示 hidden state 会用全0的 state
        out = self.fc(r_out[:, -1, :])
        return out, wei
