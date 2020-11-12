import math
import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5)
        self.dense = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden=None):
        x = x.view(x.size(0), 1, -1)
        x, hidden_states = self.lstm(x, hidden)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        out = F.relu(x)

        return out, hidden_states

    def init_hidden(self, x):
        return torch.zeros((2, x.size(0), self.hidden_size), device=device, dtype=dtype)
