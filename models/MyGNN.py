import math
import torch
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.zeros((input_size, output_size), device=device, dtype=dtype), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size, device=device, dtype=dtype), requires_grad=True)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, A):
        x = torch.einsum("ijk, kl->ijl", [x, self.weight])
        x = torch.einsum("ij, kjl->kil", [A, x])
        x = x + self.bias

        return x


class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_size, hidden_size)
        self.gcn2 = GraphConvolution(hidden_size, output_size)
        # self.gcn = GraphConvolution(input_size, output_size)

    def forward(self, x, A):
        x = self.gcn1(x, A)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.gcn2(x, A)
        x = F.relu(x)
        # x = self.gcn(x, A)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gcn = GCN(input_size=1, hidden_size=128, output_size=1)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5)

    def forward(self, x, A, hidden=None):
        # batch_size, timestep, N = x.size()
        # gcn_in = x.view((batch_size * timestep, -1))
        # gcn_out = self.gcn(gcn_in, A)
        # encoder_in = gcn_out.view((batch_size, timestep, -1))
        gcn_in = x.view((x.size(0), x.size(1), 1))
        gcn_out = self.gcn(gcn_in, A)
        encoder_in = gcn_out.view((x.size(0), 1, x.size(1)))
        encoder_out, encoder_states = self.lstm(encoder_in, hidden)

        return encoder_out, encoder_states

    def init_hidden(self, x):
        return torch.zeros((2, x.size(0), self.hidden_size), device=device, dtype=dtype)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
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
        x, decoder_states = self.lstm(x, hidden)
        x = x.view(x.size(0), -1)
        # x = F.relu(x)
        x = self.dense(x)
        decoder_out = F.relu(x)

        return decoder_out, decoder_states

    def init_hidden(self, x):
        return torch.zeros((2, x.size(0), self.hidden_size), device=device, dtype=dtype)
