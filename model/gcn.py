from torch.nn.parameter import Parameter
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        support = torch.bmm(input, self.weight.repeat(input.size(0), 1, 1))
        output = torch.bmm(adj, support)
        output = F.relu(output + self.bias)
        return output


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1 = nn.GRU(90, 256, 2, batch_first=True)  # 128

    def forward(self, x, hn=None):
        x, hn = self.rnn1(x, hn)
        hn = nn.Tanh()(hn)
        return x, hn


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(1024, 10)  # 256 or 256 * 16

    def forward(self, hn):
        hn = self.fc1(hn)
        return hn
