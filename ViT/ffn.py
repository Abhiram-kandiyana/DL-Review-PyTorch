import torch
import torch.nn as nn
from torch.nn.functional import gelu


class Ffn(nn.Module):
    def __init__(self, d_ff=2048, d_model=512, p_dropout = 0.1):
        super().__init__()

        # linear layers which treat each position as an individual input.
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):

        x = self.w1(x)
        x = gelu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x
