import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, p_dropout=0.1, seq_len=128, d_model=512):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_enc = torch.zeros(self.seq_len, self.d_model)
        self.register_buffer("p_enc", self.pos_enc)
        self.dropout = nn.Dropout(p=p_dropout)
        positions = torch.arange(seq_len).unsqueeze(1)
        dims = torch.arange(d_model)
        div_term = 1/torch.pow(10000, dims/self.d_model)
        self.pos_enc = positions * div_term
        self.pos_enc[:, 0::2] = torch.sin(self.pos_enc[:, 0::2])
        self.pos_enc[:, 1::2] = torch.cos(self.pos_enc[:, 1::2])

    def forward(self, x):

        '''

        x: shape = (batch_size, seq_len, d_model)
        self.pos_enc: shape = (seq_len, d_model)
        '''

        x = x + self.pos_enc[:x.shape[1], :].to(x.device)
        return self.dropout(x)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 8
    X = torch.randn(batch_size, seq_len, d_model)

    pos_encodings = PositionalEncoding(seq_len=5, d_model=8)
    x_new = pos_encodings(X)

    print("Output Shape", x_new.shape)
