import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, p_dropout=0.1, seq_len=128, d_model=512):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=p_dropout)

        # initialize pos_enc matrix with zeroes
        self.pos_enc = torch.zeros(self.seq_len, self.d_model)

        # add pos_enc to register so Pytorch knows that it belongs to this class even if it doesn't require grads
        self.register_buffer("p_enc", self.pos_enc)

        # get a tensor of all the positions in the sequence.
        # Unsqueeze positions so it can broadcasted when multiplied with dims
        positions = torch.arange(seq_len).unsqueeze(1)

        # get a tensor of all the dimensions
        dims = torch.arange(d_model)

        # calculate the common denominator once so you can directly use it for all positions
        div_term = 1/torch.pow(10000, dims/self.d_model)

        self.pos_enc = positions * div_term

        # apply sin to even positions (starting from 0) and cos to odd positions.
        self.pos_enc[:, 0::2] = torch.sin(self.pos_enc[:, 0::2])
        self.pos_enc[:, 1::2] = torch.cos(self.pos_enc[:, 1::2])

    def forward(self, x):

        '''

        x: shape = (batch_size, seq_len, d_model)
        self.pos_enc: shape = (seq_len, d_model)
        '''

        # move pos_enc to the GPU where X is before adding it to X
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
