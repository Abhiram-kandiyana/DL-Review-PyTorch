
import torch
import torch.nn as nn


class RNN(nn.Module):
  def __init__(self, x_dim, hidden_dim, batch_size):

    self.hidden_dim = hidden_dim

    self.h = torch.zeros(batch_size, self.hidden_dim)

    self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

    self.W_xh = nn.Parameter(torch.randn(hidden_dim, x_dim) * 0.1)

    self.b_h = nn.Parameter(torch.zeros(1,hidden_dim) * 0.1)

    self.W_hy = nn.Parameter(torch.randn(x_dim, hidden_dim) * 0.1)

    self.b_y = nn.Parameter(torch.zeros(1,x_dim) * 0.1)

  def step(self, x):

    if self.h is None:
        self.h = torch.tanh((self.W_hh @ self.h) + (self.W_xh @ x) + self.b_h)

    y = self.W_hy @ self.h + self.b_y

    return y


# if __name__ == "__main__":
#     batch_size = 2
#     seq_len = 5
#     d_model = 64
#     d_ff = d_model*2
#     n_heads = 8
#
#     X = torch.randn(batch_size, seq_len, d_model)
#
#     op = RNN(d_model,seq_len)
#     encoder_op = op(X)
#
#     print("Output Shape", encoder_op.shape)