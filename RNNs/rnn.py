
import torch
import torch.nn as nn


class RNN(nn.Module):
  def __init__(self, x_dim, hidden_dim, y_dim):

    super().__init__()

    self.hidden_dim = hidden_dim

    self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

    self.W_xh = nn.Parameter(torch.randn(x_dim, hidden_dim) * 0.1)

    self.b_h = nn.Parameter(torch.zeros(1, hidden_dim) * 0.1)

    self.W_hy = nn.Parameter(torch.randn(hidden_dim, y_dim) * 0.1)

    self.b_y = nn.Parameter(torch.zeros(1, y_dim) * 0.1)

  def step(self, x, h):

    h = torch.tanh((h @ self.W_hh) + (x @ self.W_xh) + self.b_h)  #

    y = h @ self.W_hy + self.b_y

    return y, h

  def forward(self, X):

    h = torch.zeros(X.shape[0], self.hidden_dim, device = X.device) #(batch_size, hid_dim)

    X = X.transpose(0, 1) # (seq_len, batch_size, x_dim)

    outputs = []

    for i, x in enumerate(X):
      y, h = self.step(x, h)
      outputs.append(y)

    return torch.stack(outputs, dim=1)






if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 64
    vocab_len = 10

    X = torch.randn(batch_size, seq_len, d_model)

    rnn1 = RNN(d_model, 64, 64)
    rnn2 = RNN(64, 128, 128)
    rnn3 = RNN(128, 64, 64)
    rnn4 = RNN(64, 64, vocab_len)

    y1 = rnn1(X)
    y2 = rnn2(y1)
    y3 = rnn3(y2)
    y4 = rnn4(y3)

    output = torch.softmax(y4, dim=2)

    print(output)





