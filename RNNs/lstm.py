import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, x_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        # input gate
        self.W_hi = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_xi = nn.Parameter(torch.randn(x_dim, hidden_dim) * 0.1)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        # forget gate
        self.W_hf = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_xf = nn.Parameter(torch.randn(x_dim, hidden_dim) * 0.1)
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        # output gate
        self.W_ho = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_xo = nn.Parameter(torch.randn(x_dim, hidden_dim) * 0.1)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        # update weights
        self.W_hz = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.W_xz = nn.Parameter(torch.randn(x_dim, hidden_dim) * 0.1)
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))

    def step(self, x, h, c):
        i = torch.sigmoid((h @ self.W_hi) + (x @ self.W_xi) + self.b_i)
        f = torch.sigmoid((h @ self.W_hf) + (x @ self.W_xf) + self.b_f)
        o = torch.sigmoid((h @ self.W_ho) + (x @ self.W_xo) + self.b_o)
        z = torch.tanh((h @ self.W_hz) + (x @ self.W_xz) + self.b_z)

        c = (i * z) + (f * c)
        h = o * torch.tanh(c)

        return h, c

    def forward(self, X):
        h = torch.zeros(X.shape[0], self.hidden_dim, device=X.device)  # (batch_size, hid_dim)
        c = torch.zeros(X.shape[0], self.hidden_dim, device=X.device)

        X = X.transpose(0, 1)  # (seq_len, batch_size, x_dim)

        outputs = []

        for x in X:
            h, c = self.step(x, h, c)
            outputs.append(h)

        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 64
    vocab_len = 10

    X = torch.randn(batch_size, seq_len, d_model)

    lstm1 = LSTM(d_model, 64)
    lstm2 = LSTM(64, 128)
    lstm3 = LSTM(128, 64)
    lstm4 = LSTM(64, 64)

    y1 = lstm1(X)
    y2 = lstm2(y1)
    y3 = lstm3(y2)
    y4 = lstm4(y3)

    op_layer = nn.Linear(64, vocab_len)

    y = op_layer(y4)
    output = torch.softmax(y, dim=2)

    print(output.shape)
