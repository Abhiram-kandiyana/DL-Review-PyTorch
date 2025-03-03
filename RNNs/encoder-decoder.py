import torch
import torch.nn as nn
import lstm
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_dim):

        super().__init__()

        self.lstm1 = lstm.LSTM(x_dim,hidden_dim)
        self.lstm2 = lstm.LSTM(hidden_dim, 2*hidden_dim)
        self.lstm3 = lstm.LSTM(2*hidden_dim, 2*hidden_dim)


    def forward(self, x):

        h1, _ = self.lstm1(x)
        h2, _ = self.lstm2(h1)
        h3, c3 = self.lstm3(h2)

        return h3,c3

#inference time Decoder
class DecoderLSTM(nn.Module):
    def __init__(self, x_dim, seq_len, hidden_dim, vocab_size):
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

        self.linear_list = clones(self.Linear(hidden_dim, vocab_size), seq_len)





    def step(self, x, h, c):
        i = torch.sigmoid((h @ self.W_hi) + (x @ self.W_xi) + self.b_i)
        f = torch.sigmoid((h @ self.W_hf) + (x @ self.W_xf) + self.b_f)
        o = torch.sigmoid((h @ self.W_ho) + (x @ self.W_xo) + self.b_o)
        z = torch.tanh((h @ self.W_hz) + (x @ self.W_xz) + self.b_z)

        c = (i * z) + (f * c)
        h = o * torch.tanh(c)

        return h, c

    def forward(self, x, h, c):

        x = x.transpose(0, 1)  # (seq_len, batch_size, x_dim)

        outputs = []
        cell_states = []

        x_first = x[0]
        h, c = self.step(x_first, h, c)
        y = torch.softmax(self.linear_list[0](h), dim=-1)
        x_curr = torch.argmax(y, dim=-1)
        x_curr = embedding_layer(x_curr)
        outputs.append(y)
        cell_states.append(c)


        for t in range(x[1:].shape[0]):
            h, c = self.step(x_curr, h, c)
            y = torch.softmax(self.linear_list[t](h), dim=-1)
            x_curr = torch.argmax(y, dim=-1)
            x_curr = embedding_layer(x_curr)
            outputs.append(y)
            cell_states.append(c)

        return torch.stack(outputs, dim=1), torch.stack(cell_states, dim=1)










