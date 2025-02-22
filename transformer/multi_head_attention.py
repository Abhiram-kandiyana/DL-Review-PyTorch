import torch
import torch.nn as nn
import os


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=512, W_q=None, W_v=None, W_k=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Parameter(W_q if W_q is not None else torch.randn(d_model, n_heads * self.d_head))
        self.W_k = nn.Parameter(W_k if W_k is not None else torch.randn(d_model, n_heads * self.d_head))
        self.W_v = nn.Parameter(W_v if W_v is not None else torch.randn(d_model, n_heads * self.d_head))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, X):
        self.Q = X @ self.W_q  # (batch_size, seq_len, num_heads * d_head)
        self.K = X @ self.W_k  # (batch_size, seq_len, num_heads * d_head)
        self.V = X @ self.W_v  # (batch_size, seq_len, num_heads * d_head)

        self.Q = self.Q.reshape((self.Q.shape[0], self.Q.shape[1],
                                 self.n_heads, self.d_head))  # (batch_size, seq_len, num_heads, d_head)
        self.K = self.K.reshape((self.Q.shape[0], self.Q.shape[1],
                                 self.n_heads, self.d_head))  # (batch_size, seq_len, num_heads, d_head)
        self.V = self.V.reshape((self.Q.shape[0], self.Q.shape[1],
                                 self.n_heads, self.d_head))  # (batch_size, seq_len, num_heads, d_head)

        self.Q = self.Q.permute(2, 0, 1, 3)  # (num_heads, batch_size, seq_len, d_head)
        self.K = self.K.permute(2, 0, 1, 3)  # (num_heads, batch_size, seq_len, d_head)
        self.V = self.V.permute(2, 0, 1, 3)  # (num_heads, batch_size, seq_len, d_head)

        self.scores = torch.softmax(
            (self.Q @ self.K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float)),
            dim=-1)  # (num_heads, batch_size, seq_len, seq_len)
        output = self.scores @ self.V  # (num_heads, batch_size, seq_len, d_head)

        # instead of dividing this into multiple outputs (one per head) and concatentating
        output = (
            output.permute(1, 2, 0, 3)  # (batch_size, seq_len, num_heads, d_head)
            .contiguous()
            .view(X.shape[0], -1, self.n_heads * self.d_head)  # (batch_size, seq_len, n_heads * d_head)
        )

        output = output @ self.W_o  # (batch_size, seq_len, n_heads * d_head)

        return output


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 512
    X = torch.randn(batch_size, seq_len, d_model)

    multi_head_att = MultiHeadAttention()

    output = multi_head_att(X)

    print("Output Shape", output.shape)
