import torch
import torch.nn as nn
import os

class SelfAttention(nn.Module):
    def __init__(self, d_model = 512, W_q = None, W_v = None, W_k = None):
        super().__init__()
        self.W_q = nn.parameter.Parameter(W_q if W_q is not None else torch.randn(d_model, d_model))
        self.W_v = nn.parameter.Parameter(W_v if W_v is not None else torch.randn(d_model, d_model))
        self.W_k = nn.parameter.Parameter(W_k if W_k is not None else torch.randn(d_model, d_model))


    def forward(self, X):
        self.Q = X @ self.W_q
        self.K = X @ self.W_k
        self.V = X @ self.W_v

        self.d_k = self.K.shape[-1]
        self.scores = torch.softmax((self.Q @ self.K.transpose(-2, -1))/torch.sqrt(torch.tensor(self.d_k, dtype = torch.float32)),dim=-1)

        output = self.scores @ self.V

        return output


batch_size = 2
seq_len = 5
d_model = 512

X = torch.randn(batch_size,seq_len,d_model)

self_att = SelfAttention(d_model)

output = self_att(X)

print("Output Shape", output.shape)




    # def forward(self, X):


