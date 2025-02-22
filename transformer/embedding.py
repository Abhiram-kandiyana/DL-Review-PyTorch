import torch
import torch.nn as nn
import math


class Embedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model

        # pre-trained embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):

        # As per the paper, the output (or the weights) are multiplied by the sqrt of d_model
        return self.embedding_layer(token_ids) * math.sqrt(self.d_model)
