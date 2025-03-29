import torch
import torch.nn as nn
import sys
from encoder import EncoderBlock
sys.path.append('/Users/abhiramkandiyana/LLMsFromScratch/transformer')
from decoder import DecoderBlock
from embedding import Embedding
from ffn import Ffn


class Encoder(nn.Module):

    def __init__(self, d_model, seq_len, n_layers, n_heads, d_ff):
        super().__init__()

        self.encoder_blocks = [EncoderBlock(d_model, seq_len, n_heads, d_ff) for _ in range(n_layers)]

    def forward(self, x):

        z = x
        for encoder_block in self.encoder_blocks:
            z = encoder_block(z)



        # k, v = self.encoder_block_3.attn.K, self.encoder_block_3.attn.V

        return z

class VisionTransformer(nn.Module):

    def __init__(self, d_model, seq_len, n_layers, n_heads, d_ff, class_len, pre_training = False):
        super().__init__()
        self.d_model = d_model
        self.embedding = Embedding(class_len, d_model)
        self.pre_training = pre_training
        self.encoder = Encoder(d_model, seq_len, n_layers, n_heads, d_ff)
        self.mlp = Ffn(d_ff,d_model)
        self.output = nn.Linear(d_model, class_len)


    def forward(self, x):

        batch_size = x.shape[0]

        x = self.embedding(x)

        y = self.encoder(x)

        if self.pre_training:
            y = self.mlp(y)

        cls_token_repr = y[:, 0, :]

        logits = self.output(cls_token_repr)

        return logits

#
# if __name__ == "__main__":
#     with torch.no_grad():
#         d_model
#         model = Transformer()
#         output = model(x)

