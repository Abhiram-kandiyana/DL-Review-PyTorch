import torch
import torch.nn as nn
import sys
sys.path.append('/Users/abhiramkandiyana/LLMsFromScratch/transformer')
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from ffn import Ffn


class EncoderBlock(nn.Module):
    def __init__(self, d_model, seq_len, n_heads, d_ff, p_ffn_dropout=0.1, p_pos_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.ffn = Ffn(d_ff=d_ff, d_model=d_model, p_dropout=p_ffn_dropout)
        self.init_norm = nn.LayerNorm(d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_norm = self.init_norm(x)

        # multi-head self attention with residual connection (post norm) applied
        z = self.attn(x_norm) + x

        z_norm = self.attn_norm(z)

        # FFN applied to each token (position) in the input
        ffn_op = self.last_norm(self.ffn(z_norm) + z)

        return ffn_op


if __name__ == "__main__":

    batch_size = 2
    seq_len = 5
    d_model = 64
    d_ff = d_model*2
    n_heads = 8

    X = torch.randn(batch_size, seq_len, d_model)

    op = EncoderBlock(d_model,seq_len,n_heads,d_ff)
    encoder_op = op(X)

    print("Output Shape", encoder_op.shape)
