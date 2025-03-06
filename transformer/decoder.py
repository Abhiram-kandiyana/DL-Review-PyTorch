import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from ffn import Ffn
from encoder import EncoderBlock


class DecoderBlock(nn.Module):
    def __init__(self, d_model, seq_len, n_heads, d_ff, p_ffn_dropout=0.1, p_pos_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model, apply_mask=True)
        self.cross_attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.pos = PositionalEncoding(seq_len=seq_len, d_model=d_model, p_dropout=p_pos_dropout)
        self.ffn = Ffn(d_ff=d_ff, d_model=d_model, p_dropout=p_ffn_dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_k, encoder_v):
        # adding positional encoding to the input
        pos_x = self.pos(x)

        # multi-head self attention with residual connection (post norm) applied
        z = self.attn_norm(self.attn(pos_x) + pos_x)

        z = self.cross_attn_norm(self.cross_attn(z, encoder_k, encoder_v) + z)

        # FFN applied to each token (position) in the input
        ffn_op = self.ffn_norm(self.ffn(z) + z)

        return ffn_op


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 64
    d_ff = d_model*2
    n_heads = 8

    X = torch.randn(batch_size, seq_len, d_model)

    encoder_obj = EncoderBlock(d_model, seq_len, n_heads, d_ff)
    encoder_op = encoder_obj(X)


    decoder_obj = DecoderBlock(d_model,seq_len,n_heads,d_ff, encoder_obj.attn.W_v, encoder_obj.attn.W_k)
    decoder_op = decoder_obj(X)

    print("Output Shape", decoder_op.shape)
