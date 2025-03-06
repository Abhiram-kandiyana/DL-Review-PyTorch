import torch
import torch.nn as nn
from encoder import EncoderBlock
from decoder import DecoderBlock


class Encoder(nn.Module):

    def __init__(self, d_model, seq_len, n_heads, d_ff):
        super().__init__()
        self.encoder_block_1 = EncoderBlock(d_model, seq_len, n_heads, d_ff)
        self.encoder_block_2 = EncoderBlock(d_model, seq_len, n_heads, d_ff)
        self.encoder_block_3 = EncoderBlock(d_model, seq_len, n_heads, d_ff)

    def forward(self, x):
        x = self.encoder_block_1(x)
        x = self.encoder_block_2(x)
        z = self.encoder_block_3(x)

        k, v = self.encoder_block_3.attn.K, self.encoder_block_3.attn.V

        return z, k, v


class Decoder(nn.Module):

    def __init__(self, d_model, seq_len, n_heads, d_ff):
        super().__init__()
        self.decoder_block_1 = DecoderBlock(d_model, seq_len, n_heads, d_ff)
        self.decoder_block_2 = DecoderBlock(d_model, seq_len, n_heads, d_ff)
        self.decoder_block_3 = DecoderBlock(d_model, seq_len, n_heads, d_ff)

    def forward(self, x, enc_k, enc_v):
        x = self.decoder_block_1(x, enc_k, enc_v)
        x = self.decoder_block_2(x, enc_k, enc_v)
        z = self.decoder_block_3(x, enc_k, enc_v)

        return z


# class Transformer(nn.Module):
#
#     def __init__(self):
