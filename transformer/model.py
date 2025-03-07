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


class Transformer(nn.Module):

    def __init__(self, d_model, seq_len, n_heads, d_ff, vocab_len):
        super.__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_len, d_model)

        self.encoder = Encoder(d_model, seq_len, n_heads, d_ff)
        self.decoder = Decoder(d_model, seq_len, n_heads, d_ff)
        self.output = nn.Linear(d_model, vocab_len)


    def forward(self, x, max_len=50, sos_token=1, eos_token=0, pad_token=2):

        batch_size = x.shape[0]

        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model))

        encoder_y, encoder_k, encoder_v = self.encoder(x)

        start_token_embedding = self.embedding(torch.tensor([[sos_token] for _ in batch_size],device=x.device))

        pad_token_embedding = self.embedding(torch.tensor([[pad_token]], device=x.device))

        pad_token_embedding = pad_token_embedding[0]

        op_tokens = [start_token_embedding]

        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(max_len):

            decoder_input = torch.cat(op_tokens, dim=1) # concatenate along the sequence

            y = self.decoder(decoder_input, encoder_k, encoder_v)
            logits = self.output(y[:, -1, :])
            predicted_token = torch.argmax(logits, dim=-1, keepdim=True)

            predicted_token_embedding = self.embedding(predicted_token)

            predicted_token_embedding = torch.where(finished.unsqueeze(-1), pad_token_embedding, predicted_token_embedding)

            op_tokens.append(predicted_token_embedding)

            for i in range(batch_size):
                if predicted_token[i] == eos_token:
                    finished[i] = 1

            if (finished == 1).all():
                break



        return torch.cat(op_tokens, dim=1)

#
# if __name__ == "__main__":
#     with torch.no_grad():
#         model = Transformer()
#         output = model(x)

