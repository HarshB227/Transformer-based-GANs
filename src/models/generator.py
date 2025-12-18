# src/models/generator.py

import torch
import torch.nn as nn


class MusicTransformerGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        genre_count: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 4,
        latent_dim: int = 64,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.latent_to_emb = nn.Linear(latent_dim, d_model)
        self.genre_emb = nn.Embedding(genre_count, d_model)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, z, genre_ids, tokens):
        B, L = tokens.shape

        cond = self.latent_to_emb(z) + self.genre_emb(genre_ids)

        tok_emb = self.token_emb(tokens)
        pos = torch.arange(L, device=tokens.device)
        pos_emb = self.pos_emb(pos)[None, :, :]

        dec_in = tok_emb + pos_emb
        memory = cond.unsqueeze(1)

        mask = torch.triu(
            torch.ones(L, L, device=tokens.device) * float("-inf"),
            diagonal=1,
        )

        out = self.decoder(dec_in, memory, tgt_mask=mask)
        return self.fc_out(out)

    @torch.no_grad()
    def generate(self, z, genre_ids, max_len=512, start_token=257, end_token=258):
        device = z.device
        B = z.size(0)

        seq = torch.full((B, 1), start_token, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(z, genre_ids, seq)
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == end_token).all():
                break

        return seq
