# src/models/discriminator.py

import torch
import torch.nn as nn


class MusicTransformerDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        genre_count: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.genre_emb = nn.Embedding(genre_count, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.real_fake_head = nn.Linear(d_model, 1)
        self.genre_head = nn.Linear(d_model, genre_count)

    def forward(self, tokens, genre_ids):
        """
        tokens:    (B, L)
        genre_ids: (B,)
        """
        B, L = tokens.size()

        tok_emb = self.token_emb(tokens)                # (B, L, d_model)
        pos = torch.arange(L, device=tokens.device)
        pos_emb = self.pos_emb(pos)[None, :, :]         # (1, L, d_model)

        genre_emb = self.genre_emb(genre_ids).unsqueeze(1)  # (B, 1, d_model)

        x = tok_emb + pos_emb + genre_emb               # (B, L, d_model)
        enc = self.encoder(x)                           # (B, L, d_model)

        pooled = enc.mean(dim=1)                        # (B, d_model)

        real_fake_logits = self.real_fake_head(pooled)  # (B, 1)
        genre_logits = self.genre_head(pooled)          # (B, G)

        return real_fake_logits, genre_logits
