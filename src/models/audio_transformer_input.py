# src/models/audio_transformer_input.py
"""
Audio Transformer input module.

Pipeline:
    (B, n_mels, time)
            │
            ▼
    Patch Embedding
            │
            ▼
    Add CLS Token
            │
            ▼
    Add Learnable Position Embeddings
            │
            ▼
    Input Dropout
            │
            ▼
    (B, num_patches + 1, embed_dim)

This module converts Mel spectrograms into a sequence of transformer
tokens compatible with the encoder.
"""

import torch
import torch.nn as nn

from .patch_embedding import SpectrogramPatchEmbedding
from .positional_encoding import PositionalEncoding


class AudioTransformerInput(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        patch_size: int = 25,
        embed_dim: int = 256,
        max_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = SpectrogramPatchEmbedding(
            n_mels=n_mels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        nn.init.normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(embed_dim, max_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------------------------
        # Patch embedding
        # (B, N, D)
        # -------------------------
        x = self.patch_embed(x)

        # -------------------------
        # Add CLS token
        # (B, N+1, D)
        # -------------------------
        B = x.size(0)

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, x], dim=1)

        # -------------------------
        # Add learnable position embeddings
        # -------------------------
        x = self.pos_enc(x)
        x = self.dropout(x)

        return x
