import torch
import torch.nn as nn

from .patch_embedding import SpectrogramPatchEmbedding
from .positional_encoding import PositionalEncoding


class AudioTransformerInput(nn.Module):

    def __init__(
        self,
        n_mels = 128,
        patch_size = 25,
        embed_dim = 256,
        dropout = 0.1
    ):
        super().__init__()

        self.patch_embed = SpectrogramPatchEmbedding(
            n_mels=n_mels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        nn.init.normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

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
        # Position embeddings
        # -------------------------
        x = self.pos_enc(x)
        x = self.dropout(x)

        return x
