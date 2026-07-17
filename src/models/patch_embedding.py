# src/models/patch_embedding.py

import torch
import torch.nn as nn

class SpectrogramPatchEmbedding(nn.Module):
    def __init__(self, n_mels, patch_size, embed_dim): # (128, 25, 256)
        super().__init__()

        self.n_mels = n_mels
        self.patch_size = patch_size

        # each patch is flattened then projected
        self.proj = nn.Linear(n_mels * patch_size, embed_dim)

    def forward(self, x):
        """
        x: (B, n_mels, time)
        returns: (B, num_patches, embed_dim)
        """

        B, M, T = x.shape

        # ensure divisible
        T_trim = (T // self.patch_size) * self.patch_size
        x = x[:, :, :T_trim]

        # reshape into patches
        x = x.reshape(
            B,
            M,
            T_trim // self.patch_size,
            self.patch_size
        )

        # now: (B, M, num_patches, patch_size)

        # move patch dim next to mel
        x = x.permute(0, 2, 1, 3)
        # (B, num_patches, M, patch_size)

        # flatten each patch
        x = x.reshape(B, -1, M * self.patch_size)
        # (B, num_patches, patch_dim)

        # linear projection
        x = self.proj(x)

        return x
