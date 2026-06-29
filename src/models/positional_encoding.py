import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len): # max_len = 1000
        super().__init__()

        # 1. Define a learnable parameter of shape (1, max_len, embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # 2. Initialize with a small standard deviation (standard ViT/DeiT convention)
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, x):
        """
        x: (B, T, D) -> Where T is (num_patches + 1) due to the CLS token
        """
        T = x.size(1)

        # 3. Slice the parameter matrix up to the current sequence length T
        return x + self.position_embeddings[:, :T]
