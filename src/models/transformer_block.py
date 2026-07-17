# src/models/transformer_block.py

import torch.nn as nn
from .self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """
        Args:
            embed_size: Dimensionality of the input embeddings
            heads: Number of attention heads
            dropout: Dropout probability
            forward_expansion: Factor to expand the hidden dimension in the FeedForward layer (usually 4)
        """
        super(TransformerBlock, self).__init__()

        # 1. Modified: Pass dropout to SelfAttention
        self.attention = SelfAttention(embed_size, heads, dropout=dropout)

        # Layer Normalization for both sub-layers
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed Forward Network (Position-wise)
        # 2. Modified: Changed nn.ReLU() to nn.GELU()
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(), 
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # 1. Multi-Head Self Attention
        attention = self.attention(value, key, query, mask)

        # 2. Add & Norm (Residual Connection + LayerNorm)
        x = self.dropout(self.norm1(attention + query))

        # 3. Feed Forward Network
        forward = self.feed_forward(x)

        # 4. Add & Norm (Second Residual Connection)
        out = self.dropout(self.norm2(forward + x))

        return out
