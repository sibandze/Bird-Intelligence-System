import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.0):
        super(SelfAttention, self).__init__()
        assert embed_size % heads == 0, "embed_size must be divisible by heads"

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Project to Q, K, V for all heads at once
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys   = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries= nn.Linear(self.embed_size, self.embed_size, bias=False)

        # 1. Added: Attention Dropout Layer
        self.dropout = nn.Dropout(dropout)

        # Final projection back to embed_size
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, query, mask=None):
        """
        Args:
            values, keys, query: [N, seq_len, embed_size]
            mask: [N, 1, 1, seq_len] or [N, 1, seq_len, seq_len]
        Returns:
            out: [N, seq_len, embed_size]
        """
        N, seq_len, _ = query.shape

        # Linear projections -> [N, seq_len, embed_size]
        V = self.values(values)
        K = self.keys(keys)
        Q = self.queries(query)

        # Split into heads: [N, seq_len, heads, head_dim] -> [N, heads, seq_len, head_dim]
        V = V.view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        Q = Q.view(N, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Q @ K^T -> [N, heads, seq_len, seq_len]
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e10"))

        attention = torch.softmax(energy, dim=-1)  # [N, heads, seq_len, seq_len]

        # 2. Added: Apply dropout directly to the weights of the attention map
        attention = self.dropout(attention)

        # Weighted sum of values -> [N, heads, seq_len, head_dim]
        out = torch.matmul(attention, V)

        # Concat heads: [N, seq_len, heads, head_dim] -> [N, seq_len, embed_size]
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.embed_size)

        # Final linear projection
        out = self.fc_out(out)
        return out
