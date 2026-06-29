import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .positional_encoding import PositionalEncoding

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """
        x: Decoder input (Target)
        value/key: Encoder output
        """
        # 1. Masked Self-Attention (Query, Key, Value all from decoder input 'x')
        # trg_mask ensures the decoder doesn't "see" future tokens
        attention = self.attention(x, x, x, trg_mask)

        # 2. Cross-Attention
        # Query: output of self-attention (x)
        # Key & Value: from the Encoder
        # src_mask: mask for the encoder sequence (padding)
        out = self.transformer_block(value, key, attention, src_mask)

        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_len=max_length)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        x: (B, trg_seq_len) - Target indices
        enc_out: Output from the Encoder
        """
        # Convert indices to embeddings and add positional info
        x = self.dropout(self.position_embedding(self.word_embedding(x)))

        for layer in self.layers:
            # Pass through decoder blocks
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        # Final linear layer to get logits over vocabulary
        out = self.fc_out(x)

        return out
