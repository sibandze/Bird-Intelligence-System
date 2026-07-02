import torch
import torch.nn as nn
from .audio_transformer_input import AudioTransformerInput
from .transformer_block import TransformerBlock

class Encoder(nn.Module):
    def __init__(
        self,
        n_mels,
        patch_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_len = 1000,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        # Initial processing: Spectrogram -> Patches -> Embeddings + Positional Encoding
        self.input_layer = AudioTransformerInput(
            n_mels=n_mels,
            patch_size=patch_size,
            embed_dim=embed_size,
            max_len = max_len,
            dropout = dropout,
        )

        # Stack of Transformer Blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: (B, n_mels, time)
        mask: mask for the attention mechanism
        """
        # (B, num_patches, embed_size)
        out = self.dropout(self.input_layer(x))

        # Pass through each transformer block
        for layer in self.layers:
            # In Encoder, Query, Key, and Value are all the same (Self-Attention)
            out = layer(out, out, out, mask)

        return out
