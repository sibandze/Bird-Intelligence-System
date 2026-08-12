# src/models/supervised_transformer.py
"""
Mel Spectrogram
        │
        ▼
AudioTransformerInput
        │
        ▼
Transformer Encoder
        │
        ▼
CLS Token (out[:, 0])
        │
        ▼
LayerNorm
        │
        ▼
Linear
        │
        ▼
GELU
        │
        ▼
Dropout
        │
        ▼
Linear
        │
        ▼
234 Logits
"""
import torch
import torch.nn as nn
from .encoder import Encoder

class SupervisedTransformer(nn.Module):
    """
    Supervised Transformer Classifier for Bird Species ID.
    
    Architecture: Mel -> PatchEmbed -> Transformer -> CLS -> MLP Head -> Logits
    """
    def __init__(
        self,
        config: dict,
        num_classes: int,
        device: str = "cpu"
    ):
        super().__init__()
        audio_cfg = config['audio']
        model_cfg = config['model']

        n_mels = audio_cfg['n_mels']
        patch_size = model_cfg['patch_size']
        embed_dim = model_cfg['embed_dim']
        segment_size = audio_cfg['segment_size']

        num_patches = segment_size // patch_size
        max_len = num_patches + 1 + 10 # +1 CLS, + buffer

        self.encoder = Encoder(
            n_mels=n_mels,
            patch_size=patch_size,
            embed_size=embed_dim,
            num_layers=model_cfg['num_layers'],
            heads=model_cfg['heads'],
            device=device,
            forward_expansion=model_cfg['forward_expansion'],
            dropout=model_cfg['dropout'],
            max_len=max_len,
        )

        # Head: CLS -> LN -> Linear -> GELU -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(model_cfg['dropout']),
            nn.Linear(embed_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        # ViT style init for head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        x: (B, n_mels, time)  e.g. (B, 128, 187)
        mask: (B, 1, 1, seq_len) optional attention mask
        returns: (B, num_classes) logits
        """
        # Encoder out: (B, N_patches+1, embed_dim)
        enc_out = self.encoder(x, mask)

        # CLS token is first token
        cls_token = enc_out[:, 0] # (B, embed_dim)

        logits = self.head(cls_token) # (B, num_classes)
        return logits
