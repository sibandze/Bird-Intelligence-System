# src/models/encoders/cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    CNN encoder for mel spectrogram inputs.

    Takes mel spectrograms of shape [B, n_mels, time],
    then internnally adds single channel dimmension for Conv2d
    and produces embeddings in multiple formats:

    - forward(): Pooled 1D embedding [B, embed_dim] (for classification/SSL)
    - forward_features(): Full spatial-temporal feature map [B, C, H, W]
    - forward_sequence(): Temporal token sequence [B, S, D] (for transformers)

    Architecture:
        4 convolutional blocks with increasing channels
        → Multiple output pathways
    """

    def __init__(
        self,
        n_mels: int = 128,
        embed_dim: int = 512,
        base_channels: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_mels: Number of mel frequency bins in input
            embed_dim: Output embedding dimension for pooled forward()
            base_channels: Base number of channels (doubled each block)
            dropout: Dropout rate after conv blocks
        """
        super().__init__()

        self.n_mels = n_mels
        self.embed_dim = embed_dim

        # Conv Block 1: [B, 1, 128, T] → [B, 64, 64, T/2]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Conv Block 2: [B, 64, 64, T/2] → [B, 128, 32, T/4]
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Conv Block 3: [B, 128, 32, T/4] → [B, 256, 16, T/8]
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Conv Block 4: [B, 256, 16, T/8] → [B, 512, 8, T/16]
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Global average pooling over frequency for sequence/forward
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Adaptive pooling over time to fixed size for forward()
        self.time_pool = nn.AdaptiveAvgPool2d((1, 8))

        self.dropout = nn.Dropout(dropout)

        # Project to embedding dimension for forward()
        # After pooling: base_channels*8 * 8 time steps = 512*8 = 4096
        self.embed = nn.Sequential(
            nn.Linear(base_channels * 8 * 8, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram [B, n_mels, time_steps]

        Returns:
            Feature map [B, channels, n_mels/16, time_steps/16]
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, n_mels, time], got {tuple(x.shape)}"
            )

        x = x.unsqueeze(1)  # [B, 1, n_mels, time]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns temporal token sequence for transformer models.

        Pools frequency dimension to 1, then creates a sequence
        of temporal tokens.

        Args:
            x: Mel spectrogram [B, n_mels, time_steps]

        Returns:
            Token sequence [B, S, D] where:
                S = time_steps / 16
                D = base_channels * 8 = 512
        """
        x = self.forward_features(x)  # [B, 512, 8, T/16]
        x = self.freq_pool(x)  # [B, 512, 1, T/16]
        x = x.squeeze(2)  # [B, 512, T/16]
        return x.transpose(1, 2)  # [B, T/16, 512]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard pooled forward pass returning 1D embedding.

        Pools both frequency and time dimensions to fixed size,
        then projects to embed_dim.

        Args:
            x: Mel spectrogram [B, n_mels, time_steps]

        Returns:
            h: Embedding [B, embed_dim]
        """
        x = self.forward_features(x)  # [B, 512, 8, T/16]
        x = self.freq_pool(x)  # [B, 512, 1, T/16]
        x = self.time_pool(x)  # [B, 512, 1, 8]
        x = x.view(x.size(0), -1)  # [B, 4096]
        x = self.dropout(x)
        return self.embed(x)  # [B, embed_dim]

    def get_output_dim(self) -> int:
        """Return the output embedding dimension for forward()."""
        return self.embed_dim

    def get_feature_dim(self) -> int:
        """Return the feature map channel dimension."""
        return self.conv4[-3].num_features  # 512 for default

    def get_sequence_dim(self) -> int:
        """Return the sequence token dimension."""
        return self.conv4[-3].num_features  # 512 for default
