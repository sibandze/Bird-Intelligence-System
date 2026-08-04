# src/models/encoders/cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    CNN encoder for spectrogram inputs.
    
    Takes mel spectrograms of shape [B, 1, n_mels, time] 
    and produces flattened embeddings of shape [B, embed_dim].
    
    Architecture:
        4 convolutional blocks with increasing channels
        → Global average pooling over frequency dimension
        → Adaptive pooling over time dimension
        → Flatten to embedding
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
            embed_dim: Output embedding dimension
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
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        
        # Conv Block 3: [B, 128, 32, T/4] → [B, 256, 16, T/8]
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        
        # Conv Block 4: [B, 256, 16, T/8] → [B, 512, 8, T/16]
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        
        # Global average pooling over frequency
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Adaptive pooling over time to fixed size
        self.time_pool = nn.AdaptiveAvgPool2d((1, 8))
        
        self.dropout = nn.Dropout(dropout)
        
        # Project to embedding dimension
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram [B, 1, n_mels, time_steps]
            
        Returns:
            h: Embedding [B, embed_dim]
        """
        # Conv blocks
        x = self.conv1(x)   # [B, 64, 64, T/2]
        x = self.conv2(x)   # [B, 128, 32, T/4]
        x = self.conv3(x)   # [B, 256, 16, T/8]
        x = self.conv4(x)   # [B, 512, 8, T/16]
        
        # Pool frequency to 1
        x = self.freq_pool(x)   # [B, 512, 1, T/16]
        
        # Pool time to fixed size
        x = self.time_pool(x)   # [B, 512, 1, 8]
        
        # Flatten
        x = x.view(x.size(0), -1)   # [B, 4096]
        
        # Dropout and project
        x = self.dropout(x)
        h = self.embed(x)   # [B, embed_dim]
        
        return h
    
    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.embed_dim
