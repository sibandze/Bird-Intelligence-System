# src/models/heads/projection.py

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Maps encoder embeddings to a lower-dimensional space 
    where contrastive loss is applied.
    
    Architecture: Linear → BN → ReLU → Linear → BN
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
    ):
        """
        Args:
            input_dim: Dimension of encoder output h
            hidden_dim: Hidden layer dimension
            output_dim: Projection space dimension z
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            h: Encoder embeddings [B, input_dim]
            
        Returns:
            z: Projected embeddings [B, output_dim]
        """
        return self.projection(h)
