# src/models/ssl/simclr.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import CNNEncoder
from src.models.heads import ProjectionHead


class SimCLR(nn.Module):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
    
    Adapted for audio spectrograms.
    Uses NT-Xent (normalized temperature-scaled cross entropy) loss.
    
    Architecture:
        x → Encoder f → h → Projection g → z
    """
    
    def __init__(
        self,
        encoder: nn.Module = None,
        projection: nn.Module = None,
        temperature: float = 0.1,
        **encoder_kwargs,
    ):
        """
        Args:
            encoder: Pre-built encoder (if None, creates CNNEncoder)
            projection: Pre-built projection head (if None, creates ProjectionHead)
            temperature: Temperature parameter for NT-Xent loss
            **encoder_kwargs: Arguments for CNNEncoder if encoder is None
        """
        super().__init__()
        
        if encoder is None:
            encoder = CNNEncoder(**encoder_kwargs)
        
        if projection is None:
            encoder_dim = encoder.get_output_dim() if hasattr(encoder, 'get_output_dim') else encoder_kwargs.get('embed_dim', 512)
            projection = ProjectionHead(
                input_dim=encoder_dim,
                hidden_dim=256,
                output_dim=128,
            )
        
        self.encoder = encoder
        self.projection = projection
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning projected embeddings.
        
        Args:
            x: Input spectrograms [B, 1, n_mels, time]
            
        Returns:
            z: Projected embeddings [B, proj_dim]
        """
        h = self.encoder(x)
        z = self.projection(h)
        return z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoder embeddings only (without projection).
        Useful for downstream tasks.
        
        Args:
            x: Input spectrograms [B, 1, n_mels, time]
            
        Returns:
            h: Encoder embeddings [B, embed_dim]
        """
        return self.encoder(x)
    
    def contrastive_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent contrastive loss.
        
        Args:
            z: Concatenated views [2*B, proj_dim]
                First B are view1, next B are view2
        
        Returns:
            loss: Scalar contrastive loss
        """
        B = z.shape[0] // 2
        
        # Normalize projections
        z = F.normalize(z, dim=1)
        
        # Compute similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]
        
        # Positive pairs: (i, i+B) and (i+B, i) for i in [0, B)
        # Remove diagonal (self-similarity)
        sim = sim - torch.eye(2*B, device=z.device) * 1e9
        
        # Extract positive similarities
        pos_sim = torch.cat([
            sim[:B, B:].diag(),  # view1[i] vs view2[i]
            sim[B:, :B].diag(),  # view2[i] vs view1[i]
        ])  # [2B]
        
        # Compute loss
        log_prob = pos_sim - torch.logsumexp(sim, dim=1)  # [2B]
        loss = -log_prob.mean()
        
        return loss
    
    def training_step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            x1: First augmented views [B, 1, n_mels, time]
            x2: Second augmented views [B, 1, n_mels, time]
            
        Returns:
            loss: Contrastive loss
        """
        # Concatenate views
        x = torch.cat([x1, x2], dim=0)  # [2B, 1, n_mels, time]
        
        # Forward pass
        z = self.forward(x)  # [2B, proj_dim]
        
        # Compute loss
        loss = self.contrastive_loss(z)
        
        return loss
