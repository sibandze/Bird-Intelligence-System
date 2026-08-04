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
    Uses InfoNCE (NT-Xent) loss: normalized temperature-scaled cross entropy.

    Architecture:
        x → Encoder f → h → Projection g → z → L2 normalize → NT-Xent loss

    The loss maximizes agreement between differently augmented views 
    of the same sample while minimizing agreement with all other samples 
    in the batch.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        projection: nn.Module = None,
        temperature: float = 0.07,
        **encoder_kwargs,
    ):
        """
        Args:
            encoder: Pre-built encoder (if None, creates CNNEncoder)
            projection: Pre-built projection head (if None, creates ProjectionHead)
            temperature: Temperature parameter τ for NT-Xent loss
                         Lower τ → harder assignments, higher τ → softer
                         Typical range: 0.05 - 0.5
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
        Forward pass returning normalized projected embeddings.

        Args:
            x: Input spectrograms [B, 1, n_mels, time]

        Returns:
            z: L2-normalized projected embeddings [B, proj_dim]
        """
        h = self.encoder(x)
        z = self.projection(h)
        z = F.normalize(z, dim=1)  # L2 normalize to unit hypersphere
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoder embeddings only (without projection).
        Useful for downstream tasks like linear evaluation.
        
        Args:
            x: Input spectrograms [B, 1, n_mels, time]
            
        Returns:
            h: Encoder embeddings [B, embed_dim]
        """
        return self.encoder(x)
    
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
        
        Also known as InfoNCE with 2 views.
        
        For each positive pair (z1[i], z2[i]), the loss is:
            ℓ(i,j) = -log( exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )
        
        where sim(u,v) = u^T v (cosine similarity since vectors are normalized)
        
        Args:
            z1: First view projections [B, proj_dim] (already normalized)
            z2: Second view projections [B, proj_dim] (already normalized)
        
        Returns:
            loss: Scalar loss averaged over all 2B samples
        """
        B = z1.shape[0]
        device = z1.device
        
        # Concatenate all representations
        z = torch.cat([z1, z2], dim=0)  # [2B, proj_dim]
        
        # Compute similarity matrix: S[i][j] = z_i^T z_j / τ
        # Since z is normalized, this is cosine similarity / τ
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]
        
        # Mask to remove self-similarity (diagonal)
        # Set diagonal to -inf so it doesn't contribute to softmax denominator
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(self_mask, float('-inf'))
        
        # Create labels for positive pairs
        # For view1[i], positive is view2[i] (index i+B)
        # For view2[i], positive is view1[i] (index i)
        positive_indices = torch.cat([
            torch.arange(B, 2 * B, device=device),  # view1 positives: i+B
            torch.arange(0, B, device=device),        # view2 positives: i
        ])
        
        # Cross-entropy with softmax over all 2B-1 negatives
        # ℓ(i) = -log( exp(sim(z_i, z_pos) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )
        loss = F.cross_entropy(sim_matrix, positive_indices)
        
        return loss
    
    def info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Alternative InfoNCE implementation (equivalent to nt_xent_loss).
        
        Explicitly computes:
            ℒ = -𝔼[log( f(z1, z2) / (f(z1, z2) + Σ f(z1, z_neg)) )]
        
        where f(u,v) = exp(sim(u,v) / τ)
        """
        B = z1.shape[0]
        
        # Positive pair similarities
        pos_sim = torch.sum(z1 * z2, dim=1) / self.temperature  # [B]
        
        # All-pair similarities (B x 2B)
        z_all = torch.cat([z1, z2], dim=0)  # [2B, dim]
        sim_all = torch.matmul(z1, z_all.T) / self.temperature  # [B, 2B]
        
        # Mask out self (z1[i] vs z1[i])
        mask = torch.eye(B, 2*B, dtype=torch.bool, device=z1.device)
        sim_all.masked_fill_(mask, float('-inf'))
        
        # InfoNCE loss for view1→view2
        loss_1 = -pos_sim + torch.logsumexp(sim_all, dim=1)  # [B]
        
        # InfoNCE loss for view2→view1 (symmetric)
        sim_all_2 = torch.matmul(z2, z_all.T) / self.temperature  # [B, 2B]
        mask_2 = torch.eye(B, 2*B, dtype=torch.bool, device=z1.device)
        mask_2[:, B:] = True  # Mask z2[i] vs z2[i]
        sim_all_2.masked_fill_(mask_2, float('-inf'))
        
        loss_2 = -pos_sim + torch.logsumexp(sim_all_2, dim=1)  # [B]
        
        # Average over both views and batch
        loss = (loss_1.mean() + loss_2.mean()) / 2
        
        return loss
    
    def training_step(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        """
        Single training step returning loss and accuracy.
        
        Args:
            x1: First augmented views [B, 1, n_mels, time]
            x2: Second augmented views [B, 1, n_mels, time]
            
        Returns:
            loss: NT-Xent loss
            accuracy: Positive pair retrieval accuracy (for monitoring)
        """
        # Forward pass with normalization
        z1 = self.forward(x1)  # [B, proj_dim], normalized
        z2 = self.forward(x2)  # [B, proj_dim], normalized
        
        # Compute loss
        loss = self.nt_xent_loss(z1, z2)
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            acc = self._compute_accuracy(z1, z2)
        
        return loss, acc
    
    @torch.no_grad()
    def _compute_accuracy(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute positive pair retrieval accuracy.
        
        For each view, checks if the most similar embedding 
        is the corresponding positive pair.
        """
        B = z1.shape[0]
        
        # Concatenate all embeddings
        z = torch.cat([z1, z2], dim=0)  # [2B, proj_dim]
        
        # Cosine similarity matrix
        sim = torch.matmul(z1, z.T)  # [B, 2B]
        
        # Mask out self
        mask = torch.eye(B, 2*B, dtype=torch.bool, device=z1.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # For view1[i], positive is at index i+B
        _, predicted = sim.max(dim=1)  # [B]
        correct = (predicted == torch.arange(B, 2*B, device=z1.device)).float()
        
        return correct.mean()


def nt_xent_loss_explicit(z1, z2, temperature=0.07):
    """
    Standalone NT-Xent loss function for external use.
    
    This can be used with any framework, not just SimCLR class.
    
    Args:
        z1: First view projections [B, D]
        z2: Second view projections [B, D]
        temperature: Temperature parameter
    
    Returns:
        loss: Scalar loss
    """
    B, D = z1.shape
    device = z1.device
    
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    
    # Cosine similarity / temperature
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]
    
    # Positive pairs: (i, i+B) and (i+B, i)
    pos_sim = torch.cat([
        torch.diag(sim, B),   # z1[i] vs z2[i]
        torch.diag(sim, -B),  # z2[i] vs z1[i]
    ])  # [2B]
    
    # Remove self-similarities for softmax denominator
    sim = sim - torch.eye(2*B, device=device) * 1e9
    
    # NT-Xent: -log( exp(pos/τ) / Σ exp(all/τ) )
    loss = -pos_sim + torch.logsumexp(sim, dim=1)  # [2B]
    
    return loss.mean()
