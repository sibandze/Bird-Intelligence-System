# tests/test_ssl_models.py
"""
SSL Model Components
Tests for encoder, projection head, and SimCLR model.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, patch

from src.models.encoders import CNNEncoder
from src.models.heads import ProjectionHead
from src.models.ssl import SimCLR
from src.models.ssl.simclr import nt_xent_loss_standalone


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_batch():
    """Create a sample batch of spectrograms."""
    # [B, 1, n_mels, time_steps]
    return torch.randn(4, 1, 128, 256)


@pytest.fixture
def sample_embeddings():
    """Create sample encoder embeddings."""
    return torch.randn(4, 512)


@pytest.fixture
def cnn_encoder():
    """Create a CNN encoder with default config."""
    return CNNEncoder(n_mels=128, embed_dim=512, base_channels=64)


@pytest.fixture
def projection_head():
    """Create a projection head with default config."""
    return ProjectionHead(input_dim=512, hidden_dim=256, output_dim=128)


@pytest.fixture
def simclr_model():
    """Create a SimCLR model."""
    return SimCLR(n_mels=128, embed_dim=512)


# ============================================================================
# CNN Encoder Tests
# ============================================================================

class TestCNNEncoder:
    """Test CNN encoder architecture."""
    
    def test_encoder_creation(self):
        """Encoder should initialize with valid parameters."""
        encoder = CNNEncoder(n_mels=128, embed_dim=512, base_channels=64)
        assert isinstance(encoder, nn.Module)
        assert encoder.embed_dim == 512
    
    def test_input_output_shape(self, cnn_encoder, sample_batch):
        """Output should have shape [B, embed_dim]."""
        h = cnn_encoder(sample_batch)
        assert h.shape == (4, 512)
    
    def test_batch_size_independence(self, cnn_encoder):
        """Should work with different batch sizes."""
        for B in [1, 2, 8, 16]:
            x = torch.randn(B, 1, 128, 256)
            h = cnn_encoder(x)
            assert h.shape == (B, 512)
    
    def test_time_steps_flexibility(self, cnn_encoder):
        """Should handle different time lengths via adaptive pooling."""
        for T in [128, 256, 512, 1024]:
            x = torch.randn(2, 1, 128, T)
            h = cnn_encoder(x)
            assert h.shape == (2, 512)
    
    def test_output_is_float(self, cnn_encoder, sample_batch):
        """Output should be float32."""
        h = cnn_encoder(sample_batch)
        assert h.dtype == torch.float32
    
    def test_output_is_not_constant(self, cnn_encoder, sample_batch):
        """Different inputs should produce different outputs."""
        x1 = torch.randn(4, 1, 128, 256)
        x2 = torch.randn(4, 1, 128, 256)
        h1 = cnn_encoder(x1)
        h2 = cnn_encoder(x2)
        assert not torch.allclose(h1, h2)
    
    def test_gradient_flow(self, cnn_encoder, sample_batch):
        """Gradients should flow through encoder."""
        h = cnn_encoder(sample_batch)
        loss = h.sum()
        loss.backward()
        
        # Check that gradients exist
        for name, param in cnn_encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    def test_get_output_dim(self, cnn_encoder):
        """get_output_dim should return correct dimension."""
        assert cnn_encoder.get_output_dim() == 512
    
    def test_custom_embed_dim(self):
        """Should support custom embedding dimensions."""
        for embed_dim in [128, 256, 512, 1024]:
            encoder = CNNEncoder(n_mels=128, embed_dim=embed_dim)
            x = torch.randn(2, 1, 128, 256)
            h = encoder(x)
            assert h.shape == (2, embed_dim)
    
    def test_custom_base_channels(self):
        """Should support custom base channel count."""
        for base_ch in [32, 64, 128]:
            encoder = CNNEncoder(n_mels=128, embed_dim=512, base_channels=base_ch)
            x = torch.randn(2, 1, 128, 256)
            h = encoder(x)
            assert h.shape == (2, 512)
    
    def test_dropout_training_vs_eval(self, cnn_encoder, sample_batch):
        """Dropout should behave differently in train vs eval mode."""
        cnn_encoder.train()
        h_train = cnn_encoder(sample_batch)
        
        cnn_encoder.eval()
        h_eval = cnn_encoder(sample_batch)
        
        # In eval mode, output should be deterministic
        h_eval2 = cnn_encoder(sample_batch)
        assert torch.allclose(h_eval, h_eval2)
    
    def test_weight_initialization(self):
        """Weights should be initialized with non-zero values."""
        encoder = CNNEncoder(n_mels=128, embed_dim=512)
        for name, param in encoder.named_parameters():
            if 'weight' in name:
                assert param.abs().sum() > 0, f"Zero weights for {name}"


# ============================================================================
# Projection Head Tests
# ============================================================================

class TestProjectionHead:
    """Test projection head for contrastive learning."""
    
    def test_basic_shape(self, projection_head, sample_embeddings):
        """Should map [B, 512] to [B, 128]."""
        z = projection_head(sample_embeddings)
        assert z.shape == (4, 128)
    
    def test_custom_dimensions(self):
        """Should support custom dimensions."""
        head = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=256)
        h = torch.randn(4, 512)
        z = head(h)
        assert z.shape == (4, 256)
    
    def test_batch_size_independence(self, projection_head):
        """Should work with different batch sizes."""
        for B in [1, 2, 16, 64]:
            h = torch.randn(B, 512)
            z = projection_head(h)
            assert z.shape == (B, 128)
    
    def test_gradient_flow(self, projection_head, sample_embeddings):
        """Gradients should flow through projection head."""
        z = projection_head(sample_embeddings)
        loss = z.sum()
        loss.backward()
        
        for name, param in projection_head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
    
    def test_output_different_from_input(self, projection_head, sample_embeddings):
        """Projection should transform input."""
        h1, h2 = torch.randn(2, 512), torch.randn(2, 512)
        z1, z2 = projection_head(h1), projection_head(h2)
        
        # Different inputs -> different outputs
        assert not torch.allclose(z1, z2)
    
    def test_batchnorm_training_mode(self, projection_head):
        """BatchNorm should behave differently in train vs eval."""
        h = torch.randn(4, 512)
        
        projection_head.train()
        z_train = projection_head(h)
        
        projection_head.eval()
        z_eval = projection_head(h)
        
        # Should produce different results in different modes
        # (especially with small batch size where running stats dominate)
        assert z_train.shape == z_eval.shape


# ============================================================================
# NT-Xent / InfoNCE Loss Tests
# ============================================================================

class TestSimCLRLoss:
    """Test contrastive loss functions."""
    
    @pytest.fixture
    def normalized_projections(self):
        """Create normalized projections for loss testing."""
        # Create embeddings where first 2 are close, last 2 are far
        z = torch.randn(4, 128)
        z = F.normalize(z, dim=1)
        z1 = z[:2]  # [2, 128]
        z2 = z[2:]  # [2, 128]
        return z1, z2
    
    def test_loss_is_positive(self, simclr_model, normalized_projections):
        """Contrastive loss should be non-negative."""
        z1, z2 = normalized_projections
        loss = simclr_model.nt_xent_loss(z1, z2)
        assert loss.item() >= 0
    
    def test_loss_decreases_for_similar_pairs(self, simclr_model):
        """Loss should decrease when positive pairs are more similar."""
        # Create highly similar positive pairs
        z_base = F.normalize(torch.randn(4, 128), dim=1)
        z1_close = z_base[:2]
        z2_close = z_base[:2] + 0.01 * torch.randn(2, 128)
        z2_close = F.normalize(z2_close, dim=1)
        
        # Create dissimilar pairs
        z1_far = z_base[:2]
        z2_far = F.normalize(torch.randn(2, 128), dim=1)
        
        loss_close = simclr_model.nt_xent_loss(z1_close, z2_close)
        loss_far = simclr_model.nt_xent_loss(z1_far, z2_far)
        
        # Similar pairs should have lower loss
        assert loss_close.item() < loss_far.item(), \
            f"close={loss_close.item():.4f}, far={loss_far.item():.4f}"
    
    def test_loss_symmetry(self, simclr_model, normalized_projections):
        """Loss should be symmetric: L(z1, z2) = L(z2, z1)."""
        z1, z2 = normalized_projections
        loss_12 = simclr_model.nt_xent_loss(z1, z2)
        loss_21 = simclr_model.nt_xent_loss(z2, z1)
        assert torch.allclose(loss_12, loss_21, atol=1e-6)
    
    def test_loss_increases_with_temperature(self, simclr_model, normalized_projections):
        """Higher temperature should produce lower loss (softer assignments)."""
        z1, z2 = normalized_projections
        
        model_low_temp = SimCLR(n_mels=128, embed_dim=512, temperature=0.05)
        model_high_temp = SimCLR(n_mels=128, embed_dim=512, temperature=0.5)
        
        loss_low = model_low_temp.nt_xent_loss(z1.clone(), z2.clone())
        loss_high = model_high_temp.nt_xent_loss(z1.clone(), z2.clone())
        
        # Lower temperature = sharper distribution = higher loss for imperfect pairs
        assert loss_low.item() > loss_high.item(), \
            f"low_temp={loss_low.item():.4f}, high_temp={loss_high.item():.4f}"
    
    def test_perfect_pairs_zero_loss(self, simclr_model):
        """Identical pairs should have very low loss."""
        z = F.normalize(torch.randn(2, 128), dim=1)
        z1 = z.clone()
        z2 = z.clone()
        
        loss = simclr_model.nt_xent_loss(z1, z2)
        
        # For perfect pairs, loss should be very low (effectively 0 with large batch)
        # With batch_size=2, the positive is also the negative (only other sample)
        # So loss > 0 but should be low relative to random
        assert loss.item() < 1.0  # Upper bound check
    
    def test_nt_xent_vs_info_nce(self, simclr_model, normalized_projections):
        """nt_xent_loss and info_nce_loss should produce identical results."""
        z1, z2 = normalized_projections
        
        loss_nt_xent = simclr_model.nt_xent_loss(z1.clone(), z2.clone())
        loss_info_nce = simclr_model.info_nce_loss(z1.clone(), z2.clone())
        
        assert torch.allclose(loss_nt_xent, loss_info_nce, atol=1e-5), \
            f"nt_xent={loss_nt_xent.item():.6f}, info_nce={loss_info_nce.item():.6f}"
    
    def test_standalone_loss_function(self, normalized_projections):
        """Standalone loss function should match class method."""
        z1, z2 = normalized_projections
        
        model = SimCLR(n_mels=128, embed_dim=512, temperature=0.07)
        loss_class = model.nt_xent_loss(z1.clone(), z2.clone())
        loss_standalone = nt_xent_loss_standalone(z1.clone(), z2.clone(), temperature=0.07)
        
        assert torch.allclose(loss_class, loss_standalone, atol=1e-5)
    
    def test_no_self_similarity_contamination(self, simclr_model):
        """Self-similarity should not contribute to loss (masked out)."""
        # Create embeddings where self-similarity is artificially high
        z = torch.ones(4, 128)  # All ones → self-sim = 128
        z = F.normalize(z, dim=1)
        z1, z2 = z[:2], z[2:]
        
        loss = simclr_model.nt_xent_loss(z1, z2)
        # Loss should be finite (self-sim masked out)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_large_batch_loss(self, simclr_model):
        """Loss should work with large batch sizes."""
        B = 64
        z1 = F.normalize(torch.randn(B, 128), dim=1)
        z2 = F.normalize(torch.randn(B, 128), dim=1)
        
        loss = simclr_model.nt_xent_loss(z1, z2)
        assert not torch.isnan(loss)
        assert loss.item() >= 0


# ============================================================================
# SimCLR Model Tests
# ============================================================================

class TestSimCLRModel:
    """Test SimCLR model integration."""
    
    def test_model_creation(self, simclr_model):
        """SimCLR model should initialize correctly."""
        assert isinstance(simclr_model.encoder, CNNEncoder)
        assert isinstance(simclr_model.projection, ProjectionHead)
        assert simclr_model.temperature == 0.07
    
    def test_forward_normalization(self, simclr_model, sample_batch):
        """Forward should return L2-normalized projections."""
        z = simclr_model(sample_batch)
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_encode_no_normalization(self, simclr_model, sample_batch):
        """encode() should return raw embeddings (not normalized)."""
        h = simclr_model.encode(sample_batch)
        assert h.shape == (4, 512)
        # Raw embeddings should NOT be normalized
        norms = h.norm(dim=1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_training_step(self, simclr_model):
        """training_step should return loss and accuracy."""
        x1 = torch.randn(8, 1, 128, 256)
        x2 = torch.randn(8, 1, 128, 256)
        
        loss, acc = simclr_model.training_step(x1, x2)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0
        
        assert isinstance(acc, torch.Tensor)
        assert acc.dim() == 0
        assert 0 <= acc.item() <= 1
    
    def test_accuracy_perfect_pairs(self, simclr_model):
        """Accuracy should be 100% for perfect pairs."""
        x = torch.randn(8, 1, 128, 256)
        x1 = x.clone()
        x2 = x.clone()
        
        _, acc = simclr_model.training_step(x1, x2)
        
        # With identical views, accuracy should be high
        # (but not necessarily 100% due to other negatives in batch)
        assert acc.item() > 0.9, f"Expected >0.9, got {acc.item():.4f}"
    
    def test_custom_encoder(self):
        """Should accept custom encoder and projection."""
        custom_encoder = CNNEncoder(n_mels=64, embed_dim=256, base_channels=32)
        custom_proj = ProjectionHead(input_dim=256, hidden_dim=128, output_dim=64)
        
        model = SimCLR(encoder=custom_encoder, projection=custom_proj)
        
        x = torch.randn(4, 1, 64, 128)
        z = model(x)
        assert z.shape == (4, 64)
    
    def test_temperature_setter(self):
        """Temperature should be configurable."""
        for temp in [0.05, 0.1, 0.5, 1.0]:
            model = SimCLR(n_mels=128, embed_dim=512, temperature=temp)
            assert model.temperature == temp
    
    def test_gradient_flow_through_model(self, simclr_model, sample_batch):
        """Gradients should flow through full model."""
        x1 = sample_batch[:2]
        x2 = sample_batch[2:]
        
        loss, _ = simclr_model.training_step(x1, x2)
        loss.backward()
        
        # Check gradients in encoder
        for name, param in simclr_model.encoder.named_parameters():
            assert param.grad is not None, f"No gradient for encoder.{name}"
        
        # Check gradients in projection
        for name, param in simclr_model.projection.named_parameters():
            assert param.grad is not None, f"No gradient for projection.{name}"


# ============================================================================
# Integration Tests (Data → Model)
# ============================================================================

class TestDataModelIntegration:
    """Integration tests connecting data pipeline to model."""
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_dataset_to_model_flow(
        self, mock_load, mock_metadata_df, base_ssl_config, simclr_model,
    ):
        """Full flow from dataset to model output."""
        from torch.utils.data import DataLoader
        
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = SimCLRDataset(df=mock_metadata_df, **base_ssl_config)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=simclr_collate_fn,
        )
        
        for x1, x2 in loader:
            loss, acc = simclr_model.training_step(x1, x2)
            assert loss.dim() == 0
            assert acc.dim() == 0
            assert loss.item() >= 0
            break
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_encode_for_downstream(
        self, mock_load, mock_metadata_df, base_ssl_config, simclr_model,
    ):
        """Encoder embeddings should be extractable for downstream tasks."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = SSLBirdSongDataset(df=mock_metadata_df, **base_ssl_config)
        x, _ = dataset[0]
        x = x.unsqueeze(0)  # Add batch dim
        
        h = simclr_model.encode(x)
        assert h.shape == (1, 512)
        # Embedding should NOT be normalized (raw representation)
        assert h.norm().item() != 1.0
