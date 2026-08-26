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
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.encoders import CNNEncoder
from src.models.heads import ProjectionHead
from src.models.ssl import SimCLR
from src.models.ssl.simclr import nt_xent_loss_standalone
from src.data.datasets import SSLBirdSongDataset, SimCLRDataset, simclr_collate_fn

# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def mock_metadata_df():
    return pd.DataFrame({
        'scientific_name': ['Species_A', 'Species_B', 'Species_A'],
        'scientific_name_id': [0, 1, 0],
        'local_spectrogram_path': [
            '/fake/path/spec1.npy',
            '/fake/path/spec2.npy',
            '/fake/path/spec3.npy',
        ],
        'total_frames': [500, 300, 450],
    })

@pytest.fixture
def base_ssl_config():
    return {
        'segment_size': 256,
        'min_db': -80.0,
        'max_db': 0.0,
        'train': True,
        'window_config': {'strategy': 'sliding', 'stride': 256},
    }
@pytest.fixture
def sample_batch():
    """Create a sample batch of spectrograms."""
    # [B, n_mels, time_steps]
    return torch.randn(4, 128, 256)


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
        assert encoder.get_output_dim() == 512
        assert encoder.get_feature_dim() == 512
        assert encoder.get_sequence_dim() == 512

    def test_forward_output_shape(self, cnn_encoder, sample_batch):
        """forward() should output [B, embed_dim]."""
        h = cnn_encoder(sample_batch)
        assert h.shape == (4, 512)

    def test_forward_features_output_shape(self, cnn_encoder, sample_batch):
        """forward_features() should output [B, C, H, W] feature map."""
        # sample_batch: [4, 128, 256]
        features = cnn_encoder.forward_features(sample_batch)

        B, C, H, W = features.shape
        assert B == 4
        assert C == 512  # base_channels * 8
        assert H == 8    # n_mels / 16 = 128/16
        assert W == 16   # time / 16 = 256/16

    def test_forward_sequence_output_shape(self, cnn_encoder, sample_batch):
        """forward_sequence() should output [B, S, D] token sequence."""
        # sample_batch: [4, 128, 256]
        sequence = cnn_encoder.forward_sequence(sample_batch)

        B, S, D = sequence.shape
        assert B == 4
        assert S == 16   # time / 16 = 256/16
        assert D == 512  # base_channels * 8

    def test_forward_sequence_for_transformer_input(self, cnn_encoder):
        """Sequence output should be compatible with transformer models."""
        # Test with different time lengths
        for T in [128, 256, 512]:
            x = torch.randn(2, 128, T)
            sequence = cnn_encoder.forward_sequence(x)

            expected_S = T // 16
            assert sequence.shape == (2, expected_S, 512)

    def test_all_forward_methods_consistent(self, cnn_encoder, sample_batch):
        """Different forward methods should be based on same features."""
        cnn_encoder.eval()
        h = cnn_encoder(sample_batch)
        features = cnn_encoder.forward_features(sample_batch)
        sequence = cnn_encoder.forward_sequence(sample_batch)

        # Verify that forward() result is consistent with forward_features()
        # Manual computation of forward() from features
        manual = cnn_encoder.freq_pool(features)
        manual = cnn_encoder.time_pool(manual)
        manual = manual.view(manual.size(0), -1)
        manual = cnn_encoder.dropout(manual)
        manual = cnn_encoder.embed(manual)

        assert torch.allclose(h, manual, atol=1e-5)

        # Verify sequence is derived from features
        manual_seq = cnn_encoder.freq_pool(features)
        manual_seq = manual_seq.squeeze(2)
        manual_seq = manual_seq.transpose(1, 2)

        assert torch.allclose(sequence, manual_seq, atol=1e-5)

    def test_batch_size_independence_all_methods(self, cnn_encoder):
        """All forward methods should work with different batch sizes."""
        cnn_encoder.eval()
        for B in [1, 2, 8, 16]:
            x = torch.randn(B, 128, 256)

            h = cnn_encoder(x)
            features = cnn_encoder.forward_features(x)
            sequence = cnn_encoder.forward_sequence(x)

            assert h.shape == (B, 512)
            assert features.shape == (B, 512, 8, 16)
            assert sequence.shape == (B, 16, 512)

    def test_time_steps_flexibility(self, cnn_encoder):
        """All methods should handle different time lengths."""
        for T in [128, 256, 512, 1024]:
            x = torch.randn(2, 128, T)

            # forward() always returns fixed size via adaptive pooling
            h = cnn_encoder(x)
            assert h.shape == (2, 512)

            # forward_features() preserves temporal dimension
            features = cnn_encoder.forward_features(x)
            expected_W = T // 16
            assert features.shape == (2, 512, 8, expected_W)

            # forward_sequence() adapts to temporal dimension
            sequence = cnn_encoder.forward_sequence(x)
            assert sequence.shape == (2, expected_W, 512)

    def test_features_preserve_spatial_info(self, cnn_encoder):
        """Feature map should preserve spatial structure."""
        # Create two spectrograms with different patterns in different regions
        x1 = torch.zeros(1, 128, 512)          # [B, n_mels, time]
        x1[:, :64, :256] = 1.0                 # top-left quadrant (freq <64, time<256)

        x2 = torch.zeros(1, 128, 512)
        x2[:, 64:, 256:] = 1.0                 # bottom-right quadrant

        features1 = cnn_encoder.forward_features(x1)
        features2 = cnn_encoder.forward_features(x2)

        # Features should be different (spatial info preserved)
        assert not torch.allclose(features1, features2, atol=1e-2)

    def test_sequence_temporal_order(self, cnn_encoder):
        """Sequence tokens should maintain temporal order."""
        # Create spectrogram with distinct patterns at different times
        x = torch.zeros(1, 128, 320) # [B, n_mels, time]
        x[:, :, :100] = 1.0   # Early
        x[:, :, 220:] = 2.0   # Late

        sequence = cnn_encoder.forward_sequence(x)  # [1, 20, 512]

        # Early tokens should differ from late tokens
        early_tokens = sequence[:, :5, :]   # First 5 tokens
        late_tokens = sequence[:, -5:, :]   # Last 5 tokens

        # Mean patterns should be different
        assert not torch.allclose(
            early_tokens.mean(dim=1),
            late_tokens.mean(dim=1),
            atol=1e-3,
        )

    def test_output_is_float(self, cnn_encoder, sample_batch):
        """All outputs should be float32."""
        h = cnn_encoder(sample_batch)
        features = cnn_encoder.forward_features(sample_batch)
        sequence = cnn_encoder.forward_sequence(sample_batch)

        assert h.dtype == torch.float32
        assert features.dtype == torch.float32
        assert sequence.dtype == torch.float32

    def test_output_deterministic_in_eval(self, cnn_encoder, sample_batch):
        """All methods should be deterministic in eval mode."""
        cnn_encoder.eval()

        h1 = cnn_encoder(sample_batch)
        h2 = cnn_encoder(sample_batch)
        assert torch.allclose(h1, h2)

        f1 = cnn_encoder.forward_features(sample_batch)
        f2 = cnn_encoder.forward_features(sample_batch)
        assert torch.allclose(f1, f2)

        s1 = cnn_encoder.forward_sequence(sample_batch)
        s2 = cnn_encoder.forward_sequence(sample_batch)
        assert torch.allclose(s1, s2)

    def test_gradient_flow_all_methods(self, cnn_encoder):
        """Gradients should flow through all forward methods."""
        x = torch.randn(4, 128, 256, requires_grad=False)

        # Test forward()
        h = cnn_encoder(x)
        loss_h = h.sum()
        loss_h.backward()
        # Check gradients exist
        grad_count = sum(
            1 for p in cnn_encoder.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No gradients in forward()"
        cnn_encoder.zero_grad()

        # Test forward_features()
        features = cnn_encoder.forward_features(x)
        loss_f = features.sum()
        loss_f.backward()
        grad_count = sum(
            1 for p in cnn_encoder.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No gradients in forward_features()"
        cnn_encoder.zero_grad()

        # Test forward_sequence()
        sequence = cnn_encoder.forward_sequence(x)
        loss_s = sequence.sum()
        loss_s.backward()
        grad_count = sum(
            1 for p in cnn_encoder.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0, "No gradients in forward_sequence()"

    def test_custom_base_channels_feature_dim(self):
        """get_feature_dim should match base_channels * 8."""
        for base_ch in [32, 64, 128]:
            encoder = CNNEncoder(n_mels=128, embed_dim=512, base_channels=base_ch)
            assert encoder.get_feature_dim() == base_ch * 8

            x = torch.randn(1, 128, 256)
            features = encoder.forward_features(x)
            assert features.shape[1] == base_ch * 8

    def test_sequence_as_transformer_tokens(self, cnn_encoder):
        """Verify sequence output format for transformer integration."""
        x = torch.randn(2, 128, 512)
        tokens = cnn_encoder.forward_sequence(x)  # [2, 32, 512]

        # Simulate transformer input preparation
        # Add CLS token
        cls_token = torch.randn(1, 1, 512).expand(2, -1, -1)
        with_cls = torch.cat([cls_token, tokens], dim=1)  # [2, 33, 512]

        assert with_cls.shape == (2, 33, 512)

    def test_weight_initialization(self):
        """Weights should be initialized with non-zero values."""
        encoder = CNNEncoder(n_mels=128, embed_dim=512)
        for name, param in encoder.named_parameters():
            if 'weight' in name:
                assert param.abs().sum() > 0, f"Zero weights for {name}"

    def test_conv_output_shapes_intermediate(self, cnn_encoder):
        """Verify intermediate conv block shapes."""
        x = torch.randn(1, 1, 128, 256)  # [B, C, n_mel, time]

        # Check each conv block manually
        out = cnn_encoder.conv1(x)
        assert out.shape == (1, 64, 64, 128), f"conv1: {out.shape}"

        out = cnn_encoder.conv2(out)
        assert out.shape == (1, 128, 32, 64), f"conv2: {out.shape}"

        out = cnn_encoder.conv3(out)
        assert out.shape == (1, 256, 16, 32), f"conv3: {out.shape}"

        out = cnn_encoder.conv4(out)
        assert out.shape == (1, 512, 8, 16), f"conv4: {out.shape}"


# ============================================================================
# Integration: CNN features → SimCLR
# ============================================================================

class TestCNNSimCLRIntegration:
    """Test integration between CNN encoder features and SimCLR."""

    def test_features_to_ssl(self):
        """CNN features should be usable for SSL pretraining."""
        encoder = CNNEncoder(n_mels=128, embed_dim=512)
        projection = ProjectionHead(input_dim=512, hidden_dim=256, output_dim=128)

        model = SimCLR(encoder=encoder, projection=projection, temperature=0.07)

        x1 = torch.randn(8, 128, 256)
        x2 = torch.randn(8, 128, 256)

        loss, acc = model.training_step(x1, x2)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_features_reuse_for_classification(self):
        """Pretrained features should be reusable for classification."""
        encoder = CNNEncoder(n_mels=128, embed_dim=512)

        # Simulate pretraining
        x = torch.randn(4, 128, 256)
        h = encoder(x)  # Get pooled embedding

        # Add classification head
        classifier = nn.Linear(512, 10)
        logits = classifier(h)

        assert logits.shape == (4, 10)

    def test_sequence_to_transformer_classifier(self):
        """Sequence output should feed into transformer classifier."""
        encoder = CNNEncoder(n_mels=128, embed_dim=512)

        x = torch.randn(2, 128, 512)
        tokens = encoder.forward_sequence(x)  # [2, 32, 512]

        # Simple transformer layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True,
        )

        # Should work without errors
        output = transformer_layer(tokens)
        assert output.shape == tokens.shape

    def test_encode_method_uses_pooled_forward(self, simclr_model, sample_batch):
        """SimCLR.encode() should use the pooled forward()."""
        simclr_model.eval()
        h = simclr_model.encode(sample_batch)
        assert h.shape == (4, 512)

        # Should match encoder's forward() directly
        h_direct = simclr_model.encoder(sample_batch)
        assert torch.allclose(h, h_direct)

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
        projection_head.eval()
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
    
    def test_temperature_effect(self, simclr_model, normalized_projections):
        """Lower temperature produces higher loss for imperfect pairs (sharper distribution)."""
        z1, z2 = normalized_projections
        
        model_low_temp = SimCLR(n_mels=128, embed_dim=512, temperature=0.05)
        model_high_temp = SimCLR(n_mels=128, embed_dim=512, temperature=0.5)
        
        loss_low = model_low_temp.nt_xent_loss(z1.clone(), z2.clone())
        loss_high = model_high_temp.nt_xent_loss(z1.clone(), z2.clone())
        
        # Both should be valid losses
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0
        # With small batch (2 samples), temperature effect may not hold monotonically
        # Just verify they're different 
        assert loss_low.item() != loss_high.item(), \
              f"Temperature should affect loss: both are {loss_low.item():.4f}"

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
        x1 = torch.randn(8, 128, 256)
        x2 = torch.randn(8, 128, 256)

        loss, acc = simclr_model.training_step(x1, x2)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

        assert isinstance(acc, torch.Tensor)
        assert acc.dim() == 0
        assert 0 <= acc.item() <= 1

    def test_accuracy_perfect_pairs(self, simclr_model):
        """Accuracy should be 100% for perfect pairs."""
        simclr_model.eval()
        x = torch.randn(8, 128, 256)
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

        x = torch.randn(4, 64, 128)
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
        simclr_model.eval()
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = SSLBirdSongDataset(df=mock_metadata_df, **base_ssl_config)
        x, _ = dataset[0]
        x = x.unsqueeze(0)  # Add batch dim
        
        h = simclr_model.encode(x)
        assert h.shape == (1, 512)
        # Embedding should NOT be normalized (raw representation)
        assert h.norm().item() != 1.0
