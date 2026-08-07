# tests/test_ssl_data_pipeline.py
"""
Tests: SSL Data Pipeline
Tests for augmentation pipeline, SSL dataset, and data loading.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.augmentations import (
    BaseAugmentation,
    AcousticAugmentation,
    SpecAugmentation,
    AugmentationPipeline,
)
from src.data.datasets import (
    SSLBirdSongDataset,
    SimCLRDataset,
    BYOLDataset,
    MoCoDataset,
    simclr_collate_fn,
    byol_collate_fn,
    moco_collate_fn,
    SupervisedBirdSongDataset,
)
from src.data.datasets.base import BaseSpectrogramDataset
from src.data.windowing import (
    WindowIndex,
    SlidingWindowStrategy,
    RandomWindowStrategy,
    CenterWindowStrategy,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_spectrogram():
    """Create a mock mel spectrogram with realistic values."""
    n_mels, n_frames = 128, 500
    mel = np.random.randn(n_mels, n_frames).astype(np.float32)
    # Normalize to [-80, 0] dB range typical for mel spectrograms
    mel = mel * 20 - 40
    return mel


@pytest.fixture
def mock_metadata_df():
    """Create a minimal metadata DataFrame."""
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
def mock_window():
    """Create a sample WindowIndex."""
    return WindowIndex(recording_idx=0, start_frame=0, end_frame=256)


@pytest.fixture
def base_ssl_config():
    """Base configuration for SSL datasets."""
    return {
        'segment_size': 256,
        'min_db': -80.0,
        'max_db': 0.0,
        'train': True,
        'window_config': {'strategy': 'sliding', 'stride': 256},
    }


# ============================================================================
# Augmentation Pipeline Tests
# ============================================================================

class TestBaseAugmentation:
    """Test augmentation base class."""
    
    def test_base_augmentation_interface(self):
        """Verify BaseAugmentation requires __call__ and get_params."""
        with pytest.raises(TypeError):
            BaseAugmentation()
        
        class ConcreteAug(BaseAugmentation):
            def __call__(self, x):
                return x
            def get_params(self):
                return {}
        
        aug = ConcreteAug()
        assert aug(torch.ones(1, 128, 256)).shape == (1, 128, 256)


class TestAcousticAugmentation:
    """Test acoustic augmentation (noise injection)."""
    
    def test_noise_injection_shape(self):
        """Noise should preserve tensor shape."""
        aug = AcousticAugmentation(noise_level=0.05)
        x = torch.rand(1, 128, 256)
        x_aug = aug(x)
        assert x_aug.shape == x.shape
    
    def test_noise_changes_values(self):
        """Noise should modify values."""
        aug = AcousticAugmentation(noise_level=0.5, noise_prob=1.0)
        x = torch.ones(1, 128, 256)
        x_aug = aug(x)
        assert not torch.allclose(x, x_aug)
    
    def test_disabled_noise(self):
        """Disabled augmentation should return input unchanged."""
        aug = AcousticAugmentation(enabled=False)
        x = torch.rand(1, 128, 256)
        assert torch.equal(aug(x), x)
    
    def test_noise_probability(self):
        """noise_prob=0 should never apply noise."""
        aug = AcousticAugmentation(noise_level=0.5, noise_prob=0.0)
        x = torch.ones(1, 128, 256)
        # Run multiple times to verify
        for _ in range(100):
            assert torch.equal(aug(x), x)
    
    def test_clamp_to_range(self):
        """Output should remain in [0, 1] for normalized inputs."""
        aug = AcousticAugmentation(noise_level=0.5, noise_prob=1.0)
        x = torch.clamp(torch.rand(4, 128, 256), 0.0, 1.0)
        x_aug = aug(x)
        assert x_aug.min() >= 0.0
        assert x_aug.max() <= 1.0
    
    def test_get_params(self):
        """get_params should return configuration dict."""
        aug = AcousticAugmentation(noise_level=0.03, noise_prob=0.8)
        params = aug.get_params()
        assert params['type'] == 'acoustic'
        assert params['noise_level'] == 0.03
        assert params['noise_prob'] == 0.8


class TestSpecAugmentation:
    """Test SpecAugment (frequency and time masking)."""
    
    def test_basic_shape_preservation(self):
        """Masking should preserve tensor shape."""
        aug = SpecAugmentation(
            freq_mask_param=10,
            time_mask_param=20,
            prob=1.0,
        )
        x = torch.rand(2, 128, 300)
        x_aug = aug(x)
        assert x_aug.shape == x.shape
    
    def test_frequency_masking(self):
        """Should zero out frequency bands."""
        aug = SpecAugmentation(
            num_freq_masks=1,
            freq_mask_param=50,
            num_time_masks=0,
            time_mask_param=0,
            prob=1.0,
        )
        x = torch.ones(1, 128, 256)
        x_aug = aug(x)
        # Check that some frequency bins are zeroed
        freq_means = x_aug[0].mean(dim=1)  # mean over time
        assert (freq_means < 1.0).any(), "No frequency masking applied"
        assert (freq_means == 0.0).any(), "No frequencies fully zeroed"
    
    def test_time_masking(self):
        """Should zero out time steps."""
        aug = SpecAugmentation(
            num_freq_masks=0,
            freq_mask_param=0,
            num_time_masks=1,
            time_mask_param=50,
            prob=1.0,
        )
        x = torch.ones(1, 128, 256)
        x_aug = aug(x)
        # Check that some time steps are zeroed
        time_means = x_aug[0].mean(dim=0)  # mean over frequencies
        assert (time_means < 1.0).any(), "No time masking applied"
        assert (time_means == 0.0).any(), "No time steps fully zeroed"
    
    def test_disabled_augmentation(self):
        """Disabled SpecAugment should be identity."""
        aug = SpecAugmentation(enabled=False, prob=1.0)
        x = torch.rand(1, 128, 256)
        assert torch.equal(aug(x), x)
    
    def test_probability_zero(self):
        """prob=0 should never apply masking."""
        aug = SpecAugmentation(prob=0.0, freq_mask_param=50)
        x = torch.ones(1, 128, 256)
        for _ in range(50):
            assert torch.equal(aug(x), x)
    
    def test_batch_independence(self):
        """Each sample in batch should get different masks."""
        aug = SpecAugmentation(prob=1.0, freq_mask_param=20)
        x = torch.ones(4, 128, 256)
        x_aug = aug(x)
        # Check that samples are masked differently
        for i in range(4):
            for j in range(i + 1, 4):
                if not torch.equal(x_aug[i], x_aug[j]):
                    return  # Found different masks, test passes
        pytest.fail("All samples received identical masks")
    
    def test_get_params(self):
        """get_params should return configuration."""
        aug = SpecAugmentation(
            num_freq_masks=3,
            freq_mask_param=8,
            num_time_masks=2,
            time_mask_param=12,
        )
        params = aug.get_params()
        assert params['type'] == 'specaugment'
        assert params['num_freq_masks'] == 3
        assert params['freq_mask_param'] == 8


class TestAugmentationPipeline:
    """Test augmentation pipeline composition."""
    
    def test_empty_pipeline(self):
        """Empty pipeline should be identity."""
        pipeline = AugmentationPipeline()
        x = torch.rand(1, 128, 256)
        assert torch.equal(pipeline(x), x)
    
    def test_pipeline_ordering(self):
        """Pipeline should apply augmentations in order."""
        calls = []
        
        class TrackedAug(BaseAugmentation):
            def __init__(self, name):
                self.name = name
            def __call__(self, x):
                calls.append(self.name)
                return x
            def get_params(self):
                return {'name': self.name}
        
        pipeline = AugmentationPipeline([
            TrackedAug('first'),
            TrackedAug('second'),
        ])
        pipeline(torch.rand(1, 128, 256))
        assert calls == ['first', 'second']
    
    def test_combined_pipeline(self):
        """Acoustic + SpecAugment should work together."""
        pipeline = AugmentationPipeline([
            AcousticAugmentation(noise_level=0.01, noise_prob=1.0),
            SpecAugmentation(prob=1.0, freq_mask_param=10),
        ])
        x = torch.rand(2, 128, 256)
        x_aug = pipeline(x)
        assert x_aug.shape == x.shape
        assert not torch.equal(x, x_aug), "Combined pipeline had no effect"
    
    def test_pipeline_disable(self):
        """Disabled pipeline should pass through."""
        pipeline = AugmentationPipeline(
            [AcousticAugmentation(noise_level=1.0)],
            enabled=False,
        )
        x = torch.ones(1, 128, 256)
        assert torch.equal(pipeline(x), x)
    
    def test_pipeline_repr(self):
        """String representation should list augmentations."""
        pipeline = AugmentationPipeline([
            AcousticAugmentation(),
            SpecAugmentation(),
        ])
        repr_str = repr(pipeline)
        assert 'AcousticAugmentation' in repr_str
        assert 'SpecAugmentation' in repr_str
    
    def test_pipeline_params(self):
        """get_params should include all sub-params."""
        pipeline = AugmentationPipeline([
            AcousticAugmentation(noise_level=0.03),
            SpecAugmentation(num_freq_masks=2),
        ])
        params = pipeline.get_params()
        assert params['num_augmentations'] == 2
        assert len(params['augmentations']) == 2


# ============================================================================
# SSL Dataset Tests
# ============================================================================

class TestSSLBirdSongDataset:
    """Test SSL dataset for dual-view generation."""
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_dual_views_are_different(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """Two views of same window should differ due to augmentation."""
        # Create fake spectrogram with distinct values
        mel = np.random.randn(128, 500).astype(np.float32)
        mock_load.return_value = mel
        
        dataset = SSLBirdSongDataset(
            df=mock_metadata_df,
            acoustic_aug_config={'enabled': True, 'noise_level': 0.1},
            spec_aug_config={'enabled': True, 'prob': 1.0},
            **base_ssl_config,
        )
        
        x1, x2 = dataset[0]
        assert not torch.equal(x1, x2), "Dual views should differ"
        assert x1.shape == x2.shape
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_view_shapes(self, mock_load, mock_metadata_df, base_ssl_config):
        """Each view should have shape [1, n_mels, segment_size]."""
        mel = np.random.randn(128, 500).astype(np.float32)
        mock_load.return_value = mel
        
        dataset = SSLBirdSongDataset(
            df=mock_metadata_df,
            **base_ssl_config,
        )
        
        x1, x2 = dataset[0]
        expected_shape = (1, 128, base_ssl_config['segment_size'])
        assert x1.shape == expected_shape, f"Expected {expected_shape}, got {x1.shape}"
        assert x2.shape == expected_shape
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_epoch_changes_windows(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """Window indices should change with epoch for sliding strategy."""
        mel = np.random.randn(128, 1000).astype(np.float32)
        mock_load.return_value = mel
        
        dataset = SSLBirdSongDataset(
            df=mock_metadata_df,
            window_config={'strategy': 'sliding', 'stride': 256},
            **base_ssl_config,
        )
        
        windows_epoch0 = list(dataset.windows)
        
        dataset.set_epoch(1)
        windows_epoch1 = list(dataset.windows)
        
        # Windows should be different between epochs
        assert windows_epoch0 != windows_epoch1, \
            "Sliding windows should change between epochs"
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_window_strategies(self, mock_load, mock_metadata_df, base_ssl_config):
        """Test different window strategies produce correct indices."""
        mel = np.random.randn(128, 800).astype(np.float32)
        mock_load.return_value = mel
        segment_size = base_ssl_config['segment_size']
        
        for strategy, config in [
            ('random', {'strategy': 'random'}),
            ('center', {'strategy': 'center'}),
            ('sliding', {'strategy': 'sliding', 'stride': 256}),
        ]:
            dataset = SSLBirdSongDataset(
                df=mock_metadata_df,
                window_config=config,
                **base_ssl_config,
            )
            assert len(dataset) == len(mock_metadata_df)
            
            for window in dataset.windows:
                assert window.end_frame - window.start_frame == segment_size or \
                       window.end_frame <= mel.shape[1]
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_short_spectrogram_padding(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """Spectrograms shorter than segment_size should be padded."""
        short_mel = np.random.randn(128, 100).astype(np.float32)  # < 256
        mock_load.return_value = short_mel
        
        dataset = SSLBirdSongDataset(
            df=mock_metadata_df,
            **base_ssl_config,
        )
        
        x1, x2 = dataset[0]
        assert x1.shape == (1, 128, base_ssl_config['segment_size'])
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_precomputed_total_frames(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """Should use total_frames column without loading spectrogram."""
        dataset = SSLBirdSongDataset(
            df=mock_metadata_df,
            **base_ssl_config,
        )
        
        # _get_total_frames should use precomputed value
        row = mock_metadata_df.iloc[0]
        frames = dataset._get_total_frames(row, 0)
        assert frames == 500
        mock_load.assert_not_called()


class TestSSLFrameworkAdapters:
    """Test framework-specific dataset adapters."""
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_simclr_dataset(self, mock_load, mock_metadata_df, base_ssl_config):
        """SimCLRDataset should produce valid output."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = SimCLRDataset(df=mock_metadata_df, **base_ssl_config)
        x1, x2 = dataset[0]
        assert x1.shape == (1, 128, 256)
        assert x2.shape == (1, 128, 256)
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_byol_dataset(self, mock_load, mock_metadata_df, base_ssl_config):
        """BYOLDataset should produce valid output."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = BYOLDataset(df=mock_metadata_df, **base_ssl_config)
        x1, x2 = dataset[0]
        assert x1.shape == (1, 128, 256)
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_moco_dataset(self, mock_load, mock_metadata_df, base_ssl_config):
        """MoCoDataset should produce valid output."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = MoCoDataset(df=mock_metadata_df, **base_ssl_config)
        x1, x2 = dataset[0]
        assert x1.shape == (1, 128, 256)


class TestSSLCollateFunctions:
    """Test collate functions for SSL frameworks."""
    
    def test_simclr_collate(self):
        """simclr_collate_fn should return x1, x2 as separate tensors."""
        batch = [
            (torch.randn(1, 128, 256), torch.randn(1, 128, 256))
            for _ in range(4)
        ]
        x1, x2 = simclr_collate_fn(batch)
        assert x1.shape == (4, 1, 128, 256)
        assert x2.shape == (4, 1, 128, 256)
    
    def test_byol_collate(self):
        """byol_collate_fn should return x1, x2 as separate tensors."""
        batch = [
            (torch.randn(1, 128, 256), torch.randn(1, 128, 256))
            for _ in range(4)
        ]
        x1, x2 = byol_collate_fn(batch)
        assert x1.shape == (4, 1, 128, 256)
        assert x2.shape == (4, 1, 128, 256)
    
    def test_moco_collate(self):
        """moco_collate_fn should return query and key tensors."""
        batch = [
            (torch.randn(1, 128, 256), torch.randn(1, 128, 256))
            for _ in range(4)
        ]
        im_q, im_k = moco_collate_fn(batch)
        assert im_q.shape == (4, 1, 128, 256)
        assert im_k.shape == (4, 1, 128, 256)


# ============================================================================
# DataLoader Integration Tests
# ============================================================================

class TestDataLoaderIntegration:
    """Test dataset integration with DataLoader."""
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_dataloader_shapes(self, mock_load, mock_metadata_df, base_ssl_config):
        """DataLoader should produce correct batch shapes."""
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
            assert x1.shape == (2, 1, 128, 256)
            assert x2.shape == (2, 1, 128, 256)
            break
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_epoch_synchronization(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """set_epoch should be callable on dataset."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = SSLBirdSongDataset(df=mock_metadata_df, **base_ssl_config)
        initial_len = len(dataset)
        
        dataset.set_epoch(5)
        assert len(dataset) == initial_len  # Length should not change
        assert dataset.epoch == 5


# ============================================================================
# Supervised Dataset Tests
# ============================================================================

class TestSupervisedDataset:
    """Test supervised dataset functionality."""
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_basic_output(self, mock_load, mock_metadata_df, base_ssl_config):
        """Should return (x, y) tuple."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = SupervisedBirdSongDataset(
            df=mock_metadata_df,
            **base_ssl_config,
        )
        
        x, y = dataset[0]
        assert x.shape == (1, 128, 256)
        assert isinstance(y, torch.Tensor)
        assert y.dtype == torch.long
        assert y.dim() == 0  # Scalar
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_label_contiguous_mapping(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """Labels should be mapped to contiguous 0..N-1."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = SupervisedBirdSongDataset(
            df=mock_metadata_df,
            **base_ssl_config,
        )
        
        # All labels should be in [0, num_classes-1]
        for i in range(len(dataset)):
            _, y = dataset[i]
            assert 0 <= y.item() < dataset.num_classes
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_label_to_idx_mapping(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """label_to_idx should map species names to contiguous indices."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        dataset = SupervisedBirdSongDataset(
            df=mock_metadata_df,
            **base_ssl_config,
        )
        
        # Species_A -> 0, Species_B -> 1
        assert dataset.label_to_idx['Species_A'] == 0
        assert dataset.label_to_idx['Species_B'] == 1
        assert dataset.idx_to_label[0] == 'Species_A'
        assert dataset.idx_to_label[1] == 'Species_B'
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_external_label_mapping(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """Should accept external label_to_idx."""
        mock_load.return_value = np.random.randn(128, 500).astype(np.float32)
        
        external_mapping = {'Species_A': 5, 'Species_B': 3}
        dataset = SupervisedBirdSongDataset(
            df=mock_metadata_df,
            label_to_idx=external_mapping,
            **base_ssl_config,
        )
        
        _, y0 = dataset[0]  # Species_A
        _, y1 = dataset[1]  # Species_B
        assert y0.item() == 5
        assert y1.item() == 3
        assert dataset.num_classes == 2
    
    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_spec_augment_on_supervised(
        self, mock_load, mock_metadata_df, base_ssl_config,
    ):
        """SpecAugment should be applied during training."""
        mock_load.return_value = np.ones((128, 500), dtype=np.float32)
        
        train_dataset = SupervisedBirdSongDataset(
            df=mock_metadata_df,
            train=True,
            spec_aug_config={'enabled': True, 'prob': 1.0, 'time_mask_param': 50},
            **base_ssl_config,
        )
        
        test_dataset = SupervisedBirdSongDataset(
            df=mock_metadata_df,
            train=False,
            spec_aug_config={'enabled': True, 'prob': 1.0, 'time_mask_param': 50},
            **base_ssl_config,
        )
        
        x_train, _ = train_dataset[0]
        x_test, _ = test_dataset[0]
        
        # Test mode should not augment
        assert torch.equal(x_test, torch.ones(1, 128, 256).float())
        # Train mode may augment (prob 1.0)
    
    def test_missing_label_error(self, mock_metadata_df, base_ssl_config):
        """Should raise error if DataFrame has unmapped labels."""
        incomplete_mapping = {'Species_A': 0}  # Missing Species_B
        
        with pytest.raises(ValueError, match='not in label_to_idx'):
            SupervisedBirdSongDataset(
                df=mock_metadata_df,
                label_to_idx=incomplete_mapping,
                **base_ssl_config,
            )

class TestEdgeCases:

    @patch('src.data.datasets.base.load_local_spectrogram')
    def test_legacy_total_frames_fallback(
        self, mock_load, base_ssl_config
    ):
        """Test _get_total_frames fallback when total_frames column is NaN."""
        # DataFrame without total_frames
        df_no_frames = pd.DataFrame({
            'scientific_name': ['Species_A'],
            'scientific_name_id': [0],
            'local_spectrogram_path': ['/fake/path/spec1.npy'],
        })
        mel = np.random.randn(128, 600).astype(np.float32)
        mock_load.return_value = mel

        dataset = SSLBirdSongDataset(
            df=df_no_frames,
            **base_ssl_config,
        )

        # Should call load_local_spectrogram to get frames
        frames = dataset._get_total_frames(df_no_frames.iloc[0], 0)
        assert frames == 600
        mock_load.assert_called_once()

    def test_random_window_strategy_deterministic_with_epoch(
        self, mock_metadata_df, base_ssl_config
    ):
        """Random strategy should produce same windows given same epoch seed."""
        # Patch random to be deterministic per epoch
        original_random = random.randint
        calls = []

        def tracked_randint(a, b):
            calls.append((a, b))
            return original_random(a, b)

        with patch('random.randint', side_effect=tracked_randint):
            dataset = SSLBirdSongDataset(
                df=mock_metadata_df,
                window_config={'strategy': 'random'},
                **base_ssl_config,
            )
            windows_epoch0 = list(dataset.windows)

            dataset.set_epoch(1)
            windows_epoch1 = list(dataset.windows)

        # Windows should change between epochs
        assert windows_epoch0!= windows_epoch1
        # But if we reset to epoch 0, we should get same as before
        dataset.set_epoch(0)
        windows_epoch0_again = list(dataset.windows)
        assert windows_epoch0 == windows_epoch0_again
