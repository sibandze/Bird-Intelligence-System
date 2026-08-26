# tests/test_ssl_sanity.py
"""
Sanity check: Run a tiny overfitting experiment to verify
the full SSL pipeline works end-to-end.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.ssl import SimCLR
from src.data.datasets import SimCLRDataset, simclr_collate_fn


def test_simclr_overfit_small_batch():
    """
    Overfit SimCLR on a tiny synthetic batch.

    Loss should decrease significantly within a few iterations.
    """
    # Create synthetic data: 4 samples, each with 2 views
    n_samples = 4
    n_mels, n_frames = 128, 256

    # Create embeddings that are easy to learn
    x1 = torch.randn(n_samples, n_mels, n_frames)
    x2 = x1 + 0.01 * torch.randn(n_samples, n_mels, n_frames)

    # Initialize model and optimizer
    model = SimCLR(n_mels=n_mels, embed_dim=128, temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Track loss
    initial_loss = None
    final_loss = None

    # Overfit loop
    model.train()
    for step in range(50):
        optimizer.zero_grad()
        loss, acc = model.training_step(x1, x2)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    print(f"Final accuracy: {acc.item():.4f}")

    # Loss should decrease
    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    # Final loss should be reasonably low
    assert final_loss < 1.0, f"Loss too high: {final_loss:.4f}"


def test_simclr_accuracy_improves():
    """
    Verify that positive pair retrieval accuracy improves during training.
    """
    n_samples = 8
    n_mels, n_frames = 128, 256

    # Create base embeddings
    base = torch.randn(n_samples, n_mels, n_frames)

    # Create two augmented views with slight noise
    x1 = base + 0.1 * torch.randn(n_samples, n_mels, n_frames)
    x2 = base + 0.1 * torch.randn(n_samples, n_mels, n_frames)

    model = SimCLR(n_mels=n_mels, embed_dim=256, temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    initial_acc = None
    final_acc = None

    model.train()
    for step in range(100):
        optimizer.zero_grad()
        loss, acc = model.training_step(x1, x2)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_acc = acc.item()
        final_acc = acc.item()

    print(f"Initial accuracy: {initial_acc:.4f}, Final accuracy: {final_acc:.4f}")

    # Accuracy should improve
    assert final_acc > initial_acc, \
        f"Accuracy did not improve: {initial_acc:.4f} -> {final_acc:.4f}"


def test_embedding_extraction():
    """
    Verify embeddings can be extracted from trained model.
    """
    n_mels, n_frames = 128, 256

    model = SimCLR(n_mels=n_mels, embed_dim=512)
    model.eval()

    x = torch.randn(4, n_mels, n_frames)

    # Get projected embeddings (normalized)
    with torch.no_grad():
        z = model(x)
        h = model.encode(x)

    # z should be normalized
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # h should have correct dimension
    assert h.shape == (4, 512)

    # h and z should be different (projection transforms)
    assert not torch.allclose(h[:, :128], z, atol=1e-3)
