# src/training/train.py

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

# Import your custom modules
from src.data.dataset import BirdSongDataset
from src.models.bird_classifier import BirdClassifier
from src.utils.config import resolve_metadata_csv_path

def get_dataloaders(config, df):
    """Splits the dataframe and creates PyTorch DataLoaders."""
    batch_size = config['training']['batch_size']
    segment_size = config['audio']['segment_size']

    # Stratified split to ensure equal class representation in train and test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['scientific_name_id']
    )

    print(f"Data split: {len(train_df)} training samples, {len(test_df)} validation samples.")

    # Create Datasets
    train_dataset = BirdSongDataset(df=train_df, segment_size=segment_size, train=True)

    # Pass the train label map to test to ensure class IDs match perfectly
    test_dataset = BirdSongDataset(
        df=test_df,
        segment_size=segment_size,
        train=False,
        label_to_idx=train_dataset.label_to_idx
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def train_model(config):
    # 1. Setup device and directories
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    checkpoint_dir = Path(config['project_root']) / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Load Data
    csv_path = resolve_metadata_csv_path(config)

    df = pd.read_csv(csv_path)
    train_loader, test_loader = get_dataloaders(config, df)

    # 3. Initialize Model
    model = BirdClassifier(
        n_mels=config["audio"]["n_mels"],
        patch_size=config["model"]["patch_size"],
        embed_dim=config["model"]["embed_dim"],
        num_layers=config["model"]["num_layers"],
        heads=config["model"]["heads"],
        forward_expansion=config["model"]["forward_expansion"],
        dropout=config["model"]["dropout"],
        num_classes=config["data"]["num_classes"],
        time_steps=config["audio"]["segment_size"],
    ).to(device)

    # 4. Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    epochs = config['training']['epochs']

    best_val_acc = 0.0

    # 5. Training Loop
    print("\nStarting Training...")
    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm for a nice progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for mel_segments, labels in pbar:
            mel_segments, labels = mel_segments.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(mel_segments)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Tracking
            train_loss += loss.item() * mel_segments.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for mel_segments, labels in test_loader:
                mel_segments, labels = mel_segments.to(device), labels.to(device)
                logits = model(mel_segments)
                loss = criterion(logits, labels)

                val_loss += loss.item() * mel_segments.size(0)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved new best model with Val Acc: {val_acc:.4f}")

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")
