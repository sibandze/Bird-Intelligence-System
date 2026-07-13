import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.data.dataset import BirdSongDataset
from src.models.bird_classifier import BirdClassifier
from src.utils.config import resolve_metadata_csv_path

def evaluate_model(config):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Running Evaluation on device: {device}")

    # 1. Load preprocessed metadata
    csv_path = resolve_metadata_csv_path(config)
    df = pd.read_csv(csv_path)
    
    # Recreate the exact split mapping used in training
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['scientific_name_id']
    )
    
    # Map back IDs to names for clean presentation
    id_to_name = dict(zip(df['scientific_name_id'], df['scientific_name']))
    target_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    # 2. Setup Dataloader
    test_dataset = BirdSongDataset(
        df=test_df, 
        segment_size=config['audio']['segment_size'], 
        train=False
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. Initialize and Load Model Weights
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

    model_path = Path(config['project_root']) / "checkpoints" / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {model_path}. Train the model first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Inference loop
    all_preds = []
    all_labels = []

    print("Extracting predictions...")
    with torch.no_grad():
        for mel_segments, labels in test_loader:
            mel_segments = mel_segments.to(device)
            logits = model(mel_segments)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Metrics Calculations
    print("\n================ CLASSIFICATION REPORT ================")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    print("=======================================================")

    # 6. Plotting Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Bird Species Classification Confusion Matrix')
    plt.ylabel('True Species')
    plt.xlabel('Predicted Species')
    plt.tight_layout()
    
    # Save chart image to root workspace
    output_image = Path(config['project_root']) / "confusion_matrix.png"
    plt.savefig(output_image)
    print(f"Saved evaluation matrix visualization to: {output_image}\n")
