"""
MedSigLIP Training Script (Precomputed Embeddings)

This script trains a classification head on precomputed MedSigLIP embeddings
using shared utilities from training_utils.py.

The embeddings (1152-dim) must be precomputed using compute_medsiglip_embeddings.py first.

Usage:
    python train_medsiglip.py --embeddings_dir ./embeddings/medsiglip --batch_size 128 --epochs 50
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from training_utils import train_kfold, save_config, get_device


# ============================================================================
# DATASET
# ============================================================================

class MedSigLIPEmbeddingsDataset(Dataset):
    """PyTorch Dataset for precomputed MedSigLIP embeddings."""
    
    PATHOLOGY_LABELS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    def __init__(
        self,
        embeddings_path: Path,
        csv_path: Path,
        policy: str = 'ones',
        selected_labels: List[str] = None
    ):
        """
        Initialize dataset with precomputed embeddings.
        
        Args:
            embeddings_path: Path to .npz file with embeddings
            csv_path: Path to CSV with labels
            policy: How to handle uncertain labels
            selected_labels: Specific labels to use
        """
        # Load embeddings
        print(f"Loading embeddings from {embeddings_path}...")
        data = np.load(embeddings_path)
        self.embeddings = data['embeddings']  # Shape: [N, 1152]
        self.embedding_paths = data['paths']
        
        print(f"Loaded {len(self.embeddings)} embeddings with shape {self.embeddings.shape}")
        
        # Load labels
        self.data_frame = pd.read_csv(csv_path)
        
        # Determine labels
        self.labels = selected_labels if selected_labels else self.PATHOLOGY_LABELS
        self.labels = [label for label in self.labels if label in self.data_frame.columns]
        
        # Process labels
        for label in self.labels:
            if label in self.data_frame.columns:
                if policy == 'ones':
                    self.data_frame[label] = self.data_frame[label].replace(-1, 1)
                elif policy == 'zeros':
                    self.data_frame[label] = self.data_frame[label].replace(-1, 0)
                self.data_frame[label] = self.data_frame[label].fillna(0)
        
        # Create path to index mapping
        self.path_to_idx = {path: idx for idx, path in enumerate(self.embedding_paths)}
        
        # Filter data_frame to only include samples with embeddings
        self.data_frame = self.data_frame[
            self.data_frame['Path'].isin(self.embedding_paths)
        ].reset_index(drop=True)
        
        print(f"Dataset with {len(self)} samples")
        print(f"Using {len(self.labels)} labels: {self.labels}")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get path and find corresponding embedding
        path = self.data_frame.iloc[idx]['Path']
        emb_idx = self.path_to_idx[path]
        
        # Get embedding [1152]
        embedding = self.embeddings[emb_idx]
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
        # Get labels
        labels = self.data_frame.iloc[idx][self.labels].values.astype(np.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return embedding_tensor, labels_tensor


# ============================================================================
# MODEL
# ============================================================================

class MedSigLIPClassifier(nn.Module):
    """Classification head on top of precomputed MedSigLIP embeddings."""
    
    def __init__(
        self,
        embedding_dim: int = 1152,
        num_classes: int = 14,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Build classification head
        layers = []
        in_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
        print(f"✓ MedSigLIP classifier initialized")
        print(f"  Input dim: {embedding_dim}")
        print(f"  Hidden dims: {hidden_dims}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Precomputed embeddings [batch_size, 1152]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        logits = self.classifier(x)
        return logits


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train classification head on precomputed MedSigLIP embeddings'
    )
    
    # Data arguments
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default='./embeddings/medsiglip',
        help='Directory containing precomputed embeddings'
    )
    parser.add_argument('--data_dir', type=str, default='./data/CheXpert-v1.0-small')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256])
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/medsiglip')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    # Performance
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, save_dir)
    
    # Check embeddings exist
    embeddings_dir = Path(args.embeddings_dir)
    train_embeddings = embeddings_dir / 'train_embeddings.npz'
    valid_embeddings = embeddings_dir / 'valid_embeddings.npz'
    
    if not train_embeddings.exists() or not valid_embeddings.exists():
        print("\n" + "="*70)
        print("ERROR: Precomputed embeddings not found!")
        print("="*70)
        print(f"Expected files:")
        print(f"  - {train_embeddings}")
        print(f"  - {valid_embeddings}")
        print(f"\nPlease run compute_medsiglip_embeddings.py first:")
        print(f"  python compute_medsiglip_embeddings.py --data_dir {args.data_dir} --output_dir {args.embeddings_dir}")
        print("="*70)
        return
    
    # Create datasets
    data_dir = Path(args.data_dir)
    
    train_dataset = MedSigLIPEmbeddingsDataset(
        embeddings_path=train_embeddings,
        csv_path=data_dir / 'train.csv',
        policy='ones'
    )
    
    val_dataset = MedSigLIPEmbeddingsDataset(
        embeddings_path=valid_embeddings,
        csv_path=data_dir / 'valid.csv',
        policy='ones'
    )
    
    # Model factory
    def model_factory(num_classes):
        return MedSigLIPClassifier(
            embedding_dim=1152,
            num_classes=num_classes,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout
        )
    
    # Train with K-fold
    train_kfold(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_factory=model_factory,
        model_name="medsiglip",
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=save_dir,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_type=args.scheduler,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()