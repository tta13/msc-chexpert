"""
MedSigLIP Training Script for CheXpert Dataset (Refactored)

This script trains a classification head on top of MedSigLIP embeddings
using shared utilities from training_utils.py.

MedSigLIP is the vision encoder from MedGemma, specifically designed for medical imaging.

Usage:
    python train_medsiglip.py --batch_size 32 --epochs 50 --freeze_backbone
"""

import argparse
from pathlib import Path
from typing import List
import os

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import SiglipImageProcessor, SiglipVisionModel

from data_loader import CheXpertDataset, get_transforms
from training_utils import train_kfold, save_config, get_device, huggingface_login


# ============================================================================
# MODEL
# ============================================================================

class MedSigLIPClassifier(nn.Module):
    """Classification head on top of MedSigLIP embeddings."""
    
    def __init__(
        self,
        embedding_dim: int = 1152,
        num_classes: int = 14,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device

        # Load MedSigLIP
        print("Loading MedSigLIP model and processor from Hugging Face...")
        self.model_id = "google/medsiglip-448"
        self.backbone = SiglipVisionModel.from_pretrained(self.model_id).to(self.device)
        self.processor = SiglipImageProcessor.from_pretrained(self.model_id)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ MedSigLIP backbone frozen")
        else:
            print("✓ MedSigLIP backbone will be fine-tuned")
        
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
        
        print(f"✓ Classification head initialized")
    
    def forward(self, x):
        # Get MedSigLIP image embeddings
        inputs = self.processor(images=x, padding="max_length", return_tensors="pt").to(self.device)
        if self.training and not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                outputs = self.backbone(**inputs)
        else:
            outputs = self.backbone(**inputs)
        
        # outputs is already pooled: [batch_size, embedding_dim]
        embeddings = outputs["pooler_output"] / outputs["pooler_output"].norm(p=2, dim=-1, keepdim=True)
        
        # Classification
        logits = self.classifier(embeddings)
        return logits


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train classification head on MedSigLIP embeddings'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/CheXpert-v1.0-small')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--huggingface_hub_token', type=str, default=None)
    
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
    
    # Huggingface login
    huggingface_login(args.huggingface_hub_token)
        
    # Create datasets
    data_dir = Path(args.data_dir)
    
    train_dataset = CheXpertDataset(
        csv_path=data_dir / 'train.csv',
        root_dir=data_dir.parent,
        transform=transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]),
        policy='ones'
    )
    
    val_dataset = CheXpertDataset(
        csv_path=data_dir / 'valid.csv',
        root_dir=data_dir.parent,
        transform=transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]),
        policy='ones'
    )
    
    # Model factory
    def model_factory(num_classes):
        return MedSigLIPClassifier(
            embedding_dim=1152,
            num_classes=num_classes,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
            device=device
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
