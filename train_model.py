"""
Fine-tuning Script for CheXpert Dataset

This script supports training multiple architectures (EfficientNetV2, ResNet, ViT, DeiT)
with K-fold cross-validation on the CheXpert dataset.

Usage:
    python train_model.py --model efficientnet_v2_s --batch_size 32 --epochs 100
"""

import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
from tqdm import tqdm

# Import custom data loader
from data_loader import CheXpertDataset, get_transforms


# Model registry for easy architecture switching
MODEL_REGISTRY = {
    'efficientnet_v2_s': 'torchvision.models.efficientnet_v2_s',
    'efficientnet_v2_m': 'torchvision.models.efficientnet_v2_m',
    'efficientnet_v2_l': 'torchvision.models.efficientnet_v2_l',
    'resnet50': 'torchvision.models.resnet50',
    'resnet101': 'torchvision.models.resnet101',
    'vit_b_16': 'torchvision.models.vit_b_16',
    'vit_b_32': 'torchvision.models.vit_b_32',
    'vit_l_16': 'torchvision.models.vit_l_16',
}


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Get a model from the registry and modify for multi-label classification.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Modified model ready for training
    """
    import torchvision.models as models
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_REGISTRY.keys())}")
    
    # Load pretrained model
    if 'efficientnet' in model_name:
        if model_name == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(pretrained=pretrained)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m(pretrained=pretrained)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif model_name == 'efficientnet_v2_l':
            model = models.efficientnet_v2_l(pretrained=pretrained)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            
    elif 'resnet' in model_name:
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif 'vit' in model_name:
        if model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_b_32':
            model = models.vit_b_32(pretrained=pretrained)
        elif model_name == 'vit_l_16':
            model = models.vit_l_16(pretrained=pretrained)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    
    return model


class Trainer:
    """Trainer class for managing training, validation, and checkpointing."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        save_dir: Path,
        model_name: str,
        checkpoint_interval: int = 20
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.model_name = model_name
        self.checkpoint_interval = checkpoint_interval
        self.best_loss = float('inf')
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                # Get predictions (using sigmoid for multi-label)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        all_probs = np.vstack(all_probs)
        
        return epoch_loss, all_preds, all_labels, all_probs
    
    def save_checkpoint(self, epoch: int, fold: int, train_loss: float, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"{self.model_name}_fold{fold}_epoch{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'fold': fold,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """
    Compute classification metrics for multi-label classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Compute metrics (micro and macro averaging for multi-label)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # ROC-AUC (handle cases where all labels are 0 or 1)
    try:
        metrics['roc_auc_micro'] = roc_auc_score(y_true, y_prob, average='micro')
        metrics['roc_auc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError:
        metrics['roc_auc_micro'] = 0.0
        metrics['roc_auc_macro'] = 0.0
    
    return metrics


def train_kfold(
    data_dir: str,
    model_name: str,
    n_splits: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    save_dir: Path,
    checkpoint_interval: int,
    seed: int = 42
):
    """
    Train model using K-fold cross-validation.
    
    Args:
        data_dir: Directory containing the dataset
        model_name: Name of the model architecture
        n_splits: Number of folds for cross-validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints and results
        checkpoint_interval: Save checkpoint every N epochs
        seed: Random seed for reproducibility
    """
    set_seed(seed)
    
    # Load full training dataset
    data_dir = Path(data_dir)
    
    # Create separate datasets for train and validation transforms
    train_dataset = CheXpertDataset(
        csv_path=data_dir / 'train.csv',
        root_dir=data_dir.parent,
        transform=get_transforms(image_size=224, is_training=True),
        policy='ones'
    )
    
    # Create a version with validation transforms for fold validation
    train_dataset_eval = CheXpertDataset(
        csv_path=data_dir / 'train.csv',
        root_dir=data_dir.parent,
        transform=get_transforms(image_size=224, is_training=False),
        policy='ones'
    )
    
    # Get validation dataset for final testing
    val_dataset = CheXpertDataset(
        csv_path=data_dir / 'valid.csv',
        root_dir=data_dir.parent,
        transform=get_transforms(image_size=224, is_training=False),
        policy='ones'
    )
    
    num_classes = len(train_dataset.labels)
    print(f"\nTraining {model_name} on {num_classes} labels")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"K-fold splits: {n_splits}")
    
    # Check if we have enough samples for K-fold
    min_samples_per_fold = batch_size * 2  # At least 2 batches per fold
    if len(train_dataset) < n_splits * min_samples_per_fold:
        print(f"\n⚠ WARNING: Training dataset may be too small for {n_splits}-fold CV")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Minimum recommended: {n_splits * min_samples_per_fold}")
        print(f"  Consider reducing n_splits or batch_size")
    
    # K-fold cross-validation on training set only
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Store results for all folds
    all_results = []
    
    # Split indices based on training dataset
    dataset_indices = np.arange(len(train_dataset))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_indices)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"Train samples: {len(train_idx)}, Fold validation samples: {len(val_idx)}")
        print(f"{'='*70}")
        
        # Create data samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,  # Use training transforms
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        fold_val_loader = DataLoader(
            train_dataset_eval,  # Use validation transforms for fold validation
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        model = get_model(model_name, num_classes, pretrained=True)
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            save_dir=save_dir / f"fold_{fold}",
            model_name=model_name,
            checkpoint_interval=checkpoint_interval
        )
        
        # Training loop
        best_fold_loss = float('inf')
        fold_start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss = trainer.train_epoch(train_loader)
            
            # Validate on fold validation set
            val_loss, _, _, _ = trainer.validate(fold_val_loader)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save checkpoint if improved and at checkpoint interval
            if epoch % checkpoint_interval == 0 and train_loss < best_fold_loss:
                best_fold_loss = train_loss
                trainer.save_checkpoint(epoch, fold, train_loss, val_loss)
        
        # Test on validation set (using as test set)
        print(f"\n{'='*70}")
        print(f"Testing Fold {fold + 1} on Validation Set")
        print(f"{'='*70}")
        
        test_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loss, y_pred, y_true, y_prob = trainer.validate(test_loader)
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_prob)
        
        fold_time = time.time() - fold_start_time
        
        # Store results
        result = {
            'model': model_name,
            'fold': fold + 1,
            'checkpoint': f'fold{fold}_epoch{epochs}',
            'accuracy': metrics['accuracy'],
            'precision_micro': metrics['precision_micro'],
            'precision_macro': metrics['precision_macro'],
            'recall_micro': metrics['recall_micro'],
            'recall_macro': metrics['recall_macro'],
            'f1_micro': metrics['f1_micro'],
            'f1_macro': metrics['f1_macro'],
            'roc_auc_micro': metrics['roc_auc_micro'],
            'roc_auc_macro': metrics['roc_auc_macro'],
            'test_loss': test_loss,
            'training_time_seconds': fold_time
        }
        
        all_results.append(result)
        
        # Print results
        print(f"\nFold {fold + 1} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (micro/macro): {metrics['precision_micro']:.4f} / {metrics['precision_macro']:.4f}")
        print(f"  Recall (micro/macro): {metrics['recall_micro']:.4f} / {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (micro/macro): {metrics['f1_micro']:.4f} / {metrics['f1_macro']:.4f}")
        print(f"  ROC-AUC (micro/macro): {metrics['roc_auc_micro']:.4f} / {metrics['roc_auc_macro']:.4f}")
        print(f"  Training Time: {fold_time:.2f}s")
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_path = save_dir / f"{model_name}_results.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {results_path}")
    print(f"{'='*70}")
    
    # Print average metrics across folds
    print(f"\nAverage Metrics Across {n_splits} Folds:")
    for col in results_df.columns:
        if col not in ['model', 'fold', 'checkpoint']:
            print(f"  {col}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Fine-tune deep learning models on CheXpert dataset'
    )
    
    # Model and data arguments
    parser.add_argument(
        '--model',
        type=str,
        default='efficientnet_v2_s',
        choices=list(MODEL_REGISTRY.keys()),
        help='Model architecture to use'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Path to structured dataset directory'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)'
    )
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on (default: cuda if available)'
    )
    
    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=20,
        help='Save checkpoint every N epochs (default: 20)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints and results'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")
    print(f"\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Start training
    train_kfold(
        data_dir=args.data_dir,
        model_name=args.model,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=save_dir,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed
    )
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
