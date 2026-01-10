"""
Fine-tuning Script for CheXpert Dataset

This script supports training multiple architectures using TIMM library
with K-fold cross-validation on the CheXpert dataset.

Usage:
    python train_model.py --model efficientnetv2_rw_s --batch_size 32 --epochs 50
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)
from tqdm import tqdm
import timm

# Import custom data loader
from data_loader import CheXpertDataset, get_transforms


# Model registry for TIMM architectures
MODEL_REGISTRY = {
    # EfficientNet V2 (Ross Wightman's implementation - recommended)
    'efficientnetv2_rw_m': 'efficientnetv2_rw_m',
    
    # ResNet
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
    
    # ResNeXt
    'resnext50_32x4d': 'resnext50_32x4d',
    'resnext101_32x8d': 'resnext101_32x8d',
    
    # Vision Transformer
    'vit_base_patch16_224': 'vit_base_patch16_224',
    'vit_large_patch16_224': 'vit_large_patch16_224',
    
    # DeiT (Data-efficient Image Transformers)
    'deit_base_patch16_224': 'deit_base_patch16_224',
    
    # ConvNeXt
    'convnext_base': 'convnext_base',
}


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    """
    Initialize each DataLoader worker with a unique but reproducible seed.
    
    Args:
        worker_id: ID of the DataLoader worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Get a model from TIMM and modify for multi-label classification.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        Modified model ready for training
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODEL_REGISTRY.keys())}")
    
    timm_model_name = MODEL_REGISTRY[model_name]
    
    # Create model with TIMM
    model = timm.create_model(
        timm_model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    return model


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/auc
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class Trainer:
    """Trainer class for managing training, validation, and checkpointing."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object],
        save_dir: Path,
        model_name: str,
        checkpoint_interval: int = 10,
        early_stopping_patience: int = 10
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.model_name = model_name
        self.checkpoint_interval = checkpoint_interval
        self.best_loss = float('inf')
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        
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
    
    def validate(self, val_loader: DataLoader, dataset: CheXpertDataset) -> Tuple[float, pd.DataFrame]:
        """
        Validate the model and return predictions.
        
        Args:
            val_loader: Validation data loader
            dataset: Dataset object to get image paths
            
        Returns:
            Tuple of (loss, predictions_df)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc='Validation')):
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
                
                # Track indices to get paths later
                batch_size = images.size(0)
                start_idx = batch_idx * val_loader.batch_size
                all_indices.extend(range(start_idx, start_idx + batch_size))
        
        epoch_loss = running_loss / len(val_loader.dataset)
        
        # Concatenate all predictions
        all_probs = np.vstack(all_probs)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Create predictions DataFrame
        predictions_df = self._create_predictions_df(
            dataset, all_indices[:len(all_probs)], all_probs, all_preds
        )
        
        return epoch_loss, predictions_df, all_labels
    
    def _create_predictions_df(
        self, 
        dataset: CheXpertDataset, 
        indices: List[int], 
        probs: np.ndarray, 
        preds: np.ndarray
    ) -> pd.DataFrame:
        """Create DataFrame with predictions and probabilities."""
        # Get image paths
        paths = [dataset.data_frame.iloc[idx]['Path'] for idx in indices]
        
        # Create base DataFrame with paths
        df = pd.DataFrame({'Path': paths})
        
        # Add predictions and probabilities for each label
        for i, label in enumerate(dataset.labels):
            df[f'{label}_probability'] = probs[:, i]
            df[f'{label}_prediction'] = preds[:, i].astype(int)
        
        return df
    
    def save_checkpoint(
        self, 
        epoch: int, 
        fold: int, 
        train_loss: float, 
        val_loss: float
    ):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"{self.model_name}_fold{fold}_epoch{epoch}.pth"
        
        # Handle DataParallel wrapper
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'fold': fold,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")


def compute_metrics_for_label(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, label_name: str) -> Dict:
    """
    Compute classification metrics for a single label.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities
        label_name: Name of the label
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'label': label_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['roc_auc'] = 0.0
    
    return metrics


def train_kfold(
    data_dir: str,
    model_name: str,
    n_splits: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    save_dir: Path,
    checkpoint_interval: int,
    early_stopping_patience: int,
    scheduler_type: str,
    num_workers: int,
    seed: int = 42
):
    """
    Train model using K-fold cross-validation with full reproducibility.
    
    Reproducibility measures:
    - Fixed random seeds for PyTorch, NumPy, CUDA
    - Deterministic CUDNN operations
    - Worker initialization function for DataLoader
    - Generator for SubsetRandomSampler
    - Persistent workers to maintain state
    
    Args:
        data_dir: Directory containing the dataset
        model_name: Name of the model architecture
        n_splits: Number of folds for cross-validation
        epochs: Number of training epochs
        batch_size: Batch size for training (per GPU)
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        save_dir: Directory to save checkpoints and results
        checkpoint_interval: Save checkpoint every N epochs
        early_stopping_patience: Patience for early stopping
        scheduler_type: Type of learning rate scheduler
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility
    """
    set_seed(seed)
    
    # Check for multi-GPU
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    if n_gpus > 1:
        print(f"\n🚀 Multi-GPU training enabled: {n_gpus} GPUs detected")
        print(f"   Effective batch size: {batch_size} × {n_gpus} = {batch_size * n_gpus}")
    elif n_gpus == 1:
        print(f"\n💻 Single GPU training")
    else:
        print(f"\n💻 CPU training")
    
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
    print(f"Labels: {train_dataset.labels}")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"K-fold splits: {n_splits}")
    
    # Check for Pneumonia label
    pneumonia_idx = None
    if 'Pneumonia' in train_dataset.labels:
        pneumonia_idx = train_dataset.labels.index('Pneumonia')
        print(f"✓ Pneumonia label found at index {pneumonia_idx}")
    else:
        print("⚠ Warning: Pneumonia label not found in dataset")
    
    # Check if we have enough samples for K-fold
    min_samples_per_fold = batch_size * 2
    if len(train_dataset) < n_splits * min_samples_per_fold:
        print(f"\n⚠ WARNING: Training dataset may be too small for {n_splits}-fold CV")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Minimum recommended: {n_splits * min_samples_per_fold}")
        print(f"  Consider reducing n_splits or batch_size")
    
    # K-fold cross-validation on training set only
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Split indices based on training dataset
    dataset_indices = np.arange(len(train_dataset))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_indices)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"Train samples: {len(train_idx)}, Fold validation samples: {len(val_idx)}")
        print(f"{'='*70}")
        
        # Create a generator for this fold (for reproducibility)
        fold_generator = torch.Generator()
        fold_generator.manual_seed(seed + fold)  # Different seed per fold but reproducible
        
        # Create data samplers
        train_sampler = SubsetRandomSampler(train_idx, generator=fold_generator)
        val_sampler = SubsetRandomSampler(val_idx, generator=fold_generator)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,  # Ensure reproducibility
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        fold_val_loader = DataLoader(
            train_dataset_eval,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,  # Ensure reproducibility
            persistent_workers=True
        )
        
        # Initialize model
        model = get_model(model_name, num_classes, pretrained=True)
        if n_gpus > 1:
            model = nn.DataParallel()
        model = model.to(device)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        else:
            scheduler = None
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir / f"fold_{fold}",
            model_name=model_name,
            checkpoint_interval=checkpoint_interval,
            early_stopping_patience=early_stopping_patience
        )
        
        # Training loop
        best_fold_loss = float('inf')
        fold_start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_loss = trainer.train_epoch(train_loader)
            
            # Validate on fold validation set
            val_loss, _, val_labels = trainer.validate(fold_val_loader, train_dataset_eval)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Update learning rate scheduler
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint every N epochs (regardless of improvement)
            if epoch % checkpoint_interval == 0:
                trainer.save_checkpoint(epoch, fold, train_loss, val_loss, is_best=False)
            
            # Early stopping
            if trainer.early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                print(f"Best validation loss: {best_fold_loss:.4f}")
                break
        
        # Test on validation set (using as test set)
        print(f"\n{'='*70}")
        print(f"Testing Fold {fold + 1} on Validation Set")
        print(f"{'='*70}")
        
        test_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn  # Ensure reproducibility
        )
        
        test_loss, predictions_df, y_true = trainer.validate(test_loader, val_dataset)
        
        fold_time = time.time() - fold_start_time
        
        # Save predictions to CSV
        predictions_path = save_dir / f"fold_{fold}" / f"{model_name}_fold{fold}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\n✓ Predictions saved to: {predictions_path}, test loss: {test_loss}")
        
        # Compute and log metrics for Pneumonia label
        if pneumonia_idx is not None:
            pneumonia_true = y_true[:, pneumonia_idx]
            pneumonia_pred = predictions_df['Pneumonia_prediction'].values
            pneumonia_prob = predictions_df['Pneumonia_probability'].values
            
            pneumonia_metrics = compute_metrics_for_label(
                pneumonia_true, pneumonia_pred, pneumonia_prob, 'Pneumonia'
            )
            
            print(f"\n{'='*70}")
            print(f"Pneumonia Classification Metrics (Fold {fold + 1}):")
            print(f"{'='*70}")
            print(f"  Accuracy:  {pneumonia_metrics['accuracy']:.4f}")
            print(f"  Precision: {pneumonia_metrics['precision']:.4f}")
            print(f"  Recall:    {pneumonia_metrics['recall']:.4f}")
            print(f"  F1-Score:  {pneumonia_metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {pneumonia_metrics['roc_auc']:.4f}")
            print(f"{'='*70}")
        
        print(f"\nFold {fold + 1} Training Time: {fold_time:.2f}s ({fold_time/60:.2f}min)")
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"All predictions saved to: {save_dir}")
    print(f"{'='*70}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Fine-tune deep learning models on CheXpert dataset using TIMM'
    )
    
    # Model and data arguments
    parser.add_argument(
        '--model',
        type=str,
        default='efficientnetv2_rw_s',
        choices=list(MODEL_REGISTRY.keys()),
        help='Model architecture to use'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/chexpert_structured',
        help='Path to structured dataset directory'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
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
        '--weight_decay',
        type=float,
        default=0.0001,
        help='Weight decay for AdamW optimizer (default: 0.0001)'
    )
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        choices=['cosine', 'plateau', 'none'],
        help='Learning rate scheduler type (default: cosine)'
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10,
        help='Patience for early stopping (default: 10)'
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
        default=10,
        help='Save checkpoint every N epochs (default: 10)'
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
    
    # Performance
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of DataLoader workers (default: 8)'
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
