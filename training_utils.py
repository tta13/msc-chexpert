"""
Training Utilities for CheXpert Classification

This module contains shared code for training different model architectures
on the CheXpert dataset with consistent methodology.

Shared components:
- Reproducibility functions
- Early stopping
- Trainer class
- Metrics computation
- K-fold training loop
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

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
    f1_score, roc_auc_score, average_precision_score
)
from tqdm import tqdm
from huggingface_hub import login


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
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

# ============================================================================
# HUGGINGFACE SETUP
# ============================================================================

def huggingface_login(token: str):
    """
    Login in HuggingFace Hub for model access.
    
    Args:
        token: HuggingFace Hub token
    """
    if token is not None:
        login(token=token)
    else:
        print("No HUGGINGFACE_HUB_TOKEN found. Make sure you're logged in via 'huggingface-cli login'")


# ============================================================================
# EARLY STOPPING
# ============================================================================

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


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics_for_label(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    label_name: str
) -> Dict:
    """
    Compute classification metrics for a single label.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities
        label_name: Name of the label
        
    Returns:
        Dictionary of metrics including accuracy, precision, recall, F1, ROC-AUC, and AP
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
    
    # Average Precision
    try:
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['average_precision'] = 0.0
    
    return metrics


def compute_mean_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute mean Average Precision (mAP) across all classes.
    
    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_prob: Predicted probabilities [n_samples, n_classes]
        
    Returns:
        mAP score (macro-averaged AP)
    """
    aps = []
    for i in range(y_true.shape[1]):
        try:
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
            aps.append(ap)
        except ValueError:
            # Skip if no positive samples
            continue
    
    return np.mean(aps) if aps else 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    predictions_df: pd.DataFrame,
    labels: List[str]
) -> Tuple[float, List[Dict]]:
    """
    Compute all metrics including mAP and per-label metrics.
    
    Args:
        y_true: Ground truth labels [n_samples, n_classes]
        y_prob: Predicted probabilities [n_samples, n_classes]
        predictions_df: DataFrame with predictions
        labels: List of label names
        
    Returns:
        Tuple of (mAP, list of per-label metrics)
    """
    # Compute mAP
    map_score = compute_mean_average_precision(y_true, y_prob)
    
    # Compute per-label metrics
    all_metrics = []
    for i, label in enumerate(labels):
        label_true = y_true[:, i]
        label_pred = predictions_df[f'{label}_prediction'].values
        label_prob = predictions_df[f'{label}_probability'].values
        
        metrics = compute_metrics_for_label(label_true, label_pred, label_prob, label)
        all_metrics.append(metrics)
    
    return map_score, all_metrics


def display_pneumonia_metrics(metrics_list: List[Dict], fold: int):
    """
    Display Pneumonia-specific metrics.
    
    Args:
        metrics_list: List of metric dictionaries
        fold: Current fold number
    """
    # Find Pneumonia metrics
    pneumonia_metrics = None
    for metrics in metrics_list:
        if metrics['label'] == 'Pneumonia':
            pneumonia_metrics = metrics
            break
    
    if pneumonia_metrics:
        print(f"\n{'='*70}")
        print(f"Pneumonia Classification Metrics (Fold {fold}):")
        print(f"{'='*70}")
        print(f"  Accuracy:  {pneumonia_metrics['accuracy']:.4f}")
        print(f"  Precision: {pneumonia_metrics['precision']:.4f}")
        print(f"  Recall:    {pneumonia_metrics['recall']:.4f}")
        print(f"  F1-Score:  {pneumonia_metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {pneumonia_metrics['roc_auc']:.4f}")
        print(f"  AP:        {pneumonia_metrics['average_precision']:.4f}")
        print(f"{'='*70}")


# ============================================================================
# TRAINER CLASS
# ============================================================================

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
        """
        Args:
            model: PyTorch model
            device: Device to train on
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            save_dir: Directory to save checkpoints
            model_name: Name of the model (for checkpoint naming)
            checkpoint_interval: Save checkpoint every N epochs
            early_stopping_patience: Patience for early stopping
        """
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
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss
    
    def validate(
        self, 
        val_loader: DataLoader, 
        dataset
    ) -> Tuple[float, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Validate the model and return predictions.
        
        Args:
            val_loader: Validation data loader
            dataset: Dataset object (to get image paths and labels)
            
        Returns:
            Tuple of (loss, predictions_df, all_labels, all_probs)
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
                
                # Get predictions
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                batch_size = images.size(0)
                start_idx = batch_idx * val_loader.batch_size
                all_indices.extend(range(start_idx, start_idx + batch_size))
        
        epoch_loss = running_loss / len(val_loader.dataset)
        
        all_probs = np.vstack(all_probs)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        predictions_df = self._create_predictions_df(
            dataset, all_indices[:len(all_probs)], all_probs, all_preds
        )
        
        return epoch_loss, predictions_df, all_labels, all_probs
    
    def _create_predictions_df(
        self, 
        dataset, 
        indices: List[int], 
        probs: np.ndarray, 
        preds: np.ndarray
    ) -> pd.DataFrame:
        """
        Create DataFrame with predictions and probabilities.
        
        Args:
            dataset: Dataset object
            indices: List of sample indices
            probs: Predicted probabilities
            preds: Predicted labels
            
        Returns:
            DataFrame with paths, predictions, and probabilities
        """
        paths = [dataset.data_frame.iloc[idx]['Path'] for idx in indices]
        df = pd.DataFrame({'Path': paths})
        
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
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            fold: Current fold
            train_loss: Training loss
            val_loss: Validation loss
        """
        checkpoint_path = self.save_dir / f"{self.model_name}_fold{fold}_epoch{epoch}.pth"
        
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


# ============================================================================
# K-FOLD TRAINING LOOP
# ============================================================================

def train_kfold(
    train_dataset,
    val_dataset,
    model_factory: Callable,
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
    
    This is a generic K-fold training function that works with any model architecture.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (used as test set)
        model_factory: Function that creates a new model instance
        model_name: Name of the model (for saving)
        n_splits: Number of folds
        epochs: Number of epochs per fold
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        save_dir: Directory to save results
        checkpoint_interval: Save checkpoint every N epochs
        early_stopping_patience: Early stopping patience
        scheduler_type: Type of LR scheduler ('cosine', 'plateau', 'none')
        num_workers: Number of DataLoader workers
        seed: Random seed
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
    
    # K-fold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dataset_indices = np.arange(len(train_dataset))
    
    # Store all fold results
    all_fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_indices)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"Train samples: {len(train_idx)}, Fold validation samples: {len(val_idx)}")
        print(f"{'='*70}")
        
        fold_generator = torch.Generator()
        fold_generator.manual_seed(seed + fold)
        
        train_sampler = SubsetRandomSampler(train_idx, generator=fold_generator)
        val_sampler = SubsetRandomSampler(val_idx, generator=fold_generator)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        fold_val_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # Initialize model using factory function
        model = model_factory(num_classes)
        
        if n_gpus > 1:
            model = nn.DataParallel(model)
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
        fold_start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            train_loss = trainer.train_epoch(train_loader)
            val_loss, _, val_labels, val_probs = trainer.validate(fold_val_loader, train_dataset)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
            
            if epoch % checkpoint_interval == 0:
                trainer.save_checkpoint(epoch, fold, train_loss, val_loss)
            
            if trainer.early_stopping(val_loss):
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                break
        
        # Save final checkpoint
        trainer.save_checkpoint(epoch, fold, train_loss, val_loss)
        
        # Test on validation set
        print(f"\n{'='*70}")
        print(f"Testing Fold {fold + 1} on Validation Set")
        print(f"{'='*70}")
        
        test_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        
        test_loss, predictions_df, y_true, y_prob = trainer.validate(test_loader, val_dataset)
        
        fold_time = time.time() - fold_start_time
        
        # Save predictions
        predictions_path = save_dir / f"fold_{fold}" / f"{model_name}_fold{fold}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"\n✓ Predictions saved to: {predictions_path}, test loss: {test_loss:.4f}")
        
        # Compute metrics
        map_score, all_metrics = compute_all_metrics(y_true, y_prob, predictions_df, val_dataset.labels)
        
        print(f"\n{'='*70}")
        print(f"Mean Average Precision (mAP): {map_score:.4f}")
        print(f"{'='*70}")
        
        # Save metrics
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = save_dir / f"fold_{fold}" / f"{model_name}_fold{fold}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Metrics saved to: {metrics_path}")
        
        # Display Pneumonia metrics if available
        if pneumonia_idx is not None:
            display_pneumonia_metrics(all_metrics, fold + 1)
        
        print(f"\nFold {fold + 1} Training Time: {fold_time:.2f}s ({fold_time/60:.2f}min)")
        
        # Store fold results
        all_fold_metrics.append({
            'fold': fold + 1,
            'test_loss': float(test_loss),
            'map': float(map_score),
            'metrics': all_metrics
        })
    
    # Compute and save overall statistics
    print(f"\n{'='*70}")
    print(f"K-Fold Cross-Validation Summary")
    print(f"{'='*70}")
    
    fold_maps = [fm['map'] for fm in all_fold_metrics]
    print(f"Mean mAP across folds: {np.mean(fold_maps):.4f} ± {np.std(fold_maps):.4f}")
    
    # Save summary
    summary = {
        'mean_map': float(np.mean(fold_maps)),
        'std_map': float(np.std(fold_maps)),
        'fold_results': all_fold_metrics
    }
    
    summary_path = save_dir / 'cross_validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Cross-validation summary saved to: {summary_path}")
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"All results saved to: {save_dir}")
    print(f"{'='*70}")


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def save_config(args, save_dir: Path):
    """
    Save training configuration to JSON.
    
    Args:
        args: Argument namespace from argparse
        save_dir: Directory to save config
    """
    config = vars(args)
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")
    print(f"\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def get_device(device_name: str) -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_name: 'cuda' or 'cpu'
        
    Returns:
        PyTorch device
    """
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    return torch.device(device_name)
