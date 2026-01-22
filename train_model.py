"""
Training Script for Standard Architectures (TIMM) - Refactored

This script trains standard deep learning architectures using TIMM library
with shared utilities from training_utils.py.

Supports: EfficientNet, ResNet, ResNeXt, Vision Transformers, DeiT, ConvNeXt

Usage:
    python train_model.py --model efficientnetv2_rw_s --batch_size 32 --epochs 50
"""

import argparse
from pathlib import Path

import torch.nn as nn
import timm

from data_loader import CheXpertDataset, get_transforms
from training_utils import train_kfold, save_config, get_device


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY = {
    # EfficientNet V2 (Ross Wightman's implementation - recommended)
    'efficientnetv2_rw_s': 'efficientnetv2_rw_s',
    'efficientnetv2_rw_m': 'efficientnetv2_rw_m',
    'efficientnetv2_rw_l': 'efficientnetv2_rw_l',
    
    # EfficientNet V2 (Original Google)
    'tf_efficientnetv2_s': 'tf_efficientnetv2_s',
    'tf_efficientnetv2_m': 'tf_efficientnetv2_m',
    'tf_efficientnetv2_l': 'tf_efficientnetv2_l',
    
    # ResNet
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
    'resnet152': 'resnet152',
    
    # ResNeXt
    'resnext50_32x4d': 'resnext50_32x4d',
    'resnext101_32x8d': 'resnext101_32x8d',
    'resnext101_64x4d': 'resnext101_64x4d',
    
    # Vision Transformer
    'vit_base_patch16_224': 'vit_base_patch16_224',
    'vit_base_patch32_224': 'vit_base_patch32_224',
    'vit_large_patch16_224': 'vit_large_patch16_224',
    
    # DeiT (Data-efficient Image Transformers)
    'deit_base_patch16_224': 'deit_base_patch16_224',
    'deit_small_patch16_224': 'deit_small_patch16_224',
    'deit_tiny_patch16_224': 'deit_tiny_patch16_224',
    'deit_base_distilled_patch16_224': 'deit_base_distilled_patch16_224',
    'deit_small_distilled_patch16_224': 'deit_small_distilled_patch16_224',
    
    # ConvNeXt
    'convnext_tiny': 'convnext_tiny',
    'convnext_small': 'convnext_small',
    'convnext_base': 'convnext_base',
    'convnext_large': 'convnext_large',
}


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


# ============================================================================
# MAIN
# ============================================================================

def main():
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
        default='./data/CheXpert-v1.0-small',
        help='Path to structured dataset directory'
    )
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of K-folds')
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        choices=['cosine', 'plateau', 'none'],
        help='Learning rate scheduler type'
    )
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Performance
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(args, save_dir)
    
    # Create datasets
    data_dir = Path(args.data_dir)
    
    train_dataset = CheXpertDataset(
        csv_path=data_dir / 'train.csv',
        root_dir=data_dir.parent,
        transform=get_transforms(image_size=224, is_training=True),
        policy='ones'
    )
    
    val_dataset = CheXpertDataset(
        csv_path=data_dir / 'valid.csv',
        root_dir=data_dir.parent,
        transform=get_transforms(image_size=224, is_training=False),
        policy='ones'
    )
    
    # Create model factory function
    def model_factory(num_classes):
        return get_model(args.model, num_classes, pretrained=True)
    
    # Train with K-fold cross-validation
    train_kfold(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_factory=model_factory,
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
