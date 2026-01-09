"""
CheXpert Dataset Loader for PyTorch

This module provides a PyTorch Dataset class for loading CheXpert data
for training deep learning models.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for CheXpert chest X-ray data.
    
    Handles loading images and labels with customizable preprocessing.
    """
    
    # Default pathology labels in CheXpert
    PATHOLOGY_LABELS = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices'
    ]
    
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        transform: Optional[Callable] = None,
        policy: str = 'ones',
        selected_labels: Optional[List[str]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the CSV file with labels
            root_dir: Root directory containing the images
            transform: Optional transform to apply to images
            policy: How to handle uncertain labels (-1):
                    'ones' - Map to 1 (positive)
                    'zeros' - Map to 0 (negative)
                    'ignore' - Keep as -1
            selected_labels: List of specific labels to use (uses all if None)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.policy = policy
        
        # Load CSV
        self.data_frame = pd.read_csv(csv_path)
        
        # Determine which labels to use
        self.labels = selected_labels if selected_labels else self.PATHOLOGY_LABELS
        self.labels = [label for label in self.labels if label in self.data_frame.columns]
        
        # Process labels according to policy
        self._process_labels()
        
        print(f"Loaded dataset with {len(self)} samples")
        print(f"Using {len(self.labels)} labels: {self.labels}")
    
    def _process_labels(self) -> None:
        """Process uncertain labels according to the chosen policy."""
        for label in self.labels:
            if label in self.data_frame.columns:
                if self.policy == 'ones':
                    # Map uncertain (-1) to positive (1)
                    self.data_frame[label] = self.data_frame[label].replace(-1, 1)
                elif self.policy == 'zeros':
                    # Map uncertain (-1) to negative (0)
                    self.data_frame[label] = self.data_frame[label].replace(-1, 0)
                # 'ignore' policy keeps -1 as is
                
                # Fill NaN values with 0
                self.data_frame[label] = self.data_frame[label].fillna(0)
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, labels) tensors
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path
        img_path = self.data_frame.iloc[idx]['Path']
        img_path = self.root_dir / img_path
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get labels
        labels = self.data_frame.iloc[idx][self.labels].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, labels
    
    def get_label_distribution(self) -> pd.DataFrame:
        """
        Get the distribution of positive cases for each label.
        
        Returns:
            DataFrame with label statistics
        """
        stats = []
        for label in self.labels:
            positive_count = (self.data_frame[label] == 1).sum()
            negative_count = (self.data_frame[label] == 0).sum()
            stats.append({
                'Label': label,
                'Positive': positive_count,
                'Negative': negative_count,
                'Positive %': f"{positive_count / len(self) * 100:.2f}%"
            })
        
        return pd.DataFrame(stats)


def get_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Get default image transforms for CheXpert data.
    
    Args:
        image_size: Target size for images
        is_training: Whether transforms are for training (includes augmentation)
        
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    selected_labels: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Root directory containing structured data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for loading
        image_size: Target image size
        selected_labels: Specific labels to use (uses all if None)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = CheXpertDataset(
        csv_path=os.path.join(data_dir, 'train.csv'),
        root_dir=data_dir.parent,
        transform=get_transforms(image_size, is_training=True),
        policy='ones',
        selected_labels=selected_labels
    )
    
    val_dataset = CheXpertDataset(
        csv_path=data_dir / 'valid.csv',
        root_dir=data_dir.parent,
        transform=get_transforms(image_size, is_training=False),
        policy='ones',
        selected_labels=selected_labels
    )
    
    # test_dataset = CheXpertDataset(
    #     csv_path=data_dir / 'test_labels.csv',
    #     root_dir=data_dir.parent,
    #     transform=get_transforms(image_size, is_training=False),
    #     policy='ones',
    #     selected_labels=selected_labels
    # )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )
    
    return train_loader, val_loader#, test_loader


if __name__ == "__main__":
    # Example usage
    data_dir = "./data/chexpert_structured"
    
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=16,
        num_workers=2,
        image_size=224
    )
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Validation: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    
    # Show label distribution
    print("\nLabel distribution in training set:")
    print(train_loader.dataset.get_label_distribution())
    
    # Test loading a batch
    print("\nLoading a test batch...")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
