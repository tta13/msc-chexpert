"""
Compute MedSigLIP Embeddings

This script precomputes 1152-dimensional embeddings for the CheXpert dataset
using MedSigLIP and saves them as numpy arrays for faster training.

Usage:
    python compute_medsiglip_embeddings.py --data_dir ./data/CheXpert-v1.0-small --output_dir ./embeddings
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SiglipVisionModel, SiglipImageProcessor


def load_medsiglip_model(device: str = 'cuda'):
    """
    Load MedSigLIP model and processor.
    
    Args:
        device: Device to run model on
        
    Returns:
        Tuple of (model, processor)
    """
    model_id = "google/medsiglip-448"

    print("Loading MedSigLIP model from Hugging Face...")
    
    processor = SiglipImageProcessor.from_pretrained(model_id)
    
    model = SiglipVisionModel.from_pretrained(model_id)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ MedSigLIP model loaded on {device}")
    return model, processor


@torch.no_grad()
def compute_embeddings_batch(
    model, 
    processor, 
    image_paths: list, 
    device: str = 'cuda',
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute embeddings for a batch of images.
    
    Args:
        model: MedSigLIP model
        processor: MedSigLIP processor
        image_paths: List of image paths
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Embeddings array of shape [N, 1152]
    """
    all_embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load images
        images = [Image.open(path).convert('RGB') for path in batch_paths]
        
        # Process batch
        inputs = processor(images=images, return_tensors="pt")
        
        # Get embeddings
        outputs = model(**inputs)
        embeddings = outputs["pooler_output"] / outputs["pooler_output"].norm(p=2, dim=-1, keepdim=True)
        
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)


def compute_embeddings_for_split(
    model,
    processor,
    csv_path: Path,
    root_dir: Path,
    output_path: Path,
    device: str = 'cuda',
    batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for all images in a dataset split.
    
    Args:
        model: MedSigLIP model
        processor: MedSigLIP processor
        csv_path: Path to CSV with image paths
        root_dir: Root directory containing images
        output_path: Path to save embeddings
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with paths and embeddings
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"\nProcessing {len(df)} images from {csv_path.name}...")
    
    # Get full image paths
    image_paths = [root_dir / path for path in df['Path']]
    
    # Compute embeddings in batches
    embeddings_list = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Computing embeddings"):
        batch_paths = image_paths[i:i + batch_size]
        
        try:
            # Load images
            images = [Image.open(path).convert('RGB') for path in batch_paths]
            
            # Process batch
            inputs = processor(images=images, return_tensors="pt")
            
            # Get embeddings
            outputs = model(**inputs)
            embeddings = outputs["pooler_output"] / outputs["pooler_output"].norm(p=2, dim=-1, keepdim=True)
            
            embeddings_list.append(embeddings)
            
        except Exception as e:
            print(f"\n⚠ Error processing batch {i}-{i+len(batch_paths)}: {e}")
            # Process individually if batch fails
            for path in batch_paths:
                try:
                    embedding = compute_embedding(model, processor, path, device)
                    embeddings_list.append(embedding.reshape(1, -1))
                except Exception as e2:
                    print(f"\n⚠ Error processing {path}: {e2}")
    
    # Concatenate all embeddings
    embeddings_array = np.vstack(embeddings_list)  # Shape: [N, 1152]
    paths_array = np.array([str(path) for path in df['Path']])
    
    print(f"✓ Computed {len(embeddings_array)} embeddings")
    print(f"  Embeddings shape: {embeddings_array.shape}")
    
    # Save embeddings and paths
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        embeddings=embeddings_array,
        paths=paths_array
    )
    
    print(f"✓ Saved embeddings to: {output_path}")
    
    return {
        'embeddings': embeddings_array,
        'paths': paths_array
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Compute MedSigLIP embeddings for CheXpert dataset'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/CheXpert-v1.0-small',
        help='Path to CheXpert dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoints/medsiglip/embeddings',
        help='Directory to save computed embeddings'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid'],
        help='Dataset splits to process (default: train valid)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for embedding computation (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("MedSigLIP Embedding Computation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits to process: {args.splits}")
    print("="*70)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    root_dir = data_dir.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, processor = load_medsiglip_model(device=args.device)
    
    # Process each split
    for split in args.splits:
        csv_path = data_dir / f"{split}.csv"
        
        if not csv_path.exists():
            print(f"\n⚠ Warning: {csv_path} not found, skipping...")
            continue
        
        output_path = output_dir / f"{split}_embeddings.npz"
        
        # Compute embeddings
        compute_embeddings_for_split(
            model=model,
            processor=processor,
            csv_path=csv_path,
            root_dir=root_dir,
            output_path=output_path,
            device=args.device,
            batch_size=args.batch_size
        )
    
    print("\n" + "="*70)
    print("✓ All embeddings computed successfully!")
    print("="*70)
    print("\nEmbedding files:")
    for split in args.splits:
        output_path = output_dir / f"{split}_embeddings.npz"
        if output_path.exists():
            # Load and show info
            data = np.load(output_path)
            print(f"  {split}: {output_path}")
            print(f"    - Embeddings shape: {data['embeddings'].shape}")
            print(f"    - Number of samples: {len(data['paths'])}")
    
    print("\n✓ Ready for training! Use the updated train_medsiglip.py with these embeddings.")
    print("="*70)


if __name__ == "__main__":
    main()
