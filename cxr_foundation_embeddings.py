"""
Compute CXR Foundation ELIXR v2.0 Embeddings

This script precomputes 32x768 dimensional embeddings for the CheXpert dataset
using the CXR Foundation model (TensorFlow) and saves them as numpy arrays.

Usage:
    python compute_cxr_embeddings.py --data_dir ./data/CheXpert-v1.0-small --output_dir ./embeddings
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from huggingface_hub import snapshot_download


def load_cxr_foundation_model():
    """
    Load the CXR Foundation model from Hugging Face.
    
    Returns:
        TensorFlow SavedModel for inference
    """
    print("Downloading CXR Foundation model from Hugging Face...")
        
    model_dir = snapshot_download(
        repo_id="google/cxr-foundation", local_dir='./checkpoints/cxr_foundation/hf',
        allow_patterns="elixr-c-v2-pooled/*"
    )
    
    # The model is in the elixr-c-v2-pooled subdirectory
    saved_model_dir = Path(model_dir) / "elixr-c-v2-pooled"
    
    print(f"Loading model from {saved_model_dir}...")
    model = tf.saved_model.load(str(saved_model_dir))
    
    print("✓ CXR Foundation model loaded successfully")
    return model


def preprocess_image(image_path: Path, target_size: int = 512) -> np.ndarray:
    """
    Preprocess a chest X-ray image for CXR Foundation.
    
    Args:
        image_path: Path to the image
        target_size: Target size for resizing (CXR Foundation uses 512x512)
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    image = image.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    return image_array


def encode_image_to_png_bytes(image_array: np.ndarray) -> bytes:
    """
    Encode image array to PNG bytes (required by CXR Foundation).
    
    Args:
        image_array: Image as numpy array [H, W, 3]
        
    Returns:
        PNG encoded bytes
    """
    # Convert back to uint8
    image_uint8 = (image_array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    image_pil = Image.fromarray(image_uint8)
    
    # Encode as PNG bytes
    import io
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()
    
    return png_bytes


def create_tf_example(png_bytes: bytes) -> tf.train.Example:
    """
    Create a tf.Example from PNG bytes (CXR Foundation input format).
    
    Args:
        png_bytes: PNG encoded image bytes
        
    Returns:
        tf.train.Example
    """
    feature = {
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[png_bytes]))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def compute_embedding(model, image_path: Path) -> np.ndarray:
    """
    Compute CXR Foundation embedding for a single image.
    
    Args:
        model: CXR Foundation TensorFlow model
        image_path: Path to the image
        
    Returns:
        Embedding array of shape [32, 768]
    """
    # Preprocess image
    image_array = preprocess_image(image_path)
    
    # Encode to PNG bytes
    png_bytes = encode_image_to_png_bytes(image_array)
    
    # Create tf.Example
    tf_example = create_tf_example(png_bytes)
    
    # Serialize the example
    serialized_example = tf_example.SerializeToString()
    
    # Run inference
    # The model expects a batch of serialized tf.Examples
    embeddings = model(tf.constant([serialized_example]))
    
    # Extract the embedding (shape: [1, 32, 768])
    embedding = embeddings['output'].numpy()[0]  # Shape: [32, 768]
    
    return embedding


def compute_embeddings_for_split(
    model,
    csv_path: Path,
    root_dir: Path,
    output_path: Path
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for all images in a dataset split.
    
    Args:
        model: CXR Foundation model
        csv_path: Path to CSV with image paths
        root_dir: Root directory containing images
        output_path: Path to save embeddings
        
    Returns:
        Dictionary with paths and embeddings
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"\nProcessing {len(df)} images from {csv_path.name}...")
    
    # Initialize arrays
    embeddings_list = []
    paths_list = []
    
    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing embeddings"):
        img_path = root_dir / row['Path']
        
        try:
            # Compute embedding
            embedding = compute_embedding(model, img_path)
            
            embeddings_list.append(embedding)
            paths_list.append(row['Path'])
            
        except Exception as e:
            print(f"\n⚠ Error processing {img_path}: {e}")
            continue
    
    # Convert to numpy arrays
    embeddings_array = np.array(embeddings_list)  # Shape: [N, 32, 768]
    paths_array = np.array(paths_list)
    
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
        description='Compute CXR Foundation embeddings for CheXpert dataset'
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
        default='./checkpoints/cxr_foundation/embeddings',
        help='Directory to save computed embeddings'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid'],
        help='Dataset splits to process (default: train valid)'
    )    
    parser.add_argument('--huggingface_hub_token', type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    root_dir = data_dir.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CXR Foundation Embedding Computation")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Splits to process: {args.splits}")
    print("="*70)

    # Huggingface login
    huggingface_login(args.huggingface_hub_token)
    
    # Load model
    model = load_cxr_foundation_model()
    
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
            csv_path=csv_path,
            root_dir=root_dir,
            output_path=output_path
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
    
    print("\n✓ Ready for training! Use train_cxr_foundation.py with these embeddings.")


if __name__ == "__main__":
    main()
