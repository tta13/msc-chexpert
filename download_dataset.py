"""
CheXpert Dataset Download and Structure Script

This script downloads the CheXpert dataset and organizes it for deep learning training.
CheXpert is a large chest X-ray dataset for automated chest radiograph interpretation.

Usage:
    python download_chexpert.py --output_dir ./data --dataset_type small
"""

import os
import argparse
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import shutil
from typing import Optional


class CheXpertDownloader:
    """Handler for downloading and structuring the CheXpert dataset."""
    
    # CheXpert dataset URLs (you'll need to register and get actual download links)
    DATASET_URLS = {
        'small': 'https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2',
        'full': 'https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c'
    }
    
    # CheXpert test set labels from GitHub
    TEST_SET_LABELS_URL = 'https://raw.githubusercontent.com/rajpurkarlab/cheXpert-test-set-labels/master/groundtruth.csv'
    
    # Kaggle dataset information
    KAGGLE_DATASET = 'ashery/chexpert'
    
    def __init__(self, output_dir: str = './data', dataset_type: str = 'small', use_kaggle: bool = False):
        """
        Initialize the downloader.
        
        Args:
            output_dir: Directory to save the dataset
            dataset_type: 'small' or 'full' dataset version
            use_kaggle: Whether to use Kaggle API for download
        """
        self.output_dir = Path(output_dir)
        self.dataset_type = dataset_type
        self.use_kaggle = use_kaggle
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, destination: Path) -> None:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            destination: Path to save the file
        """
        print(f"Downloading to {destination}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> None:
        """
        Extract a zip archive.
        
        Args:
            archive_path: Path to the zip file
            extract_to: Directory to extract to
        """
        print(f"Extracting {archive_path}...")
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete!")
    
    def download_test_labels(self) -> bool:
        """
        Download the official CheXpert test set labels from GitHub.
        
        Returns:
            True if successful, False otherwise
        """
        print("\nDownloading CheXpert test set labels from GitHub...")
        
        test_labels_path = self.output_dir / "chexpert_structured" / "test_labels.csv"
        test_labels_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            response = requests.get(self.TEST_SET_LABELS_URL, timeout=30)
            response.raise_for_status()
            
            with open(test_labels_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Test labels downloaded successfully to: {test_labels_path}")
            
            # Validate the downloaded file
            df = pd.read_csv(test_labels_path)
            print(f"  - Test set contains {len(df)} samples")
            print(f"  - Columns: {', '.join(df.columns.tolist())}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error downloading test labels: {e}")
            print(f"  You can manually download from: {self.TEST_SET_LABELS_URL}")
            return False
        except Exception as e:
            print(f"✗ Error processing test labels: {e}")
            return False
    
    def download_from_kaggle(self) -> bool:
        """
        Download CheXpert dataset from Kaggle using Kaggle API.
        
        Returns:
            True if successful, False otherwise
        """
        print("="*70)
        print("Downloading CheXpert dataset from Kaggle...")
        print("="*70)
        
        try:
            import kaggle
        except ImportError:
            print("\n✗ Kaggle API not installed!")
            print("Install it with: pip install kaggle")
            return False
        
        # Check for Kaggle credentials
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_config.exists():
            print("\n✗ Kaggle credentials not found!")
            print("\nTo set up Kaggle API:")
            print("1. Go to https://www.kaggle.com/settings")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New Token' or copy your username and key")
            print("4. Create the file: ~/.kaggle/kaggle.json")
            print("5. Add your credentials in JSON format:")
            print('   {')
            print('     "username": "your_kaggle_username",')
            print('     "key": "your_kaggle_api_key"')
            print('   }')
            print("6. On Linux/Mac, set permissions: chmod 600 ~/.kaggle/kaggle.json")
            print("\nSee the kaggle.json sample file in the project for reference.")
            return False
        
        # Download dataset
        download_path = self.output_dir / "kaggle_download"
        download_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\nDownloading from: {self.KAGGLE_DATASET}")
            print(f"Saving to: {download_path}")
            print("\nThis may take a while depending on your connection...")
            
            kaggle.api.dataset_download_files(
                self.KAGGLE_DATASET,
                path=str(download_path),
                unzip=True,
                quiet=False
            )
            
            print("\n✓ Download completed successfully!")
            
            # Move downloaded files to chexpert_raw
            raw_dir = self.output_dir / "chexpert_raw"
            if raw_dir.exists():
                print(f"\nRemoving existing raw directory: {raw_dir}")
                shutil.rmtree(raw_dir)
            
            # Move the downloaded content
            if download_path.exists():
                shutil.move(str(download_path), str(raw_dir))
                print(f"✓ Moved dataset to: {raw_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error downloading from Kaggle: {e}")
            print("\nMake sure you have:")
            print("1. Accepted the dataset terms on Kaggle website")
            print("2. Proper Kaggle API credentials configured")
            return False
    
    def download_dataset(self) -> None:
        """
        Download the CheXpert dataset.
        
        Uses Kaggle API if use_kaggle is True, otherwise shows manual instructions.
        """
        if self.use_kaggle:
            success = self.download_from_kaggle()
            if success:
                print("\n✓ Dataset downloaded successfully from Kaggle!")
                return
            else:
                print("\n✗ Kaggle download failed. Please try manual download.")
                self.use_kaggle = False
        
        # Manual download instructions
        print(f"Preparing to download CheXpert {self.dataset_type} dataset...")
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD OPTIONS:")
        print("\nOption 1 - Kaggle (Recommended):")
        print("1. Visit: https://www.kaggle.com/datasets/ashery/chexpert")
        print("2. Click 'Download' button (you may need to accept terms)")
        print("3. Extract the downloaded file")
        print("4. Place contents in:", self.output_dir / "chexpert_raw")
        print("\nOption 2 - Official Website:")
        print("1. Visit: https://stanfordmlgroup.github.io/competitions/chexpert/")
        print("2. Register and agree to the data use agreement")
        print("3. Download the dataset")
        print("="*70 + "\n")
        
        dataset_path = self.output_dir / f"chexpert_{self.dataset_type}.zip"
        extract_dir = self.output_dir / "chexpert_raw"
        
        if dataset_path.exists() and not extract_dir.exists():
            self.extract_archive(dataset_path, extract_dir)
        elif extract_dir.exists():
            print(f"✓ Raw data already exists at: {extract_dir}")
        else:
            print(f"Waiting for dataset download...")
            print(f"Once downloaded, run this script again to structure the data.")
    
    def structure_dataset(self) -> None:
        """
        Structure the dataset for training and testing.
        Creates organized directories and generates split CSV files.
        """
        print("Structuring dataset for deep learning...")
        
        raw_dir = self.output_dir / "chexpert_raw"
        structured_dir = self.output_dir / "chexpert_structured"
        
        if not raw_dir.exists():
            print(f"Raw data directory not found: {raw_dir}")
            print("Please download and extract the dataset first.")
            return
        
        # Create structured directories
        train_dir = structured_dir / "train"
        valid_dir = structured_dir / "valid"
        test_dir = structured_dir / "test"
        
        for dir_path in [train_dir, valid_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Download test set labels from GitHub
        print("\n" + "="*70)
        self.download_test_labels()
        print("="*70)
        
        # Look for CSV files in raw directory
        csv_files = list(raw_dir.glob("**/*.csv"))
        
        if not csv_files:
            print("No CSV files found in raw directory.")
            return
        
        print(f"\nFound {len(csv_files)} CSV files in raw directory")
        
        # Process each CSV file
        for csv_file in csv_files:
            self._process_csv(csv_file, raw_dir, structured_dir)
        
        print("\nDataset structuring complete!")
        print(f"Structured data saved to: {structured_dir}")
    
    def _process_csv(self, csv_path: Path, raw_dir: Path, structured_dir: Path) -> None:
        """
        Process a CSV file and organize images.
        
        Args:
            csv_path: Path to the CSV file
            raw_dir: Raw data directory
            structured_dir: Structured output directory
        """
        print(f"\nProcessing {csv_path.name}...")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} records")
            
            # Determine split type from filename
            if 'train' in csv_path.name.lower():
                split = 'train'
            elif 'valid' in csv_path.name.lower():
                split = 'valid'
            else:
                split = 'test'
            
            # Save processed CSV
            output_csv = structured_dir / f"{split}_labels.csv"
            df.to_csv(output_csv, index=False)
            print(f"Saved labels to: {output_csv}")
            
            # Print dataset statistics
            print(f"\nDataset Statistics for {split}:")
            print(f"Total samples: {len(df)}")
            
            # Show label distributions if available
            label_columns = [col for col in df.columns if col not in ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
            if label_columns:
                print("\nLabel distributions (positive cases):")
                for col in label_columns[:5]:  # Show first 5 labels
                    if col in df.columns:
                        positive_count = (df[col] == 1).sum()
                        print(f"  {col}: {positive_count} ({positive_count/len(df)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
    
    def generate_split_info(self) -> None:
        """Generate summary information about the dataset splits."""
        structured_dir = self.output_dir / "chexpert_structured"
        
        info = {
            'train': structured_dir / "train_labels.csv",
            'valid': structured_dir / "valid_labels.csv",
            'test': structured_dir / "test_labels.csv"
        }
        
        summary = []
        for split, csv_path in info.items():
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                summary.append(f"{split.capitalize()}: {len(df)} samples")
        
        if summary:
            print("\n" + "="*50)
            print("Dataset Summary:")
            print("\n".join(summary))
            print("="*50)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Download and structure CheXpert dataset for deep learning'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Output directory for the dataset (default: ./data)'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='small',
        choices=['small', 'full'],
        help='Dataset version to download (default: small)'
    )
    parser.add_argument(
        '--structure_only',
        action='store_true',
        help='Only structure existing data, skip download'
    )
    parser.add_argument(
        '--download_test_labels',
        action='store_true',
        help='Download only the test set labels from GitHub'
    )
    parser.add_argument(
        '--use_kaggle',
        action='store_true',
        help='Download dataset from Kaggle using Kaggle API'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = CheXpertDownloader(
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        use_kaggle=args.use_kaggle
    )
    
    # If only downloading test labels
    if args.download_test_labels:
        downloader.download_test_labels()
        return
    
    # Download dataset (if not structure_only)
    if not args.structure_only:
        downloader.download_dataset()
    
    # Structure the dataset
    downloader.structure_dataset()
    
    # Generate summary
    downloader.generate_split_info()
    
    print("\n✓ Process complete!")


if __name__ == "__main__":
    main()
