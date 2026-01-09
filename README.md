# CheXpert Dataset Downloader and Structurer

A Python project for training deep learning architectures on the CheXpert dataset.

## Overview

CheXpert is a large public dataset for chest radiograph interpretation, consisting of 224,316 chest radiographs of 65,240 patients. This project provides tools to download and structure the dataset for easy use in deep learning workflows.

## Requirements

- Python 3.10.8
- See `requirements.txt` for all dependencies

## Installation

### Option 1: Using Conda (Recommended)

**For Local Machines:**

1. Create the conda environment from the YML file:

```bash
conda env create -f environment.yml
```

2. Activate the environment:

```bash
conda activate chexpert
```

**For SLURM Clusters:**

The `environment.yml` file is designed to work with any CUDA version on your cluster. Use the provided setup script:

```bash
# Make the script executable
chmod +x slurm_setup.sh

# Run the setup (detects CUDA and installs appropriate PyTorch)
./slurm_setup.sh
```

This script will:
- Detect your cluster's CUDA version
- Create the conda environment
- Install the correct PyTorch build for your CUDA version
- Verify the installation

### Option 2: Using pip with venv

1. Create a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python --version  # Should show Python 3.10.8
python -c "import torch; print(torch.__version__)"  # Should show PyTorch 2.0.1
python -c "import torch; print(torch.cuda.is_available())"  # Check GPU availability
```

## Dataset Download

The CheXpert dataset can be downloaded from multiple sources:

### Option 1: Kaggle (Recommended - Automated)

1. **Get your Kaggle API credentials:**
   - Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
   - Scroll to the "API" section
   - Click "Create New Token" or copy your existing username and key
   
2. **Create kaggle.json file:**
   - Create the file: `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<YourUsername>\.kaggle\kaggle.json` (Windows)
   - Add your credentials in this format:
   ```json
   {
     "username": "your_kaggle_username",
     "key": "your_kaggle_api_key"
   }
   ```
   - On Linux/Mac, set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Accept the dataset terms:**
   - Visit [https://www.kaggle.com/datasets/ashery/chexpert](https://www.kaggle.com/datasets/ashery/chexpert)
   - Click "Download" and accept the terms (one-time)

4. **Run the download script:**
   ```bash
   python download_chexpert.py --use_kaggle
   ```

### Option 2: Manual Download

1. Visit [https://www.kaggle.com/datasets/ashery/chexpert](https://www.kaggle.com/datasets/ashery/chexpert)
2. Download the dataset manually
3. Extract to `./data/chexpert_raw/`
4. Run: `python download_chexpert.py --structure_only`

### Option 3: Official Website (Currently Unavailable)

The official CheXpert website ([https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)) is currently not working. Please use Kaggle instead.

**Note:** The test set labels are automatically downloaded from the [official GitHub repository](https://github.com/rajpurkarlab/cheXpert-test-set-labels).

## Usage

### Basic Usage

**Automated download from Kaggle (recommended):**

```bash
python download_chexpert.py --use_kaggle --output_dir ./data
```

**Manual download and structure:**

```bash
# After manually downloading and extracting to ./data/chexpert_raw/
python download_chexpert.py --structure_only --output_dir ./data
```

### Command Line Arguments

- `--output_dir`: Directory to save the dataset (default: `./data`)
- `--dataset_type`: Choose between `small` or `full` dataset (default: `small`)
- `--use_kaggle`: Download dataset from Kaggle using Kaggle API (recommended)
- `--structure_only`: Only structure existing data, skip download step
- `--download_test_labels`: Download only the test set labels from GitHub

### Examples

1. **Download from Kaggle with automated API:**
```bash
python download_chexpert.py --use_kaggle
```

2. **Manual download workflow:**
```bash
# After manually downloading to ./data/chexpert_raw/
python download_chexpert.py --structure_only
```

3. **Specify custom output directory:**
```bash
python download_chexpert.py --use_kaggle --output_dir ./my_data
```

4. **Download only test set labels:**
```bash
python download_chexpert.py --download_test_labels
```

## Project Structure

```
chexpert-downloader/
├── download_chexpert.py    # Main script for downloading and structuring
├── data_loader.py          # PyTorch Dataset class for loading data
├── train_model.py          # Training script with K-fold cross-validation
├── requirements.txt        # pip dependencies
├── environment.yml         # Conda environment specification
├── .env.example            # Example configuration file (copy to .env)
├── kaggle.json            # Kaggle API credentials (sample - DO NOT COMMIT)
├── slurm_setup.sh         # SLURM cluster setup script
├── train_job.slurm        # SLURM job submission script
├── .gitignore             # Git ignore file
├── README.md              # This file
└── data/                  # Output directory (created when running script)
    ├── chexpert_raw/      # Raw downloaded data
    └── chexpert_structured/  # Organized data for training
        ├── train/
        ├── valid/
        ├── test/
        ├── train_labels.csv
        ├── valid_labels.csv
        └── test_labels.csv
```

## Dataset Structure

After running the script, your data will be organized as:

- **train_labels.csv**: Training set labels and metadata
- **valid_labels.csv**: Validation set labels and metadata
- **test_labels.csv**: Test set labels and metadata

Each CSV contains:
- Image paths
- Patient demographics (age, sex)
- View information (Frontal/Lateral, AP/PA)
- Labels for 14 observations (pathologies)

## CheXpert Labels

The dataset includes labels for 14 observations:

1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Opacity
5. Lung Lesion
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices

Labels use the following convention:
- `1.0`: Positive
- `0.0`: Negative
- `-1.0`: Uncertain
- `NaN`: Not mentioned

## Using the Dataset for Training

Example code to load the structured dataset:

```python
import pandas as pd
from pathlib import Path

# Load training data
train_df = pd.read_csv('data/chexpert_structured/train_labels.csv')

# Access image paths and labels
image_paths = train_df['Path']
labels = train_df[['Cardiomegaly', 'Edema', 'Consolidation']]  # Select specific labels

print(f"Training samples: {len(train_df)}")
print(f"Label columns: {labels.columns.tolist()}")
```

## Citation

If you use the CheXpert dataset, please cite:

```
@inproceedings{irvin2019chexpert,
  title={Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison},
  author={Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Shpanskaya, Katie and others},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={33},
  number={01},
  pages={590--597},
  year={2019}
}
```

## License

This project is provided as-is for educational purposes. The CheXpert dataset itself is subject to its own license and data use agreement. Please review the terms at the official CheXpert website.

## Contributing

Feel free to submit issues or pull requests to improve this tool.

## Support

For questions about the dataset, visit the [CheXpert website](https://stanfordmlgroup.github.io/competitions/chexpert/).

For issues with this tool, please open an issue in the repository.

## Running on SLURM Clusters

### Setup

1. **Load required modules** (adjust based on your cluster):
```bash
module load anaconda3
# module load cuda/11.8  # Optional - usually auto-detected
```

2. **Run setup script**:
```bash
chmod +x slurm_setup.sh
./slurm_setup.sh
```

3. **Prepare your data**:
```bash
# Download dataset (with Kaggle API already configured)
python download_chexpert.py --use_kaggle --output_dir ./data
```

4. **Configure training parameters**:
```bash
# Copy the example configuration file
cp .env.example .env

# Edit .env with your preferred settings
nano .env  # or vim .env
```

### Submit Training Job

1. **Edit the SLURM script** (`train_job.slurm`) if needed:
   - Adjust `#SBATCH` parameters for your cluster
   - Set partition name (check with `sinfo`)
   - Configure GPU requirements
   - Modify module loads if needed

2. **The training configuration is now in `.env` file**:
```bash
# Example .env content:
MODEL=efficientnet_v2_s
EPOCHS=100
BATCH_SIZE=32
DEVICE=cuda
```

3. **Submit the job**:
```bash
sbatch train_job.slurm
```

4. **Monitor the job**:
```bash
# Check job status
squeue -u $USER

# View output logs
tail -f logs/train_<job_id>.out

# View error logs
tail -f logs/train_<job_id>.err
```

### Training Different Models

Simply edit the `.env` file:
```bash
# For EfficientNetV2-Medium
MODEL=efficientnet_v2_m
BATCH_SIZE=16
EPOCHS=100

# For Vision Transformer
MODEL=vit_b_16
BATCH_SIZE=16
LEARNING_RATE=0.00005

# For ResNet50
MODEL=resnet50
BATCH_SIZE=64
EPOCHS=100
```

Then submit:
```bash
sbatch train_job.slurm
```

### Multiple Experiments

Run multiple experiments with different configurations:
```bash
# Create different config files
cp .env.example .env.efficientnet
cp .env.example .env.resnet
cp .env.example .env.vit

# Edit each file with different settings
# ...

# Submit jobs with different configs
cp .env.efficientnet .env && sbatch train_job.slurm
cp .env.resnet .env && sbatch train_job.slurm
cp .env.vit .env && sbatch train_job.slurm
```

### CUDA Version Notes

- **You don't need to specify CUDA version** - the setup script detects it automatically
- PyTorch will use whatever CUDA is available on the compute node
- If you get CUDA errors, check: `nvidia-smi` and `nvcc --version` on the compute node
