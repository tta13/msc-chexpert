#!/bin/bash
#SBATCH --job-name=crx_foundation_chexpert_embeddings
#SBATCH --mem 64G
#SBATCH -c 16
#SBATCH -p short-simple
#SBATCH --gpus=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tta@cin.ufpe.br

# Load module
module load Python3.10

# Print CUDA information
echo ""
echo "CUDA Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    nvcc --version 2>/dev/null || echo "nvcc not available"
else
    echo "nvidia-smi not available (running on CPU?)"
fi

# Activate virtual environment
ENV_NAME=.venv_tf
python -m venv $HOME/$ENV_NAME
source $HOME/$ENV_NAME/bin/activate
which python
pip install -r requirements_tf.txt

# Verify Tensorflow can see GPU
echo ""
echo "Tensorflow GPU check:"
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU'))); print(tf.config.list_physical_devices('GPU'))"

# Load configuration from .env file
echo ""
echo "Loading configuration from .env file..."
if [ -f ".env" ]; then
    # Export variables from .env file
    export $(grep -v '^#' .env | xargs)
    echo "✓ Configuration loaded from .env"
else
    echo "✗ ERROR: .env file not found!"
    echo ""
    echo "Please create a .env file with your training configuration:"
    echo "  cp .env.example .env"
    echo "  nano .env  # Edit with your settings"
    echo ""
    echo "See .env.example for available configuration options."
    exit 1
fi

echo ""
echo "================================================"
echo "Training Configuration:"
echo "Data Directory: $DATA_DIR"
echo "Save Directory: $SAVE_DIR"
echo "================================================"
echo ""

# Run embedding computation
python cxr_foundation_embeddings.py \
    --huggingface_hub_token "$HUGGINGFACE_HUB_TOKEN"\
    --data_dir "$DATA_DIR" \
    --output_dir "$SAVE_DIR" \

echo ""
echo "================================================"
echo "Job finished at: $(date)"
echo "================================================"
