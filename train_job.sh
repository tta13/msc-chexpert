#!/bin/bash
#SBATCH --job-name=tta_chexpert
#SBATCH --mem 32G
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
ENV_NAME=.venv
python -m venv $HOME/$ENV_NAME
source $HOME/$ENV_NAME/bin/activate
which python
pip install -r requirements.txt

# Verify PyTorch can see GPU
echo ""
echo "PyTorch GPU check:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Training configuration
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
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "K-Folds: $N_SPLITS"
echo "Device: $DEVICE"
echo "Data Directory: $DATA_DIR"
echo "================================================"
echo ""

# Run training
python train_model.py \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_splits $N_SPLITS \
    --device $DEVICE \
    --data_dir $DATA_DIR \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --save_dir ./checkpoints \
    --seed 42

echo ""
echo "================================================"
echo "Job finished at: $(date)"
echo "================================================"
