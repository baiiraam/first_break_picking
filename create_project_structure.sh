#!/bin/bash

# ============================================================
# SEISMIC FBP PROJECT - FOLDER STRUCTURE CREATOR
# ============================================================
# This script creates the complete folder structure for the
# seismic first break picking project.
#
# Usage:
#   chmod +x create_project_structure.sh
#   ./create_project_structure.sh
# ============================================================

set -e  # Exit on error

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}SEISMIC FBP PROJECT - FOLDER STRUCTURE CREATOR${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Get the project root (where this script is run from)
PROJECT_ROOT=$(pwd)
echo -e "${GREEN}Creating project structure in:${NC} $PROJECT_ROOT"
echo ""

# --- Function to create directory with message ---
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo -e "  ${GREEN}✅ Created:${NC} $1"
    else
        echo -e "  ${YELLOW}⚠️  Already exists:${NC} $1"
    fi
}

# --- Function to create file with content ---
create_file() {
    if [ ! -f "$1" ]; then
        echo -e "$2" > "$1"
        echo -e "  ${GREEN}✅ Created:${NC} $1"
    else
        echo -e "  ${YELLOW}⚠️  Already exists:${NC} $1"
    fi
}

# ============================================================
# DATA DIRECTORY
# ============================================================
echo -e "${BLUE}📁 Creating data directories...${NC}"
create_dir "data/raw"
create_dir "data/chunks"
create_dir "data/chunks/Halfmile"
create_dir "data/chunks/Brunswick"
create_dir "data/chunks/Lalor"
create_dir "data/chunks/Sudbury"
create_dir "data/processed"

# ============================================================
# SOURCE CODE DIRECTORY (src/)
# ============================================================
echo -e "${BLUE}📁 Creating source code directories...${NC}"

# Main src
create_dir "src"
create_file "src/__init__.py" "# Seismic FBP package"

# Config
create_dir "src/config"
create_file "src/config/__init__.py" "# Config module"

# Preprocessing
create_dir "src/preprocessing"
create_file "src/preprocessing/__init__.py" "# Preprocessing module"
create_file "src/preprocessing/chunker.py" "# Chunk assignment logic\n\nclass Chunker:\n    pass"
create_file "src/preprocessing/processor.py" "# Shot processing logic\n\nclass Processor:\n    pass"
create_file "src/preprocessing/writer.py" "# Chunk writing logic\n\nclass Writer:\n    pass"
create_file "src/preprocessing/manifest.py" "# Manifest generation\n\nclass Manifest:\n    pass"

# Data
create_dir "src/data"
create_file "src/data/__init__.py" "# Data module"
create_file "src/data/chunked_dataset.py" "# ChunkedSeismicDataset class\n\nclass ChunkedSeismicDataset:\n    pass"
create_file "src/data/cache.py" "# LRU cache management\n\nclass CacheManager:\n    pass"
create_file "src/data/hdf5_dataset.py" "# HDF5SeismicDataset class\n\nclass HDF5SeismicDataset:\n    pass"

# Models
create_dir "src/models"
create_file "src/models/__init__.py" "# Models module"
create_file "src/models/unet.py" "# U-Net architecture\n\nimport torch.nn as nn\n\nclass UNet(nn.Module):\n    pass"

# Training
create_dir "src/training"
create_file "src/training/__init__.py" "# Training module"
create_file "src/training/trainer.py" "# SeismicTrainer class\n\nclass SeismicTrainer:\n    pass"
create_file "src/training/metrics.py" "# Evaluation metrics\n\nclass Metrics:\n    pass"
create_file "src/training/callbacks.py" "# Callbacks for training\n\nclass Callbacks:\n    pass"

# Utils
create_dir "src/utils"
create_file "src/utils/__init__.py" "# Utils module"
create_file "src/utils/logger.py" "# Loguru configuration\n\nimport loguru\n\nlogger = loguru.logger"
create_file "src/utils/mlflow_utils.py" "# MLflow helpers\n\nclass MLflowManager:\n    pass"
create_file "src/utils/tensorboard_utils.py" "# TensorBoard helpers\n\nclass TensorBoardManager:\n    pass"
create_file "src/utils/hdf5_utils.py" "# HDF5 helper functions\n\ndef load_hdf5():\n    pass"

# ============================================================
# SCRIPTS DIRECTORY
# ============================================================
echo -e "${BLUE}📁 Creating scripts...${NC}"
create_dir "scripts"
create_file "scripts/preprocess.py" "#!/usr/bin/env python3\n\"\"\"Run preprocessing pipeline.\"\"\"\n\nif __name__ == \"__main__\":\n    print(\"Running preprocessing...\")"
create_file "scripts/train.py" "#!/usr/bin/env python3\n\"\"\"Run training pipeline.\"\"\"\n\nif __name__ == \"__main__\":\n    print(\"Running training...\")"
create_file "scripts/evaluate.py" "#!/usr/bin/env python3\n\"\"\"Run evaluation.\"\"\"\n\nif __name__ == \"__main__\":\n    print(\"Running evaluation...\")"
create_file "scripts/visualize.py" "#!/usr/bin/env python3\n\"\"\"Run visualization.\"\"\"\n\nif __name__ == \"__main__\":\n    print(\"Running visualization...\")"
create_file "scripts/export_model.py" "#!/usr/bin/env python3\n\"\"\"Export model to ONNX/TorchScript.\"\"\"\n\nif __name__ == \"__main__\":\n    print(\"Exporting model...\")"

# Make scripts executable
chmod +x scripts/*.py

# ============================================================
# CONFIG DIRECTORY
# ============================================================
echo -e "${BLUE}📁 Creating config files...${NC}"
create_dir "configs"
create_file "configs/default.yaml" "# Default configuration\n\ndataset_name: \"default\"\ntarget_traces: 1578\nn_samples: 751\nstrip_width: 8\nbatch_size: 4\nlearning_rate: 0.001\nn_epochs: 30\ndevice: \"mps\""
create_file "configs/halfmile.yaml" "# Halfmile configuration\n\ndataset_name: \"Halfmile\"\nhdf5_path: \"data/raw/Halfmile3D_add_geom_sorted.hdf5\"\nchunk_dir: \"data/chunks\"\ntarget_traces: 1578\nn_samples: 751\nstrip_width: 8\nchunk_size: 69\nrandom_seed: 42\ntrain_split: 0.8\nval_split: 0.1\ntest_split: 0.1\n\n# Training\nbatch_size: 4\nlearning_rate: 0.001\nn_epochs: 30\ndevice: \"mps\"\nlr_scheduler: \"plateau\"\ngradient_clip_value: 1.0\nearly_stopping_patience: 5"
create_file "configs/brunswick.yaml" "# Brunswick configuration\n\ndataset_name: \"Brunswick\"\nhdf5_path: \"data/raw/Brunswick_orig_1500ms_V2.hdf5\"\nchunk_dir: \"data/chunks\""
create_file "configs/lalor.yaml" "# Lalor configuration\n\ndataset_name: \"Lalor\"\nhdf5_path: \"data/raw/Lalor_raw_z_1500ms_norp_geom_v3.hdf5\"\nchunk_dir: \"data/chunks\""
create_file "configs/sudbury.yaml" "# Sudbury configuration\n\ndataset_name: \"Sudbury\"\nhdf5_path: \"data/raw/preprocessed_Sudbury3D.hdf\"\nchunk_dir: \"data/chunks\""
create_file "configs/production.yaml" "# Production configuration\n\n# Override for production runs"
create_file "configs/experiment_001.yaml" "# Experiment 001\n\n# Override for specific experiment"

# ============================================================
# CHECKPOINTS DIRECTORY
# ============================================================
echo -e "${BLUE}📁 Creating checkpoint directories...${NC}"
create_dir "checkpoints"
create_dir "checkpoints/Halfmile"
create_dir "checkpoints/Brunswick"
create_dir "checkpoints/Lalor"
create_dir "checkpoints/Sudbury"

# ============================================================
# RUNS DIRECTORY (TensorBoard)
# ============================================================
echo -e "${BLUE}📁 Creating TensorBoard directories...${NC}"
create_dir "runs"
create_dir "runs/Halfmile"
create_dir "runs/Brunswick"
create_dir "runs/Lalor"
create_dir "runs/Sudbury"

# ============================================================
# MLFLOW DIRECTORY
# ============================================================
echo -e "${BLUE}📁 Creating MLflow directory...${NC}"
create_dir "mlruns"

# ============================================================
# LOGS DIRECTORY
# ============================================================
echo -e "${BLUE}📁 Creating logs directory...${NC}"
create_dir "logs"

# ============================================================
# TESTS DIRECTORY
# ============================================================
echo -e "${BLUE}📁 Creating tests...${NC}"
create_dir "tests"
create_file "tests/__init__.py" "# Tests module"
create_file "tests/test_chunker.py" "#!/usr/bin/env python3\n\"\"\"Tests for chunker module.\"\"\"\n\ndef test_chunker():\n    pass"
create_file "tests/test_dataset.py" "#!/usr/bin/env python3\n\"\"\"Tests for dataset module.\"\"\"\n\ndef test_dataset():\n    pass"
create_file "tests/test_trainer.py" "#!/usr/bin/env python3\n\"\"\"Tests for trainer module.\"\"\"\n\ndef test_trainer():\n    pass"
create_file "tests/test_preprocessing.py" "#!/usr/bin/env python3\n\"\"\"Tests for preprocessing.\"\"\"\n\ndef test_preprocessing():\n    pass"
create_file "tests/test_logging.py" "#!/usr/bin/env python3\n\"\"\"Tests for logging.\"\"\"\n\ndef test_logging():\n    pass"

# ============================================================
# ROOT FILES
# ============================================================
echo -e "${BLUE}📁 Creating root files...${NC}"

# .gitignore
create_file ".gitignore" "# Python\n__pycache__/\n*.pyc\n*.pyo\n*.pyd\n.Python\n*.so\n*.egg\n*.egg-info/\ndist/\nbuild/\n.venv/\nvenv/\nENV/\n\n# Data\n*.hdf5\n*.pt\n*.npy\n*.npz\n\n# Checkpoints\ncheckpoints/*.pt\n\n# Logs\n*.log\n\n# TensorBoard\nruns/\n\n# MLflow\nmlruns/\n\n# IDE\n.vscode/\n.idea/\n*.swp\n*.swo\n\n# OS\n.DS_Store\nThumbs.db"

# requirements.txt
create_file "requirements.txt" "# Core scientific\nnumpy>=1.24.0\nscipy>=1.10.0\nh5py>=3.8.0\n\n# PyTorch\n# For MPS (Mac) use: pip3 install torch torchvision torchaudio\n# For CUDA (NVIDIA) use: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\ntorch>=2.0.0\ntorchvision>=0.15.0\n\n# Visualization\nmatplotlib>=3.7.0\nseaborn>=0.12.0\n\n# ML & Experiment Tracking\nmlflow>=2.3.0\ntensorboard>=2.12.0\nloguru>=0.7.0\n\n# Utilities\ntqdm>=4.65.0\npyyaml>=6.0\nscikit-learn>=1.2.0"

# pyproject.toml
create_file "pyproject.toml" "[build-system]\nrequires = [\"setuptools>=61.0\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"seismic-fbp\"\nversion = \"1.0.0\"\ndescription = \"Seismic First Break Picking with U-Net\"\nauthors = [{name = \"Your Name\", email = \"you@example.com\"}]\nlicense = {text = \"MIT\"}\nreadme = \"README.md\"\nrequires-python = \">=3.10\"\ndependencies = [\n    \"numpy>=1.24.0\",\n    \"h5py>=3.8.0\",\n    \"torch>=2.0.0\",\n    \"matplotlib>=3.7.0\",\n    \"mlflow>=2.3.0\",\n    \"tensorboard>=2.12.0\",\n    \"loguru>=0.7.0\",\n    \"tqdm>=4.65.0\",\n    \"pyyaml>=6.0\",\n    \"scikit-learn>=1.2.0\",\n]"

# setup.py
create_file "setup.py" "from setuptools import setup, find_packages\n\nsetup(\n    name=\"seismic-fbp\",\n    version=\"1.0.0\",\n    description=\"Seismic First Break Picking with U-Net\",\n    author=\"Your Name\",\n    packages=find_packages(where=\"src\"),\n    package_dir={\"\": \"src\"},\n    python_requires=\">=3.10\",\n)"

# README.md
create_file "README.md" "# Seismic First Break Picking with U-Net\n\n## Overview\n\nThis project implements a production-grade pipeline for automatic first break picking on seismic data using a U-Net architecture.\n\n## Features\n\n- Memory-efficient HDF5 data loading with vectorized mask generation\n- Chunked preprocessing for large datasets\n- Training with MLflow + TensorBoard logging\n- Multi-GPU support (DataParallel)\n- Checkpoint resuming with full state\n- Early stopping and gradient clipping\n\n## Installation\n\n```bash\npip install -r requirements.txt\n```\n\n## Usage\n\n### 1. Preprocess the data\n\n```bash\npython scripts/preprocess.py --config configs/halfmile.yaml\n```\n\n### 2. Train the model\n\n```bash\npython scripts/train.py --config configs/halfmile.yaml\n```\n\n### 3. Evaluate the model\n\n```bash\npython scripts/evaluate.py --config configs/halfmile.yaml --model checkpoints/Halfmile/best_model.pt\n```\n\n## Project Structure\n\n```\nseismic_fbp/\n├── data/\n│   ├── raw/      # Original HDF5 files\n│   └── chunks/   # Preprocessed chunks\n├── src/          # Source code\n├── scripts/      # Execution scripts\n├── configs/      # Configuration files\n├── checkpoints/  # Training checkpoints\n├── runs/         # TensorBoard logs\n├── mlruns/       # MLflow logs\n└── logs/         # Log files\n```\n\n## License\n\nMIT"

# .env
create_file ".env" "# Environment variables\n\n# MLflow Tracking URI\nMLFLOW_TRACKING_URI=file:./mlruns\n\n# TensorBoard log directory\nTENSORBOARD_LOG_DIR=./runs\n\n# Logging level\nLOG_LEVEL=INFO\n\n# Device configuration\nDEVICE=mps\n"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ PROJECT STRUCTURE CREATED SUCCESSFULLY!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo -e "${YELLOW}📊 Summary:${NC}"
echo ""

# Count directories
DIR_COUNT=$(find . -type d -not -path "*/\.*" -not -path "." | wc -l | xargs)
FILE_COUNT=$(find . -type f -not -path "*/\.*" | wc -l | xargs)

echo -e "  ${GREEN}Directories created:${NC} $DIR_COUNT"
echo -e "  ${GREEN}Files created:${NC} $FILE_COUNT"
echo ""

echo -e "${YELLOW}📁 Next steps:${NC}"
echo ""
echo -e "  1. ${GREEN}Copy your HDF5 data files to:${NC} data/raw/"
echo -e "     cp /path/to/Halfmile3D_add_geom_sorted.hdf5 data/raw/"
echo ""
echo -e "  2. ${GREEN}Create a Python virtual environment:${NC}"
echo -e "     python3 -m venv .venv"
echo -e "     source .venv/bin/activate"
echo -e "     pip install -r requirements.txt"
echo ""
echo -e "  3. ${GREEN}Run preprocessing:${NC}"
echo -e "     python scripts/preprocess.py --config configs/halfmile.yaml"
echo ""
echo -e "  4. ${GREEN}Run training:${NC}"
echo -e "     python scripts/train.py --config configs/halfmile.yaml"
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Happy coding! 🚀${NC}"
echo -e "${BLUE}============================================================${NC}"