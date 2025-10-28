#!/bin/bash
set -e

# Colors for output
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors)" -ge 8 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

VENV_NAME="bird_classifier_env"
MLFLOW_PORT="5001"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Bird Classifier Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 required${NC}"
    exit 1
fi

# Setup virtual environment
if [ ! -d "$VENV_NAME" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_NAME"
fi
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_NAME/bin/activate"

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Download dataset
echo -e "${GREEN}Downloading dataset...${NC}"
DATASET_PATH=$(python -c "
import kagglehub
dataset_path = kagglehub.dataset_download('kedarsai/bird-species-classification-220-categories')
print(dataset_path)
")
echo -e "${GREEN}Dataset: $DATASET_PATH${NC}"

# Setup environment
PROJECT_ROOT=$(pwd)
USER_HOME="$HOME"
cat > .env << EOF
PROJECT_ROOT=$PROJECT_ROOT
USER_HOME=$USER_HOME
KAGGLE_CACHE_PATH=$USER_HOME/.cache/kagglehub
DATASET_PATH=$DATASET_PATH
MLFLOW_PORT=$MLFLOW_PORT
EOF

# Setup MLflow
echo -e "${GREEN}Setting up MLflow...${NC}"
mkdir -p logs mlruns

# Clean old MLflow experiments
echo -e "${GREEN}Cleaning old MLflow experiments...${NC}"
rm -rf mlruns/*

# Check and kill existing MLflow process
if pgrep -f "mlflow ui.*--port $MLFLOW_PORT" > /dev/null; then
    echo -e "${YELLOW}Port $MLFLOW_PORT is in use, killing existing process...${NC}"
    pkill -f "mlflow ui.*--port $MLFLOW_PORT"
    sleep 2
fi

# Start MLflow UI
echo -e "${GREEN}Starting MLflow UI on port $MLFLOW_PORT...${NC}"
nohup mlflow ui --host 0.0.0.0 --port $MLFLOW_PORT > mlflow.log 2>&1 &
echo -e "${GREEN}MLflow UI: http://localhost:$MLFLOW_PORT${NC}"

# Test dataset verification
echo -e "${GREEN}Running dataset verification...${NC}"
python -c "
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root=os.path.join('$DATASET_PATH', 'Train'),
    transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f'Dataset verification successful!')
print(f'Classes: {len(dataset.classes)}')
print(f'Samples: {len(dataset)}')
"

# Test pipeline
echo -e "${GREEN}Testing pipeline...${NC}"
python main.py --config configs/test_config.yaml --search_space configs/test_search_space.yaml --optimize

# Test prediction
if [ -f "final_bird_classifier.pth" ]; then
    echo -e "${GREEN}Testing prediction...${NC}"
    TEST_IMAGE=$(find "$DATASET_PATH" -name "*.jpg" | head -1)
    python main.py --predict --model_path final_bird_classifier.pth --image_path "$TEST_IMAGE" --config configs/test_config.yaml --top_k 3
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${BLUE}MLflow UI: http://localhost:$MLFLOW_PORT${NC}"