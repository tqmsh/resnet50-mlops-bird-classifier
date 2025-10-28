# Bird Classifier Setup Script (PowerShell Version)

param(
    [string]$VenvName = "bird_classifier_env",
    [int]$MLflowPort = 5001
)

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

function Write-Status {
    param([string]$Message)
    Write-Host $Message -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host $Message -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host $Message -ForegroundColor $Colors.Red
}

function Write-Header {
    param([string]$Message)
    Write-Host $Message -ForegroundColor $Colors.Blue
}

Write-Header "========================================"
Write-Header "  Bird Classifier Setup (PowerShell)"
Write-Header "========================================"

# Check Python
try {
    $python = Get-Command python3 -ErrorAction Stop
    Write-Status "Python 3 found"
} catch {
    try {
        $python = Get-Command python -ErrorAction Stop
        Write-Status "Python found"
    } catch {
        Write-Error "Python 3 is required"
        exit 1
    }
}

# Setup virtual environment
if (-not (Test-Path $VenvName)) {
    Write-Status "Creating virtual environment..."
    & $python -m venv $VenvName
}

Write-Status "Activating virtual environment..."
if (Test-Path ".\$VenvName\Scripts\Activate.ps1") {
    . ".\$VenvName\Scripts\Activate.ps1"
} else {
    Write-Warning "Virtual environment activation script not found, using global Python"
}

# Install dependencies
Write-Status "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download dataset
Write-Status "Downloading dataset..."
$datasetPath = python -c "
import kagglehub
dataset_path = kagglehub.dataset_download('kedarsai/bird-species-classification-220-categories')
print(dataset_path)
"
Write-Status "Dataset: $datasetPath"

# Setup environment
$projectRoot = Get-Location
$userHome = $env:USERPROFILE

$envFile = @"
PROJECT_ROOT=$projectRoot
USER_HOME=$userHome
KAGGLE_CACHE_PATH=$userHome\.cache\kagglehub
DATASET_PATH=$datasetPath
MLFLOW_PORT=$MLflowPort
"@

$envFile | Out-File -FilePath ".env" -Encoding UTF8
Write-Status "Environment configured"

# Setup MLflow
Write-Status "Setting up MLflow..."
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "mlruns" | Out-Null

# Clean old MLflow experiments
Write-Status "Cleaning old MLflow experiments..."
Remove-Item -Path "mlruns\*" -Recurse -Force -ErrorAction SilentlyContinue

# Check and kill existing MLflow process
try {
    $mlflowProcesses = Get-Process -Name "mlflow" -ErrorAction SilentlyContinue
    $mlflowProcess = $mlflowProcesses | Where-Object {
        try { $_.CommandLine -like "*--port $MLflowPort*" } catch { $false }
    }
    if ($mlflowProcess) {
        Write-Warning "Port $MLflowPort is in use, killing existing process..."
        $mlflowProcess | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
} catch {
    # Fallback: try to kill any mlflow process if port detection fails
    try {
        Stop-Process -Name "mlflow" -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    } catch {
        # No mlflow process found, continue
    }
}

# Start MLflow UI
Write-Status "Starting MLflow UI on port $MLflowPort..."
try {
    Start-Process -FilePath "mlflow" -ArgumentList "ui", "--host", "0.0.0.0", "--port", "$MLflowPort" -WindowStyle Hidden
    Write-Status "MLflow UI: http://localhost:$MLflowPort"
} catch {
    Write-Warning "Failed to start MLflow UI, please start manually: mlflow ui --host 0.0.0.0 --port $MLflowPort"
}

# Test dataset verification
Write-Status "Running dataset verification..."
python -c @"
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root=os.path.join(r'$datasetPath', 'Train'),
    transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f'Dataset verification successful!')
print(f'Classes: {len(dataset.classes)}')
print(f'Samples: {len(dataset)}')
"@

# Test pipeline
Write-Status "Testing pipeline..."
python main.py --config configs/test_config.yaml --search_space configs/test_search_space.yaml --optimize

# Test prediction
if (Test-Path "final_bird_classifier.pth") {
    Write-Status "Testing prediction..."
    $testImage = Get-ChildItem -Path $datasetPath -Filter "*.jpg" -Recurse | Select-Object -First 1
    python main.py --predict --model_path final_bird_classifier.pth --image_path $testImage.FullName --config configs/test_config.yaml --top_k 3
}

Write-Status "Setup complete!"
Write-Header "MLflow UI: http://localhost:$MLflowPort"