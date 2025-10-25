# ResNet50 MLOps Bird Classifier

A complete PyTorch-based bird species classification system with automated hyperparameter tuning using MLflow and Optuna. Designed for internal company POC with enterprise-grade experiment tracking and optimization.

## 🚀 Quick End-to-End Validation Commands

**Test the complete system right now:**

```bash
# 1. Complete MLOps Pipeline (2 trials, 3 epochs for quick testing)
python main.py --data_dir "/Users/tianqinmeng/.cache/kagglehub/datasets/kedarsai/bird-species-classification-220-categories/versions/1/Train" --optimize --n_trials 2 --final_epochs 3

# 2. Basic Training Only
python main.py --data_dir "/Users/tianqinmeng/.cache/kagglehub/datasets/kedarsai/bird-species-classification-220-categories/versions/1/Train" --basic

# 3. Prediction Test (after any training completes)
python main.py --predict --model_path "best_model.pth" --image_path "/Users/tianqinmeng/.cache/kagglehub/datasets/kedarsai/bird-species-classification-220-categories/versions/1/Train/Acadian_Flycatcher/Acadian_Flycatcher_0003_29094.jpg" --data_dir "/Users/tianqinmeng/.cache/kagglehub/datasets/kedarsai/bird-species-classification-220-categories/versions/1/Train"

# 4. Check MLflow Experiment Tracking
ls -la mlruns/
```

These commands validate: ✅ Data loading ✅ Model training ✅ Hyperparameter optimization ✅ MLflow tracking ✅ Model inference

---

## Features

- ✅ **PyTorch Native**: 100% compatible with existing PyTorch codebases
- ✅ **Automated Hyperparameter Tuning**: Optuna-powered Bayesian optimization
- ✅ **Comprehensive Metrics**: Accuracy, precision, recall, F1, confusion matrix
- ✅ **Configurable Tracking**: User-selectable metrics via YAML configuration
- ✅ **MLflow Integration**: Complete experiment tracking and model registry
- ✅ **Production Ready**: Inference script with batch processing support

## Requirements Satisfaction

This solution addresses every requirement from your company contact:

| Requirement | Implementation |
|-------------|----------------|
| **Dataset flexibility** | Works with any classification dataset, configured for 200 bird species |
| **Framework choice** | MLflow (self-hosted) + Optuna (industry standard) |
| **PyTorch compatibility** | 100% native PyTorch using `torch.nn.Module`, `torchvision.models.resnet50` |
| **Comprehensive metrics** | Tracks accuracy, precision, recall, F1, loss, confusion matrices |
| **Configurable tracking** | YAML-based metric toggles (`track_accuracy: true/false`) |
| **Automated tuning** | Optuna Bayesian optimization with MLflow experiment tracking |

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd resnet50-mlops-bird-classifier

# Install dependencies
pip install -r requirements.txt

# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000
```

### Data Setup

Organize your bird dataset as follows:
```
data_dir/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
│   └── ...
└── ...
```

### One-Command Execution

```bash
# Complete automated pipeline (recommended)
python main.py --data_dir /path/to/bird_dataset --optimize --n_trials 50

# Basic training only
python main.py --data_dir /path/to/bird_dataset --basic

# Hyperparameter optimization only
python main.py --data_dir /path/to/bird_dataset --optimize_only --n_trials 100
```

### Make Predictions

```bash
# Single image prediction
python main.py --predict --model_path final_bird_classifier.pth --image_path test_image.jpg

# Batch prediction
python main.py --predict --model_path final_bird_classifier.pth --image_dir /path/to/test_images
```

## Configuration

### Training Configuration (`configs/config.yaml`)

```yaml
# Model settings
model:
  num_classes: 200
  pretrained: true

# Training parameters
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 1e-4

# Metrics tracking (user configurable)
metrics:
  track_accuracy: true
  track_precision: true
  track_recall: true
  track_f1: true
  track_confusion_matrix: true

# MLflow settings
mlflow:
  experiment_name: "bird_classifier"
  tracking_uri: "http://localhost:5000"
```

### Hyperparameter Search Space (`configs/search_space.yaml`)

Defines the ranges for automated hyperparameter tuning:
- Learning rate: 1e-5 to 1e-1 (log scale)
- Batch size: [16, 32, 64, 128]
- Weight decay: 1e-6 to 1e-3 (log scale)
- Optimizers: Adam, AdamW, SGD
- And more...

## Pipeline Components

### 1. PyTorch Model (`src/model.py`)
```python
class ResNet50BirdClassifier(nn.Module):
    # 100% native PyTorch implementation
    # Uses torchvision.models.resnet50
    # Compatible with existing PyTorch workflows
```

### 2. Automated Optimizer (`src/optimize.py`)
```python
class HyperparameterOptimizer:
    # Optuna Bayesian optimization
    # MLflow experiment tracking
    # PyTorch integration
    # Automated best hyperparameter selection
```

### 3. Complete Pipeline (`src/pipeline.py`)
```python
class MLPipeline:
    # End-to-end MLOps workflow
    # Hyperparameter tuning + final training
    # Comprehensive logging and tracking
```

### 4. Inference Engine (`src/predict.py`)
```python
class BirdPredictor:
    # Production-ready inference
    # Single and batch prediction
    # Confidence thresholding
    # PyTorch model loading
```

## MLOps Integration

### MLflow Tracking
All experiments are automatically tracked in MLflow:
- Hyperparameters and metrics
- Model artifacts and checkpoints
- Confusion matrices and visualizations
- Comparison between different runs

### Hyperparameter Optimization
- **Optuna Study**: Bayesian optimization with TPE sampler
- **Pruning**: Median pruner for efficient trial management
- **Multi-objective**: Optimizes validation F1 score
- **Reproducible**: Fixed random seeds and logging

### Model Management
- **Best Model Selection**: Automatic identification of optimal hyperparameters
- **Model Registry**: MLflow model registry with versioning
- **Artifact Storage**: Model checkpoints and training logs
- **Reproducibility**: Complete experiment reproducibility

## Advanced Usage

### Custom Hyperparameter Ranges
Modify `configs/search_space.yaml` to adjust optimization ranges:

```yaml
learning_rate:
  type: float
  low: 1e-6
  high: 1e-1
  log: true

batch_size:
  type: categorical
  choices: [16, 32, 64, 128, 256]
```

### Custom Metrics
Enable/disable specific metrics in `configs/config.yaml`:

```yaml
metrics:
  track_accuracy: true
  track_precision: false  # Disable if not needed
  track_recall: false    # Disable if not needed
  track_f1: true
  track_confusion_matrix: true
```

### GPU Acceleration
The system automatically detects and uses GPU when available:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Monitoring and Visualization

### MLflow UI
Start the MLflow UI to monitor experiments:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Access at `http://localhost:5000` to view:
- Experiment comparison
- Metric plots over time
- Hyperparameter importance
- Model artifacts and confusion matrices

### Command Line Progress
All operations provide detailed console output:
- Real-time training progress
- Optimization trial results
- Best hyperparameter updates
- Final model performance summary

## Troubleshooting

### Common Issues

1. **MLflow Connection Error**
   ```bash
   # Ensure MLflow server is running
   mlflow server --host 0.0.0.0 --port 5000
   ```

2. **CUDA Out of Memory**
   ```yaml
   # Reduce batch size in config.yaml
   training:
     batch_size: 16  # Reduce from 32
   ```

3. **Data Loading Issues**
   ```bash
   # Verify data structure
   find data_dir -type d | head -10
   ```

## Project Structure

```
resnet50-mlops-bird-classifier/
├── src/
│   ├── model.py           # PyTorch ResNet50 implementation
│   ├── data.py            # Data loading and preprocessing
│   ├── train.py           # Training logic with metrics
│   ├── optimize.py        # Optuna hyperparameter tuning
│   ├── pipeline.py        # Complete MLOps pipeline
│   └── predict.py         # Inference engine
├── configs/
│   ├── config.yaml        # Training configuration
│   └── search_space.yaml  # Hyperparameter search space
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Enterprise Features

- **Internal Deployment**: Self-hosted MLflow, no external dependencies
- **PyTorch Integration**: Seamless integration with existing PyTorch codebases
- **Scalable Architecture**: Efficient batch processing and GPU utilization
- **Comprehensive Logging**: Full experiment tracking and audit trails
- **Model Governance**: Version control and reproducibility

This system provides your company with a complete, production-ready bird classification solution that demonstrates the value of automated hyperparameter tuning while maintaining full compatibility with existing PyTorch workflows.