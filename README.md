##Quick Start

### 1. Setup Environment + quick test
```bash
./setup.sh
```


### 2. Use Your Custom PyTorch Model

Want to use your own PyTorch model? Easy to integrate!

**Step 1: Add your model to `src/model.py`**
```python
# Add this function to src/model.py
def your_custom_model(model_config):
    """Your custom PyTorch model that returns an nn.Module"""
    import torch.nn as nn
    from your_model_file import YourModel  # Your custom model

    model = YourModel(num_classes=model_config['num_classes'])
    return model
```

**Step 2: Update config to use your model**
```yaml
# configs/training_config.yaml
model:
  name: "your_custom_model"  # Function name from src/model.py
  num_classes: 200          # Your number of classes
  pretrained: false         # Usually false for custom models
```

**That's it!** The system will now:
- Load your model function from `src/model.py`
- Use it for training, optimization, and prediction
- Save/load your model weights automatically

### 3. Configure Training Settings

### 4. Train Your Model

```bash
python main.py --optimize
```
This runs the complete pipeline:
-Automatic hyperparameter tuning
-Final model training with best parameters
-Comprehensive metrics and MLflow tracking

**Quick Testing:**
```bash
python main.py --optimize --config configs/test_config.yaml
```
Runs ultra-fast trials for development/testing.

### 5. Make Predictions
```bash
# Single image
python main.py --predict --model_path final_bird_classifier.pth --image_path test_image.jpg

# Batch predictions
python main.py --predict --model_path final_bird_classifier.pth --image_dir /path/to/images/
```

### 6. Custom Datasets
To use your own dataset:

1. **Organize your data:**
   ```
   your_dataset/
   ├── train/
   │   ├── class1/
   │   │   ├── img1.jpg
   │   │   └── ...
   │   └── class2/
   │       └── ...
   └── test/
       └── ...
   ```

2. **Update config:**
   ```yaml
   # configs/training_config.yaml
   data:
     train_path: "/path/to/your_dataset"
     train_subdir: "train"  # or your folder name
   ```

##Customization Guide for Your Use Case

This template is designed for **any PyTorch image classification project**. Here's what you need to change:

###️ Critical Changes Required

**1. Import Your Model (src/model.py)**
```python
# REPLACE this import with your model
# from your_model_file import YourModel  # Your actual model

def your_custom_model(model_config):
    """Your custom PyTorch model that returns an nn.Module"""
    from your_model_file import YourModel  # Change this import
    model = YourModel(num_classes=model_config['num_classes'])
    return model
```

**2. Update Model Names (configs/training_config.yaml)**
```yaml
model:
  name: "your_custom_model"  # Must match function name in src/model.py
  num_classes: YOUR_CLASS_COUNT  # e.g., 10, 50, 1000 - your number of classes
```

**3. Change Objective Function (src/optimize.py:~line 120)**
```python
# CURRENT: Maximizes F1 score
return trial_value  # Change this to your objective

# EXAMPLES:
# return trial_value              # Minimize loss (lower is better)
# return -trial_value             # Maximize accuracy (higher is better)
# return custom_metric_score      # Your custom metric
```

**4. Check for Hardcoded Imports**
These files may have hardcoded references to "bird" or specific classes:

| File | Search For | Replace With |
|------|-----------|--------------|
| **src/pipeline.py** | "bird_classifier_dev" | "your_project_name" |
| **src/optimize.py** | "bird_classification" | "your_classification" |
| **src/predict.py** | `class_names = [f"class_{i}"` | Save/load actual class names |

**Example - Update Project Names:**
```python
# src/optimize.py ~line 100
study_name = "your_project_optimization"  # Was "bird_classification_optimization"

# src/pipeline.py ~line 42
experiment_name = "your_project"  # Was "bird_classifier_dev_optimization"
```

###File-by-File Customization Guide

| File | What to Change | Why |
|------|----------------|-----|
| **src/model.py** | Add your model function | Core model architecture |
| **configs/training_config.yaml** | `name`, `num_classes`, `image_size` | Model configuration |
| **src/optimize.py** | Objective function line 120 | What Optuna optimizes |
| **configs/search_space.yaml** | Hyperparameter ranges | Your model's parameters |
| **src/train.py** | Loss function (if needed) | Training objective |
| **src/data.py** | Data transforms (if needed) | Input preprocessing |
