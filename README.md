## Quick Start

### 1. Setup Environment + quick test
```bash
./setup.sh
```

### 2. Train Your Model
```bash
python main.py --optimize
```
This runs the complete pipeline:
- Automatic hyperparameter tuning
- Final model training with best parameters
- Comprehensive metrics and MLflow tracking

### 3. Make Predictions
```bash
# Single image
python main.py --predict --model_path final_bird_classifier.pth --image_path test_image.jpg

# Batch predictions
python main.py --predict --model_path final_bird_classifier.pth --image_dir /path/to/images/
```

### 4. Use Custom Datasets
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

### 5. Use Custom PyTorch Model

Contact us for integration assistance
### 6. Use Custom Objective Function

Change Objective Function (src/optimize.py:~line 135)