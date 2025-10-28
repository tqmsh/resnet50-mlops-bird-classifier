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

### 7. Quick Test Images
<img width="411" height="182" alt="Screenshot 2025-10-27 at 11 43 04 PM" src="https://github.com/user-attachments/assets/83bea1bb-3e40-41be-b404-294fce64e0ac" />
<img width="1414" height="287" alt="Screenshot 2025-10-27 at 11 43 14 PM" src="https://github.com/user-attachments/assets/1a8a9f71-cc8d-4879-aa31-68290d520e9e" />
<img width="1183" height="778" alt="Screenshot 2025-10-27 at 11 43 32 PM" src="https://github.com/user-attachments/assets/4f7a51ef-13cc-4b33-9f15-677266f9b609" />
<img width="1908" height="964" alt="Screenshot 2025-10-27 at 11 43 43 PM" src="https://github.com/user-attachments/assets/e47959ff-a5f5-490e-9cea-54e1fdb0ab79" />
<img width="1185" height="595" alt="Screenshot 2025-10-27 at 11 44 01 PM" src="https://github.com/user-attachments/assets/8f01b073-b323-4f2e-a72a-a2eac3029e0d" />
<img width="1279" height="372" alt="Screenshot 2025-10-27 at 11 44 09 PM" src="https://github.com/user-attachments/assets/f976ebe7-024b-4586-becf-b89fbe60dc40" />
<img width="1070" height="184" alt="Screenshot 2025-10-27 at 11 44 24 PM" src="https://github.com/user-attachments/assets/08f9c4de-6efb-4cb9-a8bd-e2f14f8fac78" />
<img width="1198" height="785" alt="Screenshot 2025-10-27 at 11 44 32 PM" src="https://github.com/user-attachments/assets/429607da-6631-4744-97c4-578a764d4472" />
<img width="1907" height="965" alt="Screenshot 2025-10-27 at 11 44 43 PM" src="https://github.com/user-attachments/assets/04f1c0e7-114a-44bd-83cc-e263f823bffb" />
<img width="1497" height="151" alt="Screenshot 2025-10-27 at 11 44 50 PM" src="https://github.com/user-attachments/assets/6dbfa07d-6598-4b0c-a339-0cde0d292271" />
