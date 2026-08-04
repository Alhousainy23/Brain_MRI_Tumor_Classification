from pathlib import Path
import torch
# ============================================================
# 1. Project Paths
# ============================================================
PROJECT_ROOT = Path(r"D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code")
# ============================================================
# 2. Dataset Directories
# ============================================================
TRAIN_DIR = PROJECT_ROOT / "Training"
TEST_DIR = PROJECT_ROOT / "Testing"
# ============================================================
# 3. Dataset Configuration
# ============================================================
CLASS_NAMES = ["glioma","meningioma","notumor","pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
# ============================================================
# 4. Image Configuration
# ============================================================
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
# ============================================================
# 5. Data Split Configuration
# ============================================================
VAL_SPLIT = 0.20
RANDOM_SEED = 42
# ============================================================
# 6. DataLoader Configuration
# ============================================================
BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
# ============================================================
# 7. Training Configuration
# ============================================================
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
# ============================================================
# 8. Early Stopping Configuration
# ============================================================
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001
# ============================================================
# 9. Device Configuration
# ============================================================
DEVICE = torch.device("cuda"if torch.cuda.is_available()else "cpu")
# ============================================================
# 10. Output Directories
# ============================================================
MODELS_DIR = (PROJECT_ROOT / "models")
OUTPUTS_DIR = (PROJECT_ROOT / "outputs")
ARCHITECTURES_DIR = (OUTPUTS_DIR / "architectures")
PLOTS_DIR = (OUTPUTS_DIR / "plots")
EVALUATION_DIR = (OUTPUTS_DIR / "evaluation")
# ============================================================
# 11. Model Paths
# ============================================================
BASELINE_MODEL_PATH = (MODELS_DIR/ "baseline_cnn_best.pth")
# ============================================================
# ============================================================
# ResNet18 Model Paths
# ============================================================
RESNET18_MODEL_PATH = (MODELS_DIR / "resnet18_best.pth")
# ============================================================
# ResNet18 Training History
# ============================================================
RESNET18_HISTORY_PATH = (OUTPUTS_DIR / "resnet18_history.pth")
#=============================================================
# ============================================================
# EfficientNet-B0 Model Paths
# ============================================================
EFFICIENTNET_B0_MODEL_PATH = (MODELS_DIR / "efficientnet_b0_best.pth")
# ============================================================
# EfficientNet-B0 Training History
# ============================================================
EFFICIENTNET_B0_HISTORY_PATH = (OUTPUTS_DIR / "efficientnet_b0_history.pth")
#=============================================================
# 12. Training History
# ===========================================================
BASELINE_HISTORY_PATH = (OUTPUTS_DIR/ "baseline_cnn_history.pth")
# ============================================================
# Create Required Directories
# ============================================================
MODELS_DIR.mkdir(parents=True,exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True,exist_ok=True)
ARCHITECTURES_DIR.mkdir(parents=True,exist_ok=True)
PLOTS_DIR.mkdir(parents=True,exist_ok=True)
EVALUATION_DIR.mkdir(parents=True,exist_ok=True)