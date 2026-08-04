import sys
from pathlib import Path
# ============================================================
# Add src Directory to Python Path
# ============================================================
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:sys.path.insert(0, str(SRC_DIR))
# ============================================================
# Imports
# ============================================================
import torch
import matplotlib.pyplot as plt
from config import (RESNET18_HISTORY_PATH,PLOTS_DIR)
# ============================================================
# Plot ResNet18 Training Results
# ============================================================
def plot_resnet18_results():
    print("\n" + "=" * 60)
    print("RESNET18 TRAINING VISUALIZATION")
    print("=" * 60)
    # ========================================================
    # Load Training History
    # ========================================================
    history = torch.load(RESNET18_HISTORY_PATH,map_location="cpu")
    # ========================================================
    # Extract History
    # ========================================================
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_accuracy = history["train_accuracy"]
    val_accuracy = history["val_accuracy"]
    # ========================================================
    # Create Epoch Range
    # ========================================================
    epochs = range(1,len(train_loss) + 1)
    # ========================================================
    # Create Output Directory
    # ========================================================
    PLOTS_DIR.mkdir(parents=True,exist_ok=True)
    # ========================================================
    # 1. Loss Curve
    # ========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(epochs,train_loss,label="Training Loss")
    plt.plot(epochs,val_loss,label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet18 Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    loss_path = (PLOTS_DIR/ "resnet18_loss_curve.png")
    plt.savefig(loss_path,dpi=300,bbox_inches="tight")
    plt.show()
    print(f"\nLoss Curve Saved At:\n"f"{loss_path}")
    # ========================================================
    # 2. Accuracy Curve
    # ========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(epochs,train_accuracy,label="Training Accuracy")
    plt.plot(epochs,val_accuracy,label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ResNet18 Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    accuracy_path = (PLOTS_DIR/ "resnet18_accuracy_curve.png")
    plt.savefig(accuracy_path,dpi=300,bbox_inches="tight")
    plt.show()
    print(f"\nAccuracy Curve Saved At:\n"f"{accuracy_path}")
    # ========================================================
    # Best Validation Accuracy
    # ========================================================
    best_val_accuracy = max(val_accuracy)
    best_epoch = (val_accuracy.index(best_val_accuracy) + 1)
    print("\n" + "=" * 60)
    print("BEST RESNET18 RESULTS")
    print("=" * 60)
    print(f"\nBest Epoch: "f"{best_epoch}")
    print(f"Best Validation Accuracy: "f"{best_val_accuracy:.2f}%")
    print(f"Best Validation Loss: "f"{val_loss[best_epoch - 1]:.4f}")
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":plot_resnet18_results()