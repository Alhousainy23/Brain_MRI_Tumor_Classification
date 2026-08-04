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
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,ConfusionMatrixDisplay)
from models import BaselineCNN
from dataset import create_dataloaders
from config import (DEVICE,CLASS_NAMES,BASELINE_MODEL_PATH,EVALUATION_DIR)
# ============================================================
# Evaluate Baseline CNN
# ============================================================
def evaluate_baseline_cnn():
    print("\n" + "=" * 60)
    print("BASELINE CNN EVALUATION")
    print("=" * 60)
    # ========================================================
    # Create DataLoaders
    # ========================================================
    (train_dataset,val_dataset,test_dataset,train_loader,val_loader,test_loader) = create_dataloaders()
    print(f"\nTest Samples: "f"{len(test_dataset)}")
    # ========================================================
    # Create Model
    # ========================================================
    model = BaselineCNN(num_classes=len(CLASS_NAMES))
    # ========================================================
    # Load Best Model
    # ========================================================
    checkpoint = torch.load(BASELINE_MODEL_PATH,map_location=DEVICE)
    # ========================================================
    # Load Model Weights
    # ========================================================
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    print("\nBest Model Loaded Successfully.")
    print(f"Best Epoch: "f"{checkpoint['epoch']}")
    print(f"Validation Loss: "f"{checkpoint['val_loss']:.4f}")
    print(f"Validation Accuracy: "f"{checkpoint['val_accuracy']:.2f}%")
    # ========================================================
    # Evaluation
    # ========================================================
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Forward Pass
            outputs = model(images)
            # Get Predictions
            _, predictions = torch.max(outputs,1)
            # Store Results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    # ========================================================
    # Convert to NumPy
    # ========================================================
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    # ========================================================
    # Calculate Metrics
    # ========================================================
    accuracy = accuracy_score(all_labels,all_predictions)
    precision = precision_score(all_labels,all_predictions,average="weighted",zero_division=0)
    recall = recall_score(all_labels,all_predictions,average="weighted",zero_division=0)
    f1 = f1_score(all_labels,all_predictions,average="weighted",zero_division=0)
    # ========================================================
    # Print Metrics
    # ========================================================
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"\nAccuracy  : "f"{accuracy * 100:.2f}%")
    print(f"Precision : "f"{precision * 100:.2f}%")
    print(f"Recall    : "f"{recall * 100:.2f}%")
    print(f"F1-Score  : "f"{f1 * 100:.2f}%")
    # ========================================================
    # Classification Report
    # ========================================================
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(all_labels,all_predictions,target_names=CLASS_NAMES,zero_division=0)
    print(report)
    # ========================================================
    # Confusion Matrix
    # ========================================================
    cm = confusion_matrix(all_labels,all_predictions)
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(cm)
    # ========================================================
    # Create Evaluation Directory
    # ========================================================
    EVALUATION_DIR.mkdir(parents=True,exist_ok=True)
    # ========================================================
    # Plot Confusion Matrix
    # ========================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=CLASS_NAMES)
    display.plot(ax=ax,cmap="Blues",xticks_rotation=45)
    plt.title("Baseline CNN - Confusion Matrix")
    plt.tight_layout()
    confusion_matrix_path = (EVALUATION_DIR/ "baseline_cnn_confusion_matrix.png")
    plt.savefig(confusion_matrix_path,dpi=300,bbox_inches="tight")
    plt.show()
    print(f"\nConfusion Matrix Saved At:\n"f"{confusion_matrix_path}")
    # ========================================================
    # Return Results
    # ========================================================
    results = {"accuracy": accuracy,"precision": precision,"recall": recall,"f1_score": f1,"confusion_matrix": cm}
    return results
# ============================================================
# Run Evaluation
# ============================================================
if __name__ == "__main__":evaluate_baseline_cnn()