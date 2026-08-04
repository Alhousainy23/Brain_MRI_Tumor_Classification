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
from torch.optim import Adam
from models import BaselineCNN
from dataset import create_dataloaders
from config import (DEVICE,NUM_EPOCHS,LEARNING_RATE,WEIGHT_DECAY,
                    EARLY_STOPPING_PATIENCE,EARLY_STOPPING_MIN_DELTA,BASELINE_MODEL_PATH,BASELINE_HISTORY_PATH)
# ============================================================
# Early Stopping Class
# ============================================================
class EarlyStopping:
    def __init__(self,patience=5,min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
    def __call__(self,validation_loss):
        # ----------------------------------------------------
        # First Validation Loss
        # ----------------------------------------------------
        if self.best_loss == float("inf"):
            self.best_loss = validation_loss
            return
        # ----------------------------------------------------
        # Check Improvement
        # ----------------------------------------------------
        improvement = (self.best_loss- validation_loss)
        if improvement > self.min_delta:
            # Validation Loss Improved
            self.best_loss = validation_loss
            self.counter = 0
        else:# No Significant Improvement
            self.counter += 1
            print(f"EarlyStopping Counter: "f"{self.counter}/{self.patience}")
            if self.counter >= self.patience:self.early_stop = True
# ============================================================
# Training Function
# ============================================================
def train_one_epoch(model,dataloader,criterion,optimizer,device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        # ----------------------------------------------------
        # Move Data to GPU / CPU
        # ----------------------------------------------------
        images = images.to(device)
        labels = labels.to(device)
        # ----------------------------------------------------
        # Clear Gradients
        # ----------------------------------------------------
        optimizer.zero_grad()
        # ----------------------------------------------------
        # Forward Pass
        # ----------------------------------------------------
        outputs = model(images)
        # ----------------------------------------------------
        # Calculate Loss
        # ----------------------------------------------------
        loss = criterion(outputs,labels)
        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------
        loss.backward()
        # ----------------------------------------------------
        # Update Weights
        # ----------------------------------------------------
        optimizer.step()
        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------
        running_loss += (loss.item()* images.size(0))
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = (running_loss/ total)
    epoch_accuracy = (correct/ total) * 100
    return (epoch_loss,epoch_accuracy)
# ============================================================
# Validation Function
# ============================================================
def validate(model,dataloader,criterion,device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            # ------------------------------------------------
            # Move Data to Device
            # ------------------------------------------------
            images = images.to(device)
            labels = labels.to(device)
            # ------------------------------------------------
            # Forward Pass
            # ------------------------------------------------
            outputs = model(images)
            # ------------------------------------------------
            # Calculate Loss
            # ------------------------------------------------
            loss = criterion(outputs,labels)
            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------
            running_loss += (loss.item()* images.size(0))
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = (running_loss/ total)
    epoch_accuracy = (correct/ total) * 100
    return (epoch_loss,epoch_accuracy)
# ============================================================
# Main Training Function
# ============================================================
def train_baseline_cnn():
    print("\n" + "=" * 60)
    print("BASELINE CNN TRAINING")
    print("=" * 60)
    # ========================================================
    # Device Information
    # ========================================================
    print(f"\nTraining Device: "f"{DEVICE}")
    if DEVICE.type == "cuda":print(f"GPU Name: "f"{torch.cuda.get_device_name(0)}")
    # ========================================================
    # Create DataLoaders
    # ========================================================
    (train_dataset,val_dataset,test_dataset,train_loader,val_loader,test_loader) = create_dataloaders()
    print(f"\nTraining Samples: "f"{len(train_dataset)}")
    print(f"Validation Samples: "f"{len(val_dataset)}")
    print(f"Testing Samples: "f"{len(test_dataset)}")
    # ========================================================
    # Create Model
    # ========================================================
    model = BaselineCNN(num_classes=4)
    model = model.to(DEVICE)
    print("\nModel:")
    print(model)
    # ========================================================
    # Loss Function
    # ========================================================
    criterion = nn.CrossEntropyLoss()
    # ========================================================
    # Optimizer
    # ========================================================
    optimizer = Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    # ========================================================
    # Early Stopping
    # ========================================================
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,min_delta=EARLY_STOPPING_MIN_DELTA)
    # ========================================================
    # Create Model Directory
    # ========================================================
    BASELINE_MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    # ========================================================
    # Training History
    # ========================================================
    history = {"train_loss": [],"val_loss": [],"train_accuracy": [],"val_accuracy": []}
    # ========================================================
    # Best Validation Loss
    # ========================================================
    best_val_loss = float("inf")
    # ========================================================
    # Training Loop
    # ========================================================
    for epoch in range(NUM_EPOCHS):
        print("\n" + "-" * 60)
        print(f"Epoch " f"{epoch + 1}" f"/" f"{NUM_EPOCHS}")
        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        train_loss, train_accuracy = train_one_epoch(model,train_loader,criterion,optimizer,DEVICE)
        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------
        val_loss, val_accuracy = validate(model,val_loader,criterion,DEVICE)
        # ----------------------------------------------------
        # Save History
        # ----------------------------------------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        # ---------------------------------------------------
        # Print Results
        # ----------------------------------------------------
        print(f"Train Loss: " f"{train_loss:.4f}")
        print(f"Train Accuracy: " f"{train_accuracy:.2f}%")
        print(f"Validation Loss: " f"{val_loss:.4f}")
        print(f"Validation Accuracy: " f"{val_accuracy:.2f}%")
        # ----------------------------------------------------
        # Save Best Model
        # ----------------------------------------------------
        if val_loss < best_val_loss:best_val_loss = val_loss
        torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"epoch":epoch + 1,
                    "val_loss":val_loss,"val_accuracy":val_accuracy},BASELINE_MODEL_PATH)
        print("\n✓ Best Model Saved!")
        # ----------------------------------------------------
        # Early Stopping Check
        # ----------------------------------------------------
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\n" + "=" * 60)
            print("EARLY STOPPING TRIGGERED")
            print("=" * 60)
            print(f"Training stopped at "f"epoch {epoch + 1}")
            break
    # ========================================================
    # Save Training History
    # ========================================================
    BASELINE_HISTORY_PATH.parent.mkdir(parents=True,exist_ok=True)
    torch.save(history,BASELINE_HISTORY_PATH)
    # ========================================================
    # Training Complete
    # ========================================================
    print("\n" + "=" * 60)
    print("BASELINE CNN TRAINING COMPLETED")
    print("=" * 60)
    print(f"\nBest Model Saved At:")
    print(BASELINE_MODEL_PATH)
    print(f"\nTraining History Saved At:")
    print(BASELINE_HISTORY_PATH)
    return (model,history)
# ============================================================
# Run Training
# ============================================================
if __name__ == "__main__":train_baseline_cnn()