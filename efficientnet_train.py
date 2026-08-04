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
import torch.optim as optim
from efficientnet_model import EfficientNetB0Model
from dataset import create_dataloaders
from config import (DEVICE,NUM_CLASSES,NUM_EPOCHS,LEARNING_RATE,WEIGHT_DECAY,
                    EARLY_STOPPING_PATIENCE,EARLY_STOPPING_MIN_DELTA,EFFICIENTNET_B0_MODEL_PATH,EFFICIENTNET_B0_HISTORY_PATH)
# ===========================================================
# Early Stopping
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
        # Check Improvement
        # ----------------------------------------------------
        if (self.best_loss- validation_loss) > self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            print(f"EarlyStopping Counter: "f"{self.counter}/{self.patience}")
            if (self.counter>= self.patience):self.early_stop = True
            return False
# ============================================================
# Train One Epoch
# ============================================================
def train_one_epoch(model,dataloader,criterion,optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        # --------------------------------------------------
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
        # Backward Pass
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
        _, predictions = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predictions== labels).sum().item()
    epoch_loss = (running_loss/ total)
    epoch_accuracy = (correct/ total) * 100
    return (epoch_loss,epoch_accuracy)
# ============================================================
# Validate One Epoch
# ============================================================
def validate_one_epoch(model,dataloader,criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # ------------------------------------------------
            # Forward Pass
            # ------------------------------------------------
            outputs = model(images)
            # ------------------------------------------------
            # Calculate Loss
            # ------------------------------------------------
            loss = criterion(outputs,labels)
            running_loss += (loss.item()* images.size(0))
            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------
            _, predictions = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predictions== labels).sum().item()
    epoch_loss = (running_loss/ total)
    epoch_accuracy = (correct/ total) * 100
    return (epoch_loss,epoch_accuracy)
# ============================================================
# Train EfficientNet-B0
# ============================================================
def train_efficientnet_b0():
    print("\n" + "=" * 60)
    print("EFFICIENTNET-B0 TRAINING")
    print("=" * 60)
    # ========================================================
    # Device Information
    # ========================================================
    print(f"\nTraining Device: "f"{DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Name: "f"{torch.cuda.get_device_name(0)}")
    # ========================================================
    # Create DataLoaders
    # ========================================================
    (train_dataset,val_dataset,test_dataset,train_loader,val_loader,test_loader) = create_dataloaders()
    print(f"\nTraining Samples: "f"{len(train_dataset)}")
    print(f"Validation Samples: "f"{len(val_dataset)}")
    print(f"Testing Samples: "f"{len(test_dataset)}")
    # ========================================================
    # Create EfficientNet-B0 Model
    # ========================================================
    model = EfficientNetB0Model(num_classes=NUM_CLASSES,pretrained=True)
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
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    # ========================================================
    # Early Stopping
    # ========================================================
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,min_delta=EARLY_STOPPING_MIN_DELTA)
    # ========================================================
    # Training History
    # ========================================================
    history = {"train_loss": [],"train_accuracy": [],"val_loss": [],"val_accuracy": []}
    # ========================================================
    # Training Loop
    # ========================================================
    for epoch in range(NUM_EPOCHS):
        print("\n" + "-" * 60)
        print(f"Epoch "f"{epoch + 1}/"f"{NUM_EPOCHS}")
        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        train_loss, train_accuracy = (train_one_epoch(model,train_loader,criterion,optimizer))
        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------
        val_loss, val_accuracy = (validate_one_epoch(model,val_loader,criterion))
        # ----------------------------------------------------
        # Save History
        # ----------------------------------------------------
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        # ----------------------------------------------------
        # Print Results
        # ----------------------------------------------------
        print(f"Train Loss: "f"{train_loss:.4f}")
        print(f"Train Accuracy: "f"{train_accuracy:.2f}%")
        print(f"Validation Loss: "f"{val_loss:.4f}")
        print(f"Validation Accuracy: "f"{val_accuracy:.2f}%")
        # ----------------------------------------------------
        # Early Stopping Check
        # ----------------------------------------------------
        is_best = early_stopping(val_loss)
        # ----------------------------------------------------
        # Save Best Model
        # ----------------------------------------------------
        if is_best:
            torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                        "epoch":epoch + 1,"val_loss":val_loss,"val_accuracy":val_accuracy},EFFICIENTNET_B0_MODEL_PATH)
            print("\n✓ Best EfficientNet-B0 Model Saved!")
        # ----------------------------------------------------
        # Early Stop
        # ----------------------------------------------------
        if early_stopping.early_stop:
            print("\n" + "=" * 60)
            print("EARLY STOPPING TRIGGERED")
            print(f"Training stopped at "f"epoch {epoch + 1}")
            break
    # ========================================================
    # Save Training History
    # ========================================================
    torch.save(history,EFFICIENTNET_B0_HISTORY_PATH)
    # ========================================================
    # Training Completed
    # ========================================================
    print("\n" + "=" * 60)
    print("EFFICIENTNET-B0 TRAINING COMPLETED")
    print("=" * 60)
    print("\nBest Model Saved At:")
    print(EFFICIENTNET_B0_MODEL_PATH)
    print("\nTraining History Saved At:")
    print(EFFICIENTNET_B0_HISTORY_PATH)
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":train_efficientnet_b0()