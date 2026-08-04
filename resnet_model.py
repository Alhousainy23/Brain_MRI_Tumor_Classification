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
from torchvision import models
from config import (NUM_CLASSES,DEVICE)
# ============================================================
# ResNet18 Model
# ============================================================
class ResNet18Model(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES,pretrained=True):
        super().__init__()
        # ====================================================
        # Load Pretrained ResNet18
        # ====================================================
        if pretrained:weights = (models.ResNet18_Weights.DEFAULT)
        else:weights = None
        self.model = models.resnet18(weights=weights)
        # ====================================================
        # Get Number of Features
        # ====================================================
        num_features = (self.model.fc.in_features)
        # ====================================================
        # Replace Final Fully Connected Layer
        # ====================================================
        self.model.fc = nn.Linear(num_features,num_classes)
    # ========================================================
    # Forward Pass
    # ========================================================
    def forward(self,x):return self.model(x)
# ============================================================
# Test ResNet18 Model
# ============================================================
def test_resnet18():
    print("\n" + "=" * 60)
    print("RESNET18 MODEL TEST")
    print("=" * 60)
    # ========================================================
    # Create Model
    # ========================================================
    model = ResNet18Model(num_classes=NUM_CLASSES,pretrained=True)
    # ========================================================
    # Move Model to Device
    # ========================================================
    model = model.to(DEVICE)
    # ========================================================
    # Print Model
    # ========================================================
    print("\nModel:")
    print(model)
    # ========================================================
    # Create Dummy Input
    # ========================================================
    dummy_input = torch.randn(1,3,224,224).to(DEVICE)
    # ========================================================
    # Forward Pass
    # ========================================================
    with torch.no_grad():output = model(dummy_input)
    # ========================================================
    # Print Shapes
    # ========================================================
    print("\nInput Shape:")
    print(dummy_input.shape)
    print("\nOutput Shape:")
    print(output.shape)
    # ========================================================
    # Check Output
    # ========================================================
    assert output.shape == (1,NUM_CLASSES)
    print("\n✓ ResNet18 Model Test Passed!")
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":test_resnet18()