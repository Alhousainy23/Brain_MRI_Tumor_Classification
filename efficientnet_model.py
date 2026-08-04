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
# EfficientNet-B0 Model
# ============================================================
class EfficientNetB0Model(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES,pretrained=True):
        super().__init__()
        # ====================================================
        # Load Pretrained EfficientNet-B0
        # ====================================================
        if pretrained:weights = (models.EfficientNet_B0_Weights.DEFAULT)
        else:weights = None
        self.model = models.efficientnet_b0(weights=weights)
        # ====================================================
        # Get Number of Input Features
        # ====================================================
        num_features = (self.model.classifier[1].in_features)
        # ====================================================
        # Replace Final Classifier
        # ====================================================
        self.model.classifier[1] = nn.Linear(num_features,num_classes)
    # ========================================================
    # Forward Pass
    # ========================================================
    def forward(self,x):
        return self.model(x)
# ============================================================
# Test EfficientNet-B0
# ============================================================
def test_efficientnet_b0():
    print("\n" + "=" * 60)
    print("EFFICIENTNET-B0 MODEL TEST")
    print("=" * 60)
    # ========================================================
    # Create Model
    # ========================================================
    model = EfficientNetB0Model(num_classes=NUM_CLASSES,pretrained=True)
    # ========================================================
    # Move to Device
    # ========================================================
    model = model.to(DEVICE)
    # ========================================================
    # Print Model
    # ========================================================
    print("\nModel:")
    print(model)
    # ========================================================
    # Dummy Input
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
    # Verify Output
    # ========================================================
    assert output.shape == (1,NUM_CLASSES)
    print("\n✓ EfficientNet-B0 Model Test Passed!")
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":test_efficientnet_b0()