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
from torchviz import make_dot
from efficientnet_model import EfficientNetB0Model
from config import (DEVICE,ARCHITECTURES_DIR)
# ============================================================
# EfficientNet-B0 Architecture Visualization
# ============================================================
def visualize_efficientnet_b0():
    print("\n" + "=" * 60)
    print("EFFICIENTNET-B0 ARCHITECTURE VISUALIZATION")
    print("=" * 60)
    # ========================================================
    # Create Model
    # ========================================================
    model = EfficientNetB0Model(pretrained=True)
    model = model.to(DEVICE)
    model.eval()
    # ========================================================
    # Dummy Input
    # ========================================================
    dummy_input = torch.randn(1,3,224,224).to(DEVICE)
    # ========================================================
    # Forward Pass
    # ========================================================
    output = model(dummy_input)
    # ========================================================
    # Create Graph
    # ========================================================
    graph = make_dot(output,params=dict(model.named_parameters()))
    # ========================================================
    # Save Architecture
    # ========================================================
    output_path = (ARCHITECTURES_DIR/ "efficientnet_b0_architecture")
    graph.render(str(output_path),format="png",cleanup=True)
    print("\nEfficientNet-B0 architecture saved successfully.")
    print("Saved at:")
    print(f"{output_path}.png")
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":visualize_efficientnet_b0()