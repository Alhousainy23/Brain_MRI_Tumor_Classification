from pathlib import Path
import torch
from torchviz import make_dot
from models import BaselineCNN
# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = (PROJECT_ROOT/ "outputs"/ "architectures")
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
# ============================================================
# Create Baseline CNN
# ============================================================
model = BaselineCNN(num_classes=4)
model.eval()
# ============================================================
# Create Dummy Input
# ============================================================
dummy_input = torch.randn(1,3,224,224)
# ============================================================
# Forward Pass
# ============================================================
output = model(dummy_input)
# ============================================================
# Generate Architecture Graph
# ============================================================
dot = make_dot(output,params=dict(model.named_parameters()))
# ============================================================
# Save Architecture
# ============================================================
save_path = (OUTPUT_DIR/ "baseline_cnn_architecture")
dot.render(str(save_path),format="png",cleanup=True)
print("\n" + "=" * 60)
print("ARCHITECTURE VISUALIZATION")
print("=" * 60)
print("Baseline CNN architecture ""saved successfully.")
print(f"Saved at:\n"f"{save_path}.png")