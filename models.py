import torch
import torch.nn as nn
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BaselineCNN, self).__init__()
        # ====================================================
        # Convolutional Feature Extractor
        # ====================================================
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # Block 2
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # Block 3
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        # ====================================================
        # Adaptive Pooling
        # ====================================================
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        # ====================================================
        # Fully Connected Classifier
        # ===================================================
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(128 * 4 * 4,256),
                                        nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(256,num_classes))
#======================================================================================================
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
#======================================================================================================
if __name__ == "__main__":
    model = BaselineCNN(num_classes=4)
    print("\n" + "=" * 60)
    print("BASELINE CNN ARCHITECTURE")
    print("=" * 60)
    print(model)
    # Create Dummy Input
    dummy_input = torch.randn(1,3,224,224)
    # Forward Pass
    output = model(dummy_input)
    print("\n" + "=" * 60)
    print("MODEL TEST")
    print("=" * 60)
    print(f"Input Shape: "f"{dummy_input.shape}")
    print(f"Output Shape: "f"{output.shape}")