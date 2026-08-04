import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
try:
    # When running from main.py
    from src.config import (
        TRAIN_DIR,
        TEST_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        VAL_SPLIT,
        RANDOM_SEED
    )

except ModuleNotFoundError:

    # When running dataset.py directly
    from config import (
        TRAIN_DIR,
        TEST_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        VAL_SPLIT,
        RANDOM_SEED
    )
# ============================================================
# 1. Set Random Seed
# ============================================================
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
# ============================================================
# 2. Training Transformations
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # Data Augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0,translate=(0.05, 0.05),scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
# ============================================================
# 3. Validation / Testing Transformations
# ============================================================
val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
# ============================================================
# 4. Create Datasets
# ============================================================
def create_datasets():
    # ========================================================
    # Create two separate ImageFolder datasets
    # ========================================================
    train_full_dataset = datasets.ImageFolder(root=TRAIN_DIR,transform=train_transform)
    val_full_dataset = datasets.ImageFolder(root=TRAIN_DIR,transform=val_test_transform)
    # ========================================================
    # Test Dataset
    # ========================================================
    test_dataset = datasets.ImageFolder(root=TEST_DIR,transform=val_test_transform)
    # ========================================================
    # Calculate Split Sizes
    # ========================================================
    total_size = len(train_full_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size
    # ========================================================
    # Create Reproducible Indices
    # ========================================================
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    indices = torch.randperm(total_size,generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    # ========================================================
    # Create Subsets
    # ========================================================
    train_dataset = torch.utils.data.Subset(train_full_dataset,train_indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset,val_indices)
    return (train_dataset,val_dataset,test_dataset)
# ============================================================
# 5. Create DataLoaders
# ============================================================
def create_dataloaders():
    (train_dataset,val_dataset,test_dataset) = create_datasets()
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
    return (train_dataset,val_dataset,test_dataset,train_loader,val_loader,test_loader)
# ============================================================
# 6. Dataset Information
# ============================================================
def print_dataset_information(train_dataset,val_dataset,test_dataset,class_names):
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")
    print("\nClasses:")
    for index, class_name in enumerate(class_names):print(f"{index} -> {class_name}")
# ============================================================
# 7. Count Images per Class
# ============================================================
def count_images_per_class(dataset,class_names,dataset_name="Dataset"):
    counts = {class_name: 0 for class_name in class_names}
    # ImageFolder Dataset
    if hasattr(dataset, "targets"):targets = dataset.targets
    # For random_split Subset
    else:targets = [dataset.dataset.targets[i] for i in dataset.indices]
    for target in targets:
        class_name = class_names[target]
        counts[class_name] += 1
    print(f"\n{dataset_name} Class Distribution:")
    print("-" * 40)
    for class_name, count in counts.items(): print(f"{class_name:<15}: {count}")
    return counts
# ============================================================
# 8. Plot Class Distribution
# ============================================================
def plot_class_distribution(counts,dataset_name):
    plt.figure(figsize=(8, 5))
    plt.bar(counts.keys(),counts.values())
    plt.title(f"{dataset_name} Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()
# ============================================================
# 9. Display Sample Images
# ============================================================
def show_sample_images(dataset,class_names,num_images=8):
    # Get indices
    indices = random.sample(range(len(dataset)),min(num_images,len(dataset)))
    plt.figure(figsize=(16, 8))
    for i, index in enumerate(indices):
        image, label = dataset[index]
        # Undo normalization
        image = image.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std) + mean
        image = np.clip(image,0,1)
        plt.subplot(2,4,i + 1)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.axis("off")
    plt.tight_layout()
    plt.show()