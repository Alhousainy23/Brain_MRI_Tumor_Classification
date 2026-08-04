from src.config import (CLASS_NAMES,DEVICE)
from src.dataset import (set_seed,create_dataloaders,print_dataset_information,count_images_per_class,plot_class_distribution,show_sample_images)
def main():
    # ========================================================
    # 1. Set Random Seed
    # ========================================================
    set_seed()
    # ========================================================
    # 2. Print Device
    # ========================================================
    print("\n" + "=" * 60)
    print("BRAIN TUMOR MRI CLASSIFICATION PROJECT")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {DEVICE}")
        print(f"GPU Name: "f"{__import__('torch').cuda.get_device_name(0)}")
    # ========================================================
    # 3. Create Datasets and DataLoaders
    # ========================================================
    (train_dataset,val_dataset,test_dataset,train_loader,val_loader,test_loader) = create_dataloaders()
    # ========================================================
    # 4. Print Dataset Information
    # ========================================================
    print_dataset_information(train_dataset,val_dataset,test_dataset,CLASS_NAMES)
    # ========================================================
    # 5. Count Images per Class
    # ========================================================
    train_counts = count_images_per_class(train_dataset,CLASS_NAMES,"Training")
    val_counts = count_images_per_class(val_dataset,CLASS_NAMES,"Validation")
    test_counts = count_images_per_class(test_dataset,CLASS_NAMES,"Testing")
    # ========================================================
    # 6. Plot Class Distribution
    # ========================================================
    plot_class_distribution(train_counts,"Training")
    plot_class_distribution(val_counts,"Validation")
    plot_class_distribution(test_counts,"Testing")
    # ========================================================
    # 7. Display Sample Images
    # ========================================================
    print("\nDisplaying Sample Images...")
    show_sample_images(train_dataset,CLASS_NAMES,num_images=8)
    # ========================================================
    # 8. Check One Batch
    # ========================================================
    images, labels = next(iter(train_loader))
    print("\n" + "=" * 60)
    print("BATCH INFORMATION")
    print("=" * 60)
    print(f"Images Shape: {images.shape}")
    print(f"Labels Shape: {labels.shape}")
    print(f"Labels: {labels[:10]}")
    print(f"Image Min: {images.min().item():.4f}")
    print(f"Image Max: {images.max().item():.4f}")
    print("\nDataset Preparation Completed Successfully!")
if __name__ == "__main__":
    main()