# ============================================================
# Compare Baseline CNN vs ResNet18 vs EfficientNet-B0
# Brain MRI Tumor Classification
# ============================================================

import sys
from pathlib import Path

# ============================================================
# 1. Add src Directory to Python Path
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 2. Imports
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import create_dataloaders

from models import BaselineCNN
from resnet_model import ResNet18Model
from efficientnet_model import EfficientNetB0Model

from config import (
    DEVICE,
    NUM_CLASSES,
    CLASS_NAMES,
    BASELINE_MODEL_PATH,
    RESNET18_MODEL_PATH,
    EFFICIENTNET_B0_MODEL_PATH,
    BASELINE_HISTORY_PATH,
    RESNET18_HISTORY_PATH,
    EFFICIENTNET_B0_HISTORY_PATH,
    EVALUATION_DIR,
    PLOTS_DIR
)


# ============================================================
# 3. Load Model Checkpoint
# ============================================================

def load_checkpoint(model, model_path):

    print("\nLoading Model From:")
    print(model_path)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # ========================================================
    # Load Model State Dictionary
    # ========================================================

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    # ========================================================
    # Move Model To Device
    # ========================================================

    model = model.to(DEVICE)

    # ========================================================
    # Evaluation Mode
    # ========================================================

    model.eval()

    print("✓ Model Loaded Successfully")

    return model


# ============================================================
# 4. Count Trainable Parameters
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# 5. Evaluate Model
# ============================================================

def evaluate_model(
    model,
    test_loader,
    model_name
):

    print("\n" + "=" * 60)

    print(
        f"EVALUATING: {model_name}"
    )

    print("=" * 60)

    # ========================================================
    # Store Labels and Predictions
    # ========================================================

    all_labels = []

    all_predictions = []

    # ========================================================
    # Evaluation Mode
    # ========================================================

    model.eval()

    # ========================================================
    # Disable Gradient Calculation
    # ========================================================

    with torch.no_grad():

        for images, labels in test_loader:

            # ------------------------------------------------
            # Move Data To Device
            # ------------------------------------------------

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            # ------------------------------------------------
            # Forward Pass
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # Get Predictions
            # ------------------------------------------------

            _, predictions = torch.max(
                outputs,
                1
            )

            # ------------------------------------------------
            # Store Labels
            # ------------------------------------------------

            all_labels.extend(
                labels.cpu().numpy()
            )

            # ------------------------------------------------
            # Store Predictions
            # ------------------------------------------------

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # ========================================================
    # Convert To NumPy
    # ========================================================

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    # ========================================================
    # Calculate Metrics
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    # ========================================================
    # Print Results
    # ========================================================

    print("\n" + "-" * 60)

    print(
        f"Model      : {model_name}"
    )

    print(
        f"Accuracy   : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision  : {precision * 100:.2f}%"
    )

    print(
        f"Recall     : {recall * 100:.2f}%"
    )

    print(
        f"F1-Score   : {f1 * 100:.2f}%"
    )

    print(
        f"Parameters : {count_parameters(model):,}"
    )

    print("-" * 60)

    # ========================================================
    # Classification Report
    # ========================================================

    print("\nClassification Report:")

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(report)

    # ========================================================
    # Save Classification Report
    # ========================================================

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    report_path = (
        EVALUATION_DIR
        / f"{report_filename}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{model_name} Classification Report\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(report)

    print(
        "\nClassification Report Saved At:"
    )

    print(
        report_path
    )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    print("\nConfusion Matrix:")

    print(cm)

    # ========================================================
    # Plot Confusion Matrix
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.tight_layout()

    confusion_matrix_path = (
        EVALUATION_DIR
        / f"{report_filename}_confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        "\nConfusion Matrix Saved At:"
    )

    print(
        confusion_matrix_path
    )

    # ========================================================
    # Return Results
    # ========================================================

    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "Parameters": count_parameters(model)

    }


# ============================================================
# 6. Load Training History
# ============================================================

def load_history(history_path):

    history = torch.load(
        history_path,
        map_location="cpu"
    )

    return history


# ============================================================
# 7. Plot Training History Comparison
# ============================================================

def plot_history_comparison(
    histories
):

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Accuracy Comparison
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    for model_name, history in histories.items():

        epochs = range(
            1,
            len(
                history["train_accuracy"]
            ) + 1
        )

        plt.plot(
            epochs,
            history["train_accuracy"],
            marker="o",
            label=f"{model_name} - Train"
        )

        plt.plot(
            epochs,
            history["val_accuracy"],
            marker="x",
            linestyle="--",
            label=f"{model_name} - Validation"
        )

    plt.title(
        "Training vs Validation Accuracy Comparison"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    accuracy_path = (
        PLOTS_DIR
        / "model_accuracy_comparison.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        "\nAccuracy Comparison Saved At:"
    )

    print(
        accuracy_path
    )

    # ========================================================
    # Loss Comparison
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    for model_name, history in histories.items():

        epochs = range(
            1,
            len(
                history["train_loss"]
            ) + 1
        )

        plt.plot(
            epochs,
            history["train_loss"],
            marker="o",
            label=f"{model_name} - Train"
        )

        plt.plot(
            epochs,
            history["val_loss"],
            marker="x",
            linestyle="--",
            label=f"{model_name} - Validation"
        )

    plt.title(
        "Training vs Validation Loss Comparison"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    loss_path = (
        PLOTS_DIR
        / "model_loss_comparison.png"
    )

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        "\nLoss Comparison Saved At:"
    )

    print(
        loss_path
    )


# ============================================================
# 8. Plot Metrics Comparison
# ============================================================

def plot_metrics_comparison(
    results_df
):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    # ========================================================
    # Convert To Percentage
    # ========================================================

    plot_df = results_df.copy()

    for metric in metrics:

        plot_df[metric] = (
            plot_df[metric] * 100
        )

    # ========================================================
    # Plot
    # ========================================================

    plot_df.set_index(
        "Model"
    )[metrics].plot(
        kind="bar",
        figsize=(12, 7)
    )

    plt.title(
        "Brain MRI Model Performance Comparison"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    metrics_path = (
        PLOTS_DIR
        / "model_metrics_comparison.png"
    )

    plt.savefig(
        metrics_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        "\nMetrics Comparison Saved At:"
    )

    print(
        metrics_path
    )


# ============================================================
# 9. Save Comparison Results
# ============================================================

def save_comparison_results(
    results_df
):

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = (
        EVALUATION_DIR
        / "model_comparison_results.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    # ========================================================
    # Save Text Report
    # ========================================================

    report_path = (
        EVALUATION_DIR
        / "model_comparison_results.txt"
    )

    best_model_index = results_df[
        "F1-Score"
    ].idxmax()

    best_model = results_df.loc[
        best_model_index,
        "Model"
    ]

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "BRAIN MRI TUMOR CLASSIFICATION\n"
        )

        file.write(
            "MODEL COMPARISON RESULTS\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            results_df.to_string(
                index=False
            )
        )

        file.write(
            "\n\n"
        )

        file.write(
            "=" * 60
            + "\n"
        )

        file.write(
            f"BEST MODEL: {best_model}\n"
        )

        file.write(
            f"Best F1-Score: "
            f"{results_df.loc[best_model_index, 'F1-Score'] * 100:.2f}%\n"
        )

        file.write(
            f"Best Accuracy: "
            f"{results_df.loc[best_model_index, 'Accuracy'] * 100:.2f}%\n"
        )

        file.write(
            f"Parameters: "
            f"{results_df.loc[best_model_index, 'Parameters']:,}\n"
        )

        file.write(
            "=" * 60
            + "\n"
        )

    print(
        "\nComparison CSV Saved At:"
    )

    print(
        csv_path
    )

    print(
        "\nComparison Report Saved At:"
    )

    print(
        report_path
    )


# ============================================================
# 10. Main Comparison Function
# ============================================================

def compare_models():

    print("\n")

    print("=" * 60)

    print(
        "BRAIN MRI TUMOR CLASSIFICATION"
    )

    print(
        "MODEL COMPARISON"
    )

    print("=" * 60)

    # ========================================================
    # Device Information
    # ========================================================

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # Create DataLoaders
    # ========================================================

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    print(
        f"\nTraining Samples  : {len(train_dataset)}"
    )

    print(
        f"Validation Samples: {len(val_dataset)}"
    )

    print(
        f"Testing Samples   : {len(test_dataset)}"
    )

    # ========================================================
    # Create Baseline CNN
    # ========================================================

    print(
        "\nCreating Baseline CNN..."
    )

    baseline_model = BaselineCNN(
        num_classes=NUM_CLASSES
    )

    baseline_model = load_checkpoint(
        baseline_model,
        BASELINE_MODEL_PATH
    )

    # ========================================================
    # Create ResNet18
    # ========================================================

    print(
        "\nCreating ResNet18..."
    )

    resnet_model = ResNet18Model(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    resnet_model = load_checkpoint(
        resnet_model,
        RESNET18_MODEL_PATH
    )

    # ========================================================
    # Create EfficientNet-B0
    # ========================================================

    print(
        "\nCreating EfficientNet-B0..."
    )

    efficientnet_model = EfficientNetB0Model(
        num_classes=NUM_CLASSES,
        pretrained=False
    )

    efficientnet_model = load_checkpoint(
        efficientnet_model,
        EFFICIENTNET_B0_MODEL_PATH
    )

    # ========================================================
    # Evaluate Baseline CNN
    # ========================================================

    baseline_results = evaluate_model(
        baseline_model,
        test_loader,
        "Baseline CNN"
    )

    # ========================================================
    # Evaluate ResNet18
    # ========================================================

    resnet_results = evaluate_model(
        resnet_model,
        test_loader,
        "ResNet18"
    )

    # ========================================================
    # Evaluate EfficientNet-B0
    # ========================================================

    efficientnet_results = evaluate_model(
        efficientnet_model,
        test_loader,
        "EfficientNet-B0"
    )

    # ========================================================
    # Combine Results
    # ========================================================

    results = [

        baseline_results,

        resnet_results,

        efficientnet_results

    ]

    # ========================================================
    # Create DataFrame
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # Print Final Comparison
    # ========================================================

    print("\n")

    print("=" * 60)

    print(
        "FINAL MODEL COMPARISON"
    )

    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Find Best Model
    # ========================================================

    best_model_index = results_df[
        "F1-Score"
    ].idxmax()

    best_model = results_df.loc[
        best_model_index,
        "Model"
    ]

    print("\n" + "=" * 60)

    print(
        f"🏆 BEST MODEL: {best_model}"
    )

    print(
        f"Best Accuracy: "
        f"{results_df.loc[best_model_index, 'Accuracy'] * 100:.2f}%"
    )

    print(
        f"Best F1-Score: "
        f"{results_df.loc[best_model_index, 'F1-Score'] * 100:.2f}%"
    )

    print(
        f"Parameters: "
        f"{results_df.loc[best_model_index, 'Parameters']:,}"
    )

    print("=" * 60)

    # ========================================================
    # Save Comparison Results
    # ========================================================

    save_comparison_results(
        results_df
    )

    # ========================================================
    # Plot Metrics Comparison
    # ========================================================

    plot_metrics_comparison(
        results_df
    )

    # ========================================================
    # Load Training Histories
    # ========================================================

    histories = {

        "Baseline CNN":
            load_history(
                BASELINE_HISTORY_PATH
            ),

        "ResNet18":
            load_history(
                RESNET18_HISTORY_PATH
            ),

        "EfficientNet-B0":
            load_history(
                EFFICIENTNET_B0_HISTORY_PATH
            )

    }

    # ========================================================
    # Plot Training History Comparison
    # ========================================================

    plot_history_comparison(
        histories
    )

    # ========================================================
    # Final Message
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "MODEL COMPARISON COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    compare_models()