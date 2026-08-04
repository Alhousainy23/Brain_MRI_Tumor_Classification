# 🧠 Brain MRI Tumor Classification

A deep learning-based computer vision project for classifying brain MRI images into four different categories using **PyTorch** and Convolutional Neural Networks (CNNs).

This project implements and compares three deep learning architectures:

* **Baseline CNN**
* **ResNet18**
* **EfficientNet-B0**

The main objective is to develop an end-to-end brain MRI classification pipeline covering **data preprocessing, data augmentation, model training, evaluation, model comparison, and interactive deployment** using a **CustomTkinter GUI**.

---

## 📌 Project Overview

Brain MRI tumor classification is an important computer vision task that can assist in the automated analysis of medical images.

In this project, MRI images are classified into four categories:

1. **Glioma**
2. **Meningioma**
3. **No Tumor**
4. **Pituitary Tumor**

Three different CNN architectures were trained and evaluated using the same classification task and test dataset.

The project follows a complete deep learning workflow:

```text
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Data Augmentation
   │
   ▼
Train / Validation Split
   │
   ▼
┌───────────────────────────────┐
│       Model Training          │
│                               │
│  • Baseline CNN               │
│  • ResNet18                   │
│  • EfficientNet-B0            │
└───────────────────────────────┘
   │
   ▼
Model Evaluation
   │
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1-Score
   └── Confusion Matrix
   │
   ▼
Model Comparison
   │
   ▼
Best Model Selection
   │
   ▼
CustomTkinter GUI
   │
   ▼
Interactive MRI Classification
```

---

# 📊 Dataset

The dataset contains brain MRI images organized into four classes:

| Class      | Description                |
| ---------- | -------------------------- |
| Glioma     | Glioma tumor class         |
| Meningioma | Meningioma tumor class     |
| No Tumor   | MRI images without a tumor |
| Pituitary  | Pituitary tumor class      |

The dataset is divided into:

* **Training Set:** 4,480 images
* **Validation Set:** 1,120 images
* **Testing Set:** 1,600 images

The test set contains **400 images per class**.

### Dataset Distribution

| Dataset    | Number of Images |
| ---------- | ---------------: |
| Training   |            4,480 |
| Validation |            1,120 |
| Testing    |            1,600 |
| **Total**  |        **7,200** |

---

# 🔧 Data Preprocessing

All MRI images are processed before being passed to the deep learning models.

The preprocessing pipeline includes:

* Resize images to **224 × 224**
* Convert images to **RGB**
* Convert images to PyTorch tensors
* Normalize image pixel values

### Data Augmentation

Training images are augmented using:

* Random Horizontal Flip
* Random Rotation up to ±10°
* Random Affine Transformation
* Random Translation
* Random Scaling

Data augmentation is applied to improve model generalization and reduce overfitting.

---

# 🤖 Deep Learning Models

## 1️⃣ Baseline CNN

A custom Convolutional Neural Network was implemented as the baseline model.

The architecture includes:

* Convolutional Layers
* Batch Normalization
* ReLU Activation
* Max Pooling
* Adaptive Average Pooling
* Fully Connected Layers
* Dropout

The Baseline CNN provides a reference point for comparing the performance of more advanced CNN architectures.

---

## 2️⃣ ResNet18

**ResNet18** was used as a deep residual CNN architecture.

The model uses residual learning and skip connections to improve information flow during training.

ResNet18 was adapted to classify the four target classes:

```text
Glioma
Meningioma
No Tumor
Pituitary
```

---

## 3️⃣ EfficientNet-B0

**EfficientNet-B0** was used as a modern and efficient CNN architecture.

The model was selected to investigate the balance between classification performance and model complexity.

EfficientNet-B0 achieved the best overall performance among the evaluated models.

---

# ⚙️ Training Configuration

The models were trained using the following configuration:

| Parameter               | Configuration             |
| ----------------------- | ------------------------- |
| Framework               | PyTorch                   |
| Input Size              | 224 × 224                 |
| Number of Classes       | 4                         |
| Batch Size              | 32                        |
| Maximum Epochs          | 30                        |
| Optimizer               | Adam                      |
| Learning Rate           | 0.001                     |
| Weight Decay            | 0.0001                    |
| Early Stopping Patience | 5                         |
| Validation Split        | 20%                       |
| Random Seed             | 42                        |
| Device                  | CUDA / GPU when available |

The models were trained using GPU acceleration with:

```text
NVIDIA GeForce RTX 3060 Laptop GPU
```

---

# 📈 Model Evaluation

Each trained model was evaluated on the same testing dataset.

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Trainable Parameters

Classification reports and confusion matrices were generated for each model.

---

# 🏆 Model Comparison Results

The final evaluation produced the following results:

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |    Parameters |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ------------: |
| Baseline CNN        |     80.88% |     81.26% |     80.88% |     80.52% |       619,268 |
| ResNet18            |     93.12% |     93.83% |     93.12% |     92.99% |    11,178,564 |
| **EfficientNet-B0** | **94.25%** | **94.70%** | **94.25%** | **94.12%** | **4,012,672** |

---

# 🥇 Best Model

Based on the experimental evaluation, **EfficientNet-B0** achieved the best overall performance.

### EfficientNet-B0 Results

```text
Accuracy   : 94.25%
Precision  : 94.70%
Recall     : 94.25%
F1-Score   : 94.12%
Parameters : 4,012,672
```

EfficientNet-B0 achieved a higher classification performance than both the Baseline CNN and ResNet18.

An important observation is that EfficientNet-B0 achieved better performance than ResNet18 while using significantly fewer trainable parameters.

```text
ResNet18
11,178,564 Parameters
Accuracy: 93.12%

        VS

EfficientNet-B0
4,012,672 Parameters
Accuracy: 94.25%
```

This demonstrates a favorable balance between predictive performance and model complexity in this experiment.

---

# 🔍 Class-Level Performance

## Baseline CNN

The Baseline CNN achieved its strongest performance on the **No Tumor** and **Pituitary** classes.

The most challenging class was **Meningioma**, with:

```text
Precision : 77%
Recall    : 64%
F1-Score  : 70%
```

---

## ResNet18

ResNet18 achieved strong performance on:

* No Tumor
* Pituitary

The **Pituitary** class achieved:

```text
Precision : 99%
Recall    : 97%
F1-Score  : 98%
```

However, **Glioma** remained relatively challenging, with a recall of **78%**.

---

## EfficientNet-B0

EfficientNet-B0 achieved its strongest results on:

### Pituitary

```text
Precision : 99%
Recall    : 99%
F1-Score  : 99%
```

### No Tumor

```text
Precision : 92%
Recall    : 100%
F1-Score  : 96%
```

The **Glioma** class remained the most challenging class for the model in terms of recall:

```text
Precision : 99%
Recall    : 80%
F1-Score  : 89%
```

This indicates that some Glioma samples were classified as other tumor categories, particularly Meningioma and No Tumor.

---

# 📊 Generated Evaluation Outputs

The model comparison pipeline generates several outputs.

### Evaluation Reports

```text
outputs/evaluation/
│
├── baseline_cnn_classification_report.txt
├── resnet18_classification_report.txt
├── efficientnet_b0_classification_report.txt
├── model_comparison_results.csv
└── model_comparison_results.txt
```

### Confusion Matrices

```text
outputs/evaluation/
│
├── baseline_cnn_confusion_matrix.png
├── resnet18_confusion_matrix.png
└── efficientnet_b0_confusion_matrix.png
```

### Comparison Plots

```text
outputs/plots/
│
├── model_metrics_comparison.png
├── model_accuracy_comparison.png
└── model_loss_comparison.png
```

These visualizations are used to analyze and compare the performance and training behavior of the three models.

---

# 🖥️ Graphical User Interface

The project includes an interactive **CustomTkinter GUI** for model inference.

The GUI is designed using **Light Mode**.

The application allows users to:

* Load an MRI image
* Load a new image at any time
* Select a trained model
* Choose between:

  * Baseline CNN
  * ResNet18
  * EfficientNet-B0
* Run image classification
* Display the predicted class
* Display prediction confidence
* Display model performance information
* Identify the best-performing model
* Stop the current process
* Exit the application

The GUI provides a simple interface for testing the trained deep learning models on new MRI images.

---

# 📁 Project Structure

```text
Brain_MRI_Tumor_Classification/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── models/
│   ├── baseline_cnn_best.pth
│   ├── resnet18_best.pth
│   └── efficientnet_b0_best.pth
│
├── outputs/
│   ├── evaluation/
│   ├── plots/
│   └── architectures/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── baseline_train.py
│   ├── resnet_train.py
│   ├── efficientnet_train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── visualization.py
│   └── gui_app.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Pillow**
* **CustomTkinter**

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/Brain_MRI_Tumor_Classification.git
```

Navigate to the project directory:

```bash
cd Brain_MRI_Tumor_Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Train Baseline CNN

```bash
python src/baseline_train.py
```

## Train ResNet18

```bash
python src/resnet_train.py
```

## Train EfficientNet-B0

```bash
python src/efficientnet_train.py
```

## Compare Models

After training the models, run:

```bash
python src/compare_models.py
```

This evaluates all trained models and generates:

* Model comparison results
* Classification reports
* Confusion matrices
* Accuracy comparison
* Loss comparison
* Metrics comparison

---

# 🖥️ Run the GUI

After completing the training and model comparison process, launch the graphical interface:

```bash
python src/gui_app.py
```

The user can then:

1. Load an MRI image.
2. Select a trained model.
3. Run prediction.
4. View the predicted tumor class.
5. View the prediction confidence.
6. Load another MRI image.
7. Stop or exit the application.

---

# 🔬 Research Project

This project can also serve as the foundation for a research study on:

**Deep Learning-Based Brain MRI Tumor Classification: A Comparative Study of Baseline CNN, ResNet18, and EfficientNet-B0**

The research investigates the performance of different CNN architectures and analyzes the relationship between classification performance and model complexity.

---

# 🚧 Future Work

Future improvements may include:

* Applying advanced transfer learning and fine-tuning strategies.
* Testing additional deep learning architectures.
* Using ensemble learning.
* Applying Explainable AI techniques such as Grad-CAM.
* Performing external validation using independent datasets.
* Improving the GUI for real-world deployment.
* Packaging the application as a standalone executable.
* Optimizing the models for faster inference.
* Investigating methods to improve Glioma classification performance.

---

# ⚠️ Disclaimer

This project is developed for **educational and research purposes**.

The system is **not intended to replace professional medical diagnosis or clinical decision-making**.

---

# 👨‍💻 Author

**Alhousainy Abdelrahman**

IT Specialist Engineer | Computer Vision | Deep Learning | Intelligent Systems

---

⭐ If you find this project useful, feel free to explore the repository and follow the development of the project.

---

## 📌 Project Status

**Status:** Completed Deep Learning Classification Pipeline
**Best Model:** EfficientNet-B0
**Best Accuracy:** 94.25%
**Best F1-Score:** 94.12%
**Deployment:** CustomTkinter GUI
