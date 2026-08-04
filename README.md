============================================================
BRAIN MRI TUMOR CLASSIFICATION
MODEL COMPARISON
============================================================

Device: cuda
GPU: NVIDIA GeForce RTX 3060 Laptop GPU

Training Samples  : 4480
Validation Samples: 1120
Testing Samples   : 1600

Creating Baseline CNN...

Loading Model From:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\models\baseline_cnn_best.pth
✓ Model Loaded Successfully

Creating ResNet18...

Loading Model From:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\models\resnet18_best.pth
✓ Model Loaded Successfully

Creating EfficientNet-B0...

Loading Model From:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\models\efficientnet_b0_best.pth
✓ Model Loaded Successfully

============================================================
EVALUATING: Baseline CNN
============================================================

------------------------------------------------------------
Model      : Baseline CNN
Accuracy   : 80.88%
Precision  : 81.26%
Recall     : 80.88%
F1-Score   : 80.52%
Parameters : 619,268
------------------------------------------------------------

Classification Report:
              precision    recall  f1-score   support

      glioma       0.79      0.73      0.76       400
  meningioma       0.77      0.64      0.70       400
     notumor       0.75      0.97      0.84       400
   pituitary       0.94      0.90      0.92       400

    accuracy                           0.81      1600
   macro avg       0.81      0.81      0.81      1600
weighted avg       0.81      0.81      0.81      1600


Classification Report Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\evaluation\baseline_cnn_classification_report.txt

Confusion Matrix:
[[291  49  57   3]
 [ 51 256  73  20]
 [  6   7 387   0]
 [ 20  20   0 360]]

Confusion Matrix Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\evaluation\baseline_cnn_confusion_matrix.png

============================================================
EVALUATING: ResNet18
============================================================

------------------------------------------------------------
Model      : ResNet18
Accuracy   : 93.12%
Precision  : 93.83%
Recall     : 93.12%
F1-Score   : 92.99%
Parameters : 11,178,564
------------------------------------------------------------

Classification Report:
              precision    recall  f1-score   support

      glioma       0.99      0.78      0.87       400
  meningioma       0.84      0.98      0.91       400
     notumor       0.93      0.99      0.96       400
   pituitary       0.99      0.97      0.98       400

    accuracy                           0.93      1600
   macro avg       0.94      0.93      0.93      1600
weighted avg       0.94      0.93      0.93      1600


Classification Report Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\evaluation\resnet18_classification_report.txt

Confusion Matrix:
[[310  62  27   1]
 [  4 393   1   2]
 [  0   3 397   0]
 [  0   9   1 390]]

Confusion Matrix Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\evaluation\resnet18_confusion_matrix.png

============================================================
EVALUATING: EfficientNet-B0
============================================================

------------------------------------------------------------
Model      : EfficientNet-B0
Accuracy   : 94.25%
Precision  : 94.70%
Recall     : 94.25%
F1-Score   : 94.12%
Parameters : 4,012,672
------------------------------------------------------------

Classification Report:
              precision    recall  f1-score   support

      glioma       0.99      0.80      0.89       400
  meningioma       0.89      0.98      0.93       400
     notumor       0.92      1.00      0.96       400
   pituitary       0.99      0.99      0.99       400

    accuracy                           0.94      1600
   macro avg       0.95      0.94      0.94      1600
weighted avg       0.95      0.94      0.94      1600


Classification Report Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\evaluation\efficientnet_b0_classification_report.txt

Confusion Matrix:
[[321  45  30   4]
 [  2 392   6   0]
 [  0   0 400   0]
 [  0   5   0 395]]

Confusion Matrix Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\evaluation\efficientnet_b0_confusion_matrix.png


============================================================
FINAL MODEL COMPARISON
============================================================
          Model  Accuracy  Precision  Recall  F1-Score  Parameters
   Baseline CNN   0.80875   0.812586 0.80875  0.805216      619268
       ResNet18   0.93125   0.938274 0.93125  0.929947    11178564
EfficientNet-B0   0.94250   0.947023 0.94250  0.941189     4012672

============================================================
🏆 BEST MODEL: EfficientNet-B0
Best Accuracy: 94.25%
Best F1-Score: 94.12%
Parameters: 4,012,672
============================================================

Comparison CSV Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\evaluation\model_comparison_results.csv

Comparison Report Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\evaluation\model_comparison_results.txt

Metrics Comparison Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\plots\model_metrics_comparison.png

Accuracy Comparison Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\plots\model_accuracy_comparison.png

Loss Comparison Saved At:
D:\AI\Elvorix Diploma\3. Deep Learning\5. Final Project\Code\outputs\plots\model_loss_comparison.png

============================================================
MODEL COMPARISON COMPLETED SUCCESSFULLY
============================================================
