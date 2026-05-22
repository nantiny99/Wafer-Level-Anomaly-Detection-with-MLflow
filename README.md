# Wafer-Level-Anomaly-Detection-with-MLflow
This project evaluates anomaly detection models on semiconductor wafer manufacturing datasets under highly imbalanced industrial conditions.   The goal is to understand model effectiveness, failure behavior, and anomaly detection performance rather than relying on misleading accuracy scores.

## Problem
Semiconductor manufacturing defects are rare, costly, and difficult to detect.  
In wafer inspection, most samples are normal, while defective wafers appear only in very small numbers.

Due to this imbalance, accuracy can be misleading:
- A model can predict every wafer as normal
- Accuracy may still appear very high
- Anomaly recall can be zero

This project focuses on more meaningful anomaly detection metrics, such as:
- Precision
- Recall
- F1-score
- Failure behavior across datasets

## Dataset
Dataset used:
**STMicroelectronics ST-AWFD Dataset**  
https://github.com/STMicroelectronics/ST-AWFD

Datasets evaluated:
- **Wafer 1**: More separable anomalies
- **Wafer 2**: Higher overlap and process drift

## Models Used
| Model | Type | Labels Required |
|---|---|---|
| Isolation Forest | Unsupervised ML | No |
| Autoencoder | Unsupervised Deep Learning | No |

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- MLflow
- Jupyter Notebook

## MLflow Tracking
MLflow was used to track and compare experiment runs.

Tracked items include:
- Model type
- Dataset version
- Accuracy
- Precision
- Recall
- F1-score
- Model artifacts

Example MLflow runs:
- `IsolationForest_D1`
- `IsolationForest_D2`
- `Autoencoder_D1`
- `Autoencoder_D2`

## Output
The project compares Isolation Forest and Autoencoder performance across wafer datasets.

Key observations:
- Accuracy alone is not reliable for imbalanced anomaly detection
- Isolation Forest performs better when anomalies are more separable
- Isolation Forest struggles with overlapping process distributions
- Autoencoder performance depends heavily on reconstruction threshold selection
- Recall and F1-score are more useful than accuracy for evaluating defect detection

## Results Summary
| Dataset | Model | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Wafer 1 | Isolation Forest | 0.7919 | 0.8569 | 0.7893 | 0.8217 |
| Wafer 1 | Autoencoder | 0.9489 | 0.0000 | 0.0000 | 0.0000 |
| Wafer 2 | Isolation Forest | 0.6357 | 0.0013 | 0.4286 | 0.0026 |
| Wafer 2 | Autoencoder | 0.3892 | 0.4686 | 0.0386 | 0.0713 |

## Conclusion
Wafer anomaly detection requires more than high accuracy.  
In realistic manufacturing conditions, models must be evaluated using precision, recall, F1-score, and failure analysis.

Isolation Forest provides a fast baseline but struggles with overlapping distributions.  
Autoencoders offer flexibility for unlabeled data but require careful threshold tuning.

This project demonstrates reproducible anomaly detection experimentation using MLflow.
