# Deep Learning-Based Brain Stroke Detection Using CT Images

A comprehensive deep learning project for brain stroke detection using CT scan images with multiple CNN, Transformer, Hybrid, and Attention-based architectures.

## Overview

This project focuses on automated brain stroke detection using CT scan images and compares the performance of 26 machine learning and deep learning models.

The project includes:
- CNN Transfer Learning Models
- Hybrid CNN + RNN Architectures
- Attention-Based Models
- Vision Transformers
- Swin Transformers
- Focal Loss Optimization
- Grad-CAM Explainability

The best-performing model achieved:

- Accuracy: 98.40%
- F1-Score: 97.92%
- AUC Score: 0.9979

using a fine-tuned DenseNet architecture.

---

# Models Implemented

## Classical Machine Learning
- Logistic Regression
- Decision Tree
- Random Forest
- Naive Bayes
- SVM
- ANN
- DNN

## CNN Transfer Learning
- VGG19
- DenseNet121
- DenseNet169
- EfficientNetB0
- MobileNetV2
- ResNet50

## Hybrid Models
- VGG19 + BiLSTM
- CNN + BiLSTM
- MobileNetV2 + GRU
- MobileNetV2 + BiLSTM

## Attention-Based Models
- DenseNet + CBAM
- ResNet50 + Attention
- EfficientNet + Transformer

## Transformer Models
- Vision Transformer (ViT)
- Swin Transformer
- Swin Transformer Tiny

---

# Dataset

Dataset used:
Brain Stroke CT Image Dataset from Kaggle

- 1900 CT Images
- 950 Stroke Images
- 950 Normal Images

Image size:
- 224 × 224

Dataset Link:
https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset

---

# Training Configuration

- Optimizer: Adam
- Learning Rate: 1e-4
- Batch Size: 32
- Early Stopping
- Data Augmentation
  - Rotation
  - Flipping
  - Zoom
  - Brightness Adjustment



# Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Confusion Matrix


# Model Performance Comparison

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC |
|---|---|---|---|---|---|
| DenseNet | 98.40 | 97.24 | 98.60 | 97.92 | 0.9979 |
| EfficientNet | 98.14 | 97.89 | 97.20 | 97.54 | 0.9976 |
| Swin Transformer Tiny | 96.81 | 93.96 | 97.90 | 95.89 | 0.9936 |
| MobileNetV2 + Focal Loss | 96.28 | 92.16 | 98.60 | 95.27 | 0.9950 |
| Vision Transformer (ViT) | 93.35 | 87.82 | 95.80 | 91.64 | 0.9869 |
| Swin Transformer | 93.09 | 95.35 | 86.01 | 90.44 | 0.9796 |
| VGG19 + BiLSTM | 74.40 | 63.00 | 77.00 | 70.00 | — |
| DenseNet169 | 74.20 | 69.00 | 58.00 | 63.00 | — |
| DenseNet121 | 73.40 | 64.00 | 69.00 | 66.00 | — |
| EfficientNet + Transformer | 71.00 | 65.00 | 53.00 | 58.00 | — |
| DenseNet169 Transfer Learning | 70.60 | 63.00 | 54.00 | 58.00 | — |
| DenseNet + CBAM | 68.80 | 58.00 | 68.00 | 62.00 | — |
| EfficientNetB0 + DenseNet121 | 68.40 | 58.00 | 63.00 | 60.00 | — |
| MobileNetV2 + GRU | 65.80 | 55.00 | 58.00 | 56.00 | — |
| CNN (Baseline) | 63.00 | 54.00 | 16.00 | 25.00 | — |
| ResNet50 | 62.40 | 55.00 | 6.00 | 11.00 | — |
| ResNet50 + Attention | 62.00 | 0.00 | 0.00 | 0.00 | — |
| CNN + BiLSTM | 61.20 | 47.00 | 17.00 | 25.00 | — |
| MobileNetV2 | 60.60 | 32.00 | 3.00 | 6.00 | — |
| ResNet50 + Deep Dense | 60.00 | 47.00 | 48.00 | 48.00 | — |
| ResNet50 Tuned | 58.80 | 46.00 | 55.00 | 50.00 | — |
| MobileNetV2 Tuned | 57.80 | 37.00 | 16.00 | 22.00 | — |
| ResNet50 + GAP + GMP | 56.00 | 44.00 | 56.00 | 49.00 | — |
| MobileNetV2 + BiLSTM | 53.20 | 43.00 | 75.00 | 55.00 | — |
| MobileNetV2 (Alternative) | 46.20 | 40.00 | 78.00 | 53.00 | — |



# Best Results

| Model | Accuracy | AUC |
|------|------|------|
| DenseNet | 98.40% | 0.9979 |
| EfficientNet | 98.14% | 0.9976 |
| Swin Tiny | 96.81% | 0.9936 |
| MobileNetV2 + Focal Loss | 96.28% | 0.9950 |



# Explainability

Grad-CAM visualization was used to highlight clinically relevant regions in CT images and improve model interpretability.


# Technologies Used

- Python
- TensorFlow
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn


TO CLONE 

https://github.com/Erramshettiabhilash/Brain-Stroke-Detection
