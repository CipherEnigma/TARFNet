# TARFNet: Temporal Attention Residual Fusion Network for Subtle Sleep Movement Detection

This repository contains the implementation of **TARFNet**, a deep learning framework developed for binary classification of subtle movements in long-form sleep videos. The model is lightweight, avoids the use of optical flow or skeleton estimation, and is optimized for real-time deployment in sleep monitoring scenarios.

---

##  Key Features

- **CNN + TCN + Attention Architecture**
- Works with raw RGB video frames (no motion preprocessing)
- Optimized for real-time performance
- Binary classification: movement vs no movement
- Modular and clean codebase

---

#  Architecture 

- **Backbone**: Pretrained ResNet-18 for frame-wise spatial feature extraction
- **Temporal Modeling**: Dilated Residual Temporal Convolutional Blocks (TCNs)
- **Attention**: Multi-Head Self-Attention layer to focus on key motion frames
- **Fusion**: Residual fusion of attention-enhanced + TCN features
- **Classifier**: Fully connected binary classification head
  
---

## How to Run this model

**Install Dependencies**

- pip install -r requirements.txt


## Metrics Used
- Accuracy

- Precision

- Recall

- F1 Score

## Author
Pragunie Aditya

Shiv Nadar Institute of Eminence

Project under Prof. Tapan K. Gandhi, IIT Delhi



