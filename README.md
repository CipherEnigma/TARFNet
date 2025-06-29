# DeepFlow
# Movement Detection from Long-Duration Video Footage

## Overview

This project aims to detect movement events from long-duration videos, specifically focusing on sparse and subtle motion such as those observed during sleep. The goal is to develop a robust deep learning pipeline capable of frame-wise classification and temporal analysis, eventually contributing to healthcare-related video analytics.

## Objectives

- Detect motion vs. stillness in low-frame-rate, long-duration videos
- Design a hybrid model combining CNN-based spatial detection with temporal modeling (e.g., 3D CNN, LSTM)
- Evaluate performance using F1-score, ROC-AUC, and event-level detection metrics
- Explore additional enhancements such as optical flow and pose estimation

## Dataset

- **Source**: Self-collected long-duration sleep video (frame rate: 1 frame/3 seconds)
- **Annotations**: Subset of frames manually labeled via Roboflow (`movement`, `stillness`)
- **Format**: Roboflow-exported dataset (e.g., YOLO, COCO JSON)

## Methodology

1. **Frame Extraction**  
   - Extract frames using OpenCV at regular intervals
2. **Annotation & Preprocessing**  
   - Use Roboflow for class annotations
   - Apply image normalization and resizing
3. **Modeling**
   - **Baseline**: ResNet/MobileNet for frame-level classification
   - **Temporal**: CNN + LSTM or 3D CNN to model motion over time
   - Explore additional input modalities (optical flow, pose estimation)
4. **Evaluation**
   - Binary classification metrics (Accuracy, F1, ROC-AUC)
   - Motion event detection accuracy over time windows
  
   ## Currently working on: 

- Integrate optical flow to enhance motion representation
- Explore ConvLSTM and attention-based temporal models
- Evaluate on benchmark datasets (if available)
- Consider pose estimation for fine-grained movement detection

## Acknowledgements

This work is being carried out under the mentorship of **Prof. Tapan Kumar Gandhi (Indian Institute of Technology- Delhi** 

