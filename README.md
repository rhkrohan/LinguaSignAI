# ✨ American Sign Language Detection Model

> **A comprehensive machine learning project for real-time and high-accuracy ASL gesture detection, combining keypoint-based modeling and transfer learning.**

---
## 📑 Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Approaches](#approaches)
   - [Approach 1: Keypoint-Based Recognition](#approach-1-keypoint-based-gesture-recognition-custom-model-with-mediapipe)
   - [Approach 2: Transfer Learning on Pre-Trained Models](#approach-2-transfer-learning-on-pre-trained-deep-learning-models)
4. [Model Architectures](#model-architectures)
5. [Training Process](#training-process)
6. [Evaluation and Results](#evaluation-and-results)
7. [Future Scope](#future-scope)
8. [Conclusion](#conclusion)
---

## Introduction

Sign language is a vital communication method for the Deaf and Hard-of-Hearing community. However, the need for interpreters or specialized language training can create communication barriers in everyday interactions. To help address this challenge, our project focuses on building a machine learning-based **American Sign Language (ASL) detection model** that recognizes hand gestures in real time or from recorded video frames.

In this repository, we combine computer vision techniques and deep learning approaches to accurately classify a range of ASL gestures. Our workflow involves data extraction and preprocessing, model training using both custom and transfer learning strategies, and systematic experimentation to refine accuracy. Below is an overview of our motivations, goals, and how this project is organized:

1. **Motivation**  
   - **Bridging Communication Gaps**: Facilitating more inclusive communication for Deaf and Hard-of-Hearing individuals.  
   - **Practical Applications**: Enabling ASL recognition in various domains, such as interactive devices, educational software, and assistive technology.

2. **Goals**  
   - **Automate Gesture Recognition**: Develop an automated pipeline to detect and classify ASL signs directly from hand movement data.  
   - **Achieve High Accuracy**: Leverage both custom and transfer learning methods to improve recognition performance.  
   - **Promote Accessibility**: Provide a foundation for accessible tools that integrate gesture detection into real-world applications.

3. **Methodology Overview**  
   - **Data Extraction**: Using [MediaPipe](https://google.github.io/mediapipe/) to extract keypoints and landmarks from images or videos. (See the notebook *HandGestureDataExtraction(MediaPipe) (1).ipynb* for the hand-tracking pipeline.)  
   - **Custom Model Development**: Training a fresh neural network on the extracted dataset. (Detailed in *SignLanguageGestureDetectionModel (1).ipynb* and *SignLanguageDetection(trimmed_data).ipynb*.)  
   - **Transfer Learning**: Fine-tuning pre-trained image/video models to leverage learned features and boost performance. (Outlined in *TransferLearningModel.ipynb*.)

4. **Project Organization**  
   - **Notebooks**: Containing separate exploratory experiments, data-processing pipelines, and finalized model training.  
   - **Data Preprocessing**: Scripts and functions that read, clean, and format the dataset for quick prototyping.  
   - **Documentation**: This README (and other documents) explaining how each component fits together.  

By combining computer vision-based keypoint extraction and deep learning classification, we aim to deliver a robust solution for ASL gesture detection. In the subsequent sections, you’ll find details on the dataset, different modeling approaches, training procedures, and performance metrics that collectively illustrate the evolution of this project—from initial data collection to a refined sign language detection pipeline.

---

## Project Overview

### Objectives
The primary objectives of this project are:
- **Develop a real-time ASL recognition system** that classifies hand gestures efficiently.
- **Experiment with multiple approaches** to determine the most accurate and scalable method.
- **Leverage both custom feature extraction and transfer learning** for robust gesture classification.
- **Contribute to assistive technology** by providing a foundation for ASL recognition applications.

### Methodologies
We explored **two primary approaches** to solving this problem:

#### 1️⃣ Keypoint-Based Recognition (Custom Model with MediaPipe)
- **Directory:** `hand_landmark_model/`  
- Utilizes **Google’s MediaPipe** to extract hand landmarks from video frames.  
- Features a **custom neural network** trained on the extracted keypoints.  
- Efficient and lightweight, focusing on the positional relationship between fingers.  
- Best suited for real-time applications where speed is a priority.  
- *Notebook Reference:* `HandGestureDataExtraction(MediaPipe).ipynb` and `SignLanguageDetectionModel.ipynb`

#### 2️⃣ Transfer Learning on Pre-Trained Models
- **Directory:** `transfer_learning_model/`  
- Leverages pre-trained deep learning models (e.g., **MobileNet, ResNet, EfficientNet**) to extract rich visual features.  
- Works directly with **raw images/videos** instead of relying on keypoints.  
- More computationally intensive but **achieves higher accuracy** in classification.  
- Best suited for offline processing or high-performance applications.  
- *Notebook Reference:* `TransferLearningModel.ipynb`

### System Pipeline
1. **Data Collection & Preprocessing:**  
   - Raw image/video collection or keypoint extraction using MediaPipe.
   - Data augmentation for increased model generalization.

2. **Model Training & Evaluation:**  
   - Training neural networks for both keypoint-based and image-based models.
   - Hyperparameter tuning and performance optimization.

3. **Real-Time or Batch Inference:**  
   - Implementing real-time detection for keypoint models.
   - Running batch predictions for transfer learning models.

### Results & Findings
- The **keypoint-based model** was lightweight and performed well in real-time scenarios but struggled with complex gestures.  
- The **transfer learning model** achieved **higher accuracy** due to richer feature extraction but required more computational power.  
- A **hybrid approach**, combining keypoint tracking with deep feature extraction, may offer the best balance between speed and accuracy.

### Future Scope
- **Expand gesture vocabulary** to recognize a broader range of ASL signs.  
- **Improve model generalization** across different lighting conditions and backgrounds.  
- **Optimize for edge devices** to enable real-time processing on mobile phones or embedded systems.  

This project serves as a foundational step towards making **gesture-based communication more accessible** through AI-driven solutions.

---

## Approaches

In developing our ASL detection model, we experimented with **two distinct methodologies** to recognize hand gestures. Each approach had unique strengths and trade-offs, which we carefully evaluated through experimentation.

---

### Approach 1: Keypoint-Based Gesture Recognition (Custom Model with MediaPipe)

#### Methodology
This approach utilized **Google's MediaPipe library** to extract **21 key hand landmarks** (x, y, z coordinates) from images and video frames. Instead of feeding raw images into a deep learning model, we trained a custom **lightweight neural network** on these extracted keypoints to classify hand gestures. The pipeline involved:

1. **Data Collection & Preprocessing**  
   - Used **MediaPipe Hands** to extract **x, y, z** coordinates for each of the 21 hand landmarks.  
   - Stored keypoint data as structured numerical datasets for efficient training.  
   - Augmented data using random transformations to improve generalization.

2. **Model Training**  
   - Built a **fully connected neural network (MLP - Multi-Layer Perceptron)** using TensorFlow/Keras.  
   - Tuned hyperparameters (learning rate, dropout, activation functions).  
   - Trained on extracted keypoint datasets instead of raw images.

3. **Inference & Real-Time Detection**  
   - MediaPipe continuously extracted keypoints from a webcam feed.  
   - The trained model classified the gesture based on the real-time keypoint data.  
   - Displayed the detected ASL sign with a confidence score.

#### Reasoning
- **Efficiency:** Numerical inputs from keypoints are faster to process than images, making it ideal for **real-time applications**.  
- **Feature Extraction Control:** By focusing on hand landmarks, we minimized the influence of background noise and lighting conditions.

#### Advantages
- ✔️ **Fast & Efficient** – Processes only **numerical keypoints**, making it computationally lightweight.  
- ✔️ **Real-time Ready** – Works well on lower-end hardware, including mobile devices.  
- ✔️ **Robust to Background Variations** – Ignores pixel intensities and thus less affected by lighting changes.

#### Challenges
- ❌ **Loss of Contextual Information** – Ignores visual cues like orientation, texture, or motion.  
- ❌ **Limited Accuracy for Complex Gestures** – Subtle finger movements or two-hand interactions can be missed.  
- ❌ **Rotation & Perspective Variability** – MediaPipe’s 3D depth estimation can be inconsistent at certain angles.

**Why We Moved to the Second Approach?**  
While this method was promising, it struggled with **complex or two-handed gestures**. To improve accuracy, we moved to a **deep learning-based image classification** approach using transfer learning.

---

### Approach 2: Transfer Learning on Pre-Trained Deep Learning Models

#### Methodology
This approach leveraged **pre-trained convolutional neural networks (CNNs)** to classify ASL gestures directly from **raw images**. The pipeline involved:

1. **Data Preparation**  
   - Used a dataset of labeled ASL gesture images.  
   - Applied **data augmentation** (rotation, flipping, contrast changes) to improve generalization.

2. **Feature Extraction with Transfer Learning**  
   - Experimented with **MobileNetV2, ResNet50, and EfficientNet** (pre-trained on ImageNet).  
   - Replaced the final classification layer with a **custom dense layer** for ASL sign recognition.

3. **Fine-Tuning & Training**  
   - Initially trained only the new classification head.  
   - Fine-tuned deeper CNN layers to adapt them for ASL recognition.  
   - Used **categorical cross-entropy** loss and **Adam** optimizer.

4. **Inference & Deployment**  
   - Captured real-time frames via webcam.  
   - Preprocessed frames (resized, normalized) before inference.  
   - Model returned the predicted ASL gesture with confidence scores.

#### Reasoning
- **Higher Accuracy Potential:** CNNs learn richer representations, allowing for **better distinction** among similar gestures.  
- **Handles Complex Gestures:** Full images capture spatial and textural details, supporting intricate two-hand signs.  
- **Leverages Existing Research:** Transfer learning reduces computation and training time compared to building from scratch.

#### Advantages
- ✔️ **Higher Accuracy** – Performs well with complex gestures and varied backgrounds.  
- ✔️ **Better Generalization** – Learns robust features from large-scale datasets (ImageNet).  
- ✔️ **Supports Two-Hand Gestures** – Can capture more complex sign interactions.

#### Challenges
- ❌ **Computationally Intensive** – Requires GPU acceleration for both training and inference.  
- ❌ **Slower Real-Time Processing** – Entire images must be processed, adding latency.  
- ❌ **Background Sensitivity** – Highly dependent on consistent lighting and camera angles.

---

### Key Learnings & Final Observations
| **Feature**                   | **Keypoint-Based (Approach 1)** | **Transfer Learning (Approach 2)** |
|-------------------------------|----------------------------------|-------------------------------------|
| **Speed**                     | ✅ Fast, lightweight             | ❌ Slower, requires GPU              |
| **Accuracy**                  | ❌ Limited for complex signs      | ✅ High accuracy on diverse gestures |
| **Computational Requirements**| ✅ Low (CPU only)                | ❌ High (best with powerful GPU)     |
| **Real-Time Feasibility**     | ✅ Works seamlessly in real-time | ❌ Possible lag in inference         |
| **Handles Two-Hand Gestures** | ❌ Struggles                     | ✅ Performs well                     |

---

## Model Architectures

Below are the two main model architectures used in this project.

### 1️⃣ Keypoint-Based Model (MLP - Multi-Layer Perceptron)
- **Input:** 21 hand landmarks × 3 coordinates = 63 numerical features.
- **Hidden Layers:** 3 dense layers with **ReLU activation**.
- **Output Layer:** Softmax for multi-class classification.
- **Loss Function:** Categorical Cross-Entropy.
- **Optimizer:** Adam.
- **Learning Rate:** 0.001.

### 2️⃣ Transfer Learning Model (CNN-based Pre-Trained Network)
- **Base Model:** Pre-trained CNN (e.g., MobileNetV2, ResNet50, or EfficientNet).
- **Feature Extractor:** Convolutional layers frozen initially, then partially unfrozen for fine-tuning.
- **Classification Head:**  
  - Global Average Pooling → Dense (ReLU) → Dropout(0.5) → Dense (Softmax).
- **Loss Function:** Categorical Cross-Entropy.
- **Optimizer:** Adam.
- **Learning Rate:** 0.0001 (fine-tuned).

---

## Training Process

### Hardware & Software Environment
- **Training Platform:** Google Colab Pro  
- **GPU Used:** NVIDIA A100  
- **CPU:** Google Colab’s default vCPU  
- **RAM:** 25GB (Google Colab Pro)  
- **Frameworks:** TensorFlow, Keras, OpenCV, MediaPipe  
- **OS:** Ubuntu (Colab VM)

### Training Hyperparameters
| **Parameter**     | **Keypoint Model (MLP)** | **Transfer Learning (CNN)** |
|-------------------|--------------------------|-----------------------------|
| **Epochs**        | 100                      | 50                          |
| **Batch Size**    | 32                       | 16                          |
| **Learning Rate** | 0.001                    | 0.0001                      |
| **Optimizer**     | Adam                     | Adam                        |
| **Loss Function** | Categorical Cross-Entropy| Categorical Cross-Entropy   |
| **Training Time** | ~40 minutes             | **4 hours 30 minutes**      |

### Training Details
- **Keypoint-Based Model (MLP)**
  - Trained in **~40 minutes** using Google Colab’s default **vCPU & RAM**.
  - Faster due to the **low-dimensional numerical input** from hand landmarks.
  - Did not require extensive GPU acceleration.

- **Transfer Learning Model (CNN)**
  - **Trained for 4 hours 30 minutes** using **NVIDIA A100 GPU** on Google Colab Pro.
  - Fine-tuned **MobileNetV2, ResNet50, and EfficientNet**.
  - Required **high memory and computational power** due to image-based inputs.

### Data Augmentation
- **For Keypoint-Based Model**:
  - Random Rotations to simulate various hand orientations.
  - Scaling Variations to adjust distances between fingers.

- **For Transfer Learning Model**:
  - **Random Rotation (±30°)**
  - **Random Scaling (10–20%)**
  - **Color Jitter (brightness/contrast)**
  - **Horizontal Flipping**

---

## Evaluation and Results

### Model Performance Metrics
We evaluated both models using **Accuracy, Precision, Recall, and F1-score**.

| **Model**               | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|-------------------------|-------------:|--------------:|-----------:|-------------:|
| **Keypoint-Based (MLP)**| 82.4%        | 79.2%         | 80.6%      | 79.9%        |
| **Transfer Learning (CNN)** | **94.7%**   | **93.5%**      | **95.1%**   | **94.3%**     |

- **Confusion Matrix**  
  - The Keypoint-Based Model misclassified similar gestures due to limited visual features.  
  - The CNN-based Model performed significantly better, utilizing richer feature extraction.  

### Visualizations
- **Loss vs. Epochs:** CNN-based models demonstrated **faster convergence** and smoother validation loss.  
- **Confusion Matrix:** Transfer Learning significantly reduced misclassifications across gesture classes.
- **Sample Predictions:**  
  - ✅ **Correct:** "Hello," "Thank You," "I Love You"  
  - ❌ **Misclassified:** Complex two-hand gestures or subtle variations.

---

## Future Scope
- **Expand Gesture Vocabulary:** Recognize more ASL signs for a broader application range.  
- **Improve Model Generalization:** Test under diverse lighting, camera angles, and backgrounds.  
- **Edge Device Optimization:** Convert models to TensorFlow Lite or ONNX for real-time mobile or embedded deployment.  
- **Hybrid Approach:** Combine **keypoint tracking** with **deep feature extraction** for an optimal balance of speed and accuracy.

---

## Conclusion
Both approaches have their strengths and weaknesses. For **real-time, low-power applications**, the **keypoint-based model** is ideal due to its speed and lower computational requirements. For **high-accuracy ASL recognition**, however, the **transfer learning approach** offers significantly improved performance by leveraging deeper, richer feature representations.

A **hybrid approach**—using keypoint extraction as an initial filter and then applying a CNN-based model—could strike a balance between **efficiency** and **accuracy**, potentially extending the system’s robustness to more complex gestures and environments.

---

> **We hope this project serves as a stepping stone for further innovations in sign language recognition, making communication more accessible and inclusive for all.**
