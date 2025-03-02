# American Sign Language Detection Model

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

By combining computer vision-based keypoint extraction and deep learning classification, we aim to deliver a robust solution for ASL gesture detection. In the subsequent sections, you’ll find details on the dataset, different modeling approaches, training procedures, and performance metrics that collectively illustrate the evolution of this project from initial data collection to a refined sign language detection pipeline.


## Project Overview

### Objectives  
The primary objectives of this project are:
- **Develop a real-time ASL recognition system** that classifies hand gestures efficiently.
- **Experiment with multiple approaches** to determine the most accurate and scalable method.
- **Leverage both custom feature extraction and transfer learning** for robust gesture classification.
- **Contribute to assistive technology** by providing a foundation for ASL recognition applications.

### Methodologies  
We explored **two primary approaches** to solving this problem:

#### **1️⃣ Keypoint-Based Recognition (Custom Model with MediaPipe)**
📂 **Directory:** `hand_landmark_model/`  
- Utilizes **Google’s MediaPipe** to extract hand landmarks from video frames.  
- Features a **custom neural network** trained on the extracted keypoints.  
- Efficient and lightweight, focusing on the positional relationship between fingers.  
- Best suited for real-time applications where speed is a priority.  
- *Notebook Reference:* `HandGestureDataExtraction(MediaPipe).ipynb` and `SignLanguageDetectionModel.ipynb`

#### **2️⃣ Transfer Learning on Pre-Trained Models**
📂 **Directory:** `transfer_learning_model/`  
- Leverages pre-trained deep learning models (e.g., **MobileNet, ResNet, EfficientNet**) to extract rich visual features.  
- Works directly with **raw images/videos** instead of relying on keypoints.  
- More computationally intensive but **achieves higher accuracy** in classification.  
- Best suited for offline processing or high-performance applications.  
- *Notebook Reference:* `TransferLearningModel.ipynb`

### System Pipeline  
Both approaches follow a structured pipeline for ASL recognition:

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

## Approaches

In developing our ASL detection model, we experimented with **two distinct methodologies** to recognize hand gestures. Each approach had unique strengths and trade-offs, which we carefully evaluated through experimentation.

---

### **Approach 1: Keypoint-Based Gesture Recognition (Custom Model with MediaPipe)**

#### **Methodology**  
This approach utilized **Google's MediaPipe library** to extract **21 key hand landmarks** (x, y, z coordinates) from images and video frames. Instead of feeding raw images into a deep learning model, we trained a custom **lightweight neural network** on these extracted keypoints to classify hand gestures. The pipeline involved:

1. **Data Collection & Preprocessing:**
   - Used **MediaPipe Hands** to extract **x, y, z** coordinates for each of the 21 hand landmarks.
   - Stored keypoint data as structured numerical datasets for efficient training.
   - Augmented data using random transformations to improve generalization.

2. **Model Training:**
   - Built a **fully connected neural network (MLP - Multi-Layer Perceptron)** using TensorFlow/Keras.
   - Tuned hyperparameters (learning rate, dropout, activation functions).
   - Trained on extracted keypoint datasets instead of raw images.

3. **Inference & Real-Time Detection:**
   - MediaPipe continuously extracted keypoints from a webcam feed.
   - The trained model classified the gesture based on the real-time keypoint data.
   - Displayed the detected ASL sign with a confidence score.

#### **Reasoning**  
- **Efficiency:** Since neural networks process numerical inputs much faster than images, this method was expected to be **lightweight** and suitable for **real-time applications**.
- **Feature Extraction Control:** By focusing on hand landmarks instead of raw pixels, we aimed to reduce the influence of background noise and lighting conditions.

#### **Advantages**
✔️ **Fast & Efficient** – Processes only **numerical keypoints**, making it computationally lightweight.  
✔️ **Real-time Ready** – Works well on lower-end hardware, including mobile devices.  
✔️ **Robust to Background Variations** – Since it uses hand landmarks instead of pixel intensities, it is less affected by lighting or background clutter.  

#### **Challenges**
❌ **Loss of Contextual Information** – The model only considers hand keypoints and ignores additional visual cues such as motion, orientation, or hand texture.  
❌ **Limited Accuracy for Complex Gestures** – Some ASL signs require **subtle finger movements or two-hand interactions**, which were harder to capture using keypoints alone.  
❌ **Struggled with Rotation & Perspective Variability** – The 3D depth estimation from MediaPipe wasn't always accurate, leading to misclassification in certain angles.  

**Why We Moved to the Second Approach?**  
While this method was promising, it had limitations in recognizing more intricate ASL signs. To improve accuracy, we decided to experiment with a **deep learning-based image classification approach using transfer learning.**

---

### **Approach 2: Transfer Learning on Pre-Trained Deep Learning Models**  

#### **Methodology**  
This approach took a **traditional deep learning route** by using **pre-trained convolutional neural networks (CNNs)** to classify ASL gestures directly from **raw images** rather than relying on extracted keypoints. The steps involved:

1. **Data Preparation:**
   - Used a dataset of labeled ASL gesture images.
   - Applied **data augmentation** (rotation, flipping, contrast changes) to improve generalization.

2. **Feature Extraction with Transfer Learning:**
   - Experimented with **MobileNetV2, ResNet50, and EfficientNet**, which were pre-trained on **ImageNet**.
   - Removed the final classification layer and replaced it with a **custom dense layer** for ASL sign recognition.

3. **Fine-Tuning & Training:**
   - First trained only the custom classification head.
   - Then fine-tuned some of the deeper CNN layers to adapt them for ASL recognition.
   - Used a **categorical cross-entropy loss** and **Adam optimizer** for model convergence.

4. **Inference & Deployment:**
   - Captured real-time webcam frames.
   - Preprocessed the frames (resized and normalized).
   - Fed them into the trained CNN model to predict ASL gestures.

#### **Reasoning**  
- **Higher Accuracy Potential**: CNNs learn richer feature representations directly from images, which allows them to distinguish even subtle variations in ASL gestures.  
- **Handles Complex Gestures Better**: Unlike keypoints, **full images provide spatial and textural information**, improving recognition of similar-looking gestures.  
- **Leverages Existing Research**: Instead of training from scratch, we used transfer learning from state-of-the-art **pre-trained models**, reducing computation costs.

#### **Advantages**
✔️ **Higher Accuracy** – Achieved significantly better performance on complex ASL gestures.  
✔️ **Better Generalization** – The model learned **robust** visual features that adapted well to different backgrounds and lighting conditions.  
✔️ **Supports Two-Hand Gestures** – Unlike the keypoint-based approach, this model accurately recognized gestures requiring both hands.

#### **Challenges**
❌ **Computationally Intensive** – Requires **GPU acceleration** for both training and inference.  
❌ **Slower Real-Time Processing** – Since entire images are processed instead of just keypoints, it introduces some delay.  
❌ **Dependent on Background Consistency** – The model can be affected by changes in **lighting, camera angles, or background objects**.

---

### **Key Learnings & Final Observations**
| Feature | Keypoint-Based (Approach 1) | Transfer Learning (Approach 2) |
|---------|----------------------------|------------------------------|
| **Speed** | ✅ Very fast, lightweight | ❌ Slower, requires GPU |
| **Accuracy** | ❌ Limited for complex signs | ✅ High accuracy on diverse gestures |
| **Computational Requirements** | ✅ Low (works on CPU) | ❌ High (best on GPU) |
| **Real-Time Feasibility** | ✅ Works in real-time | ❌ Slight lag in inference |
| **Handles Two-Hand Gestures** | ❌ Struggles with complex signs | ✅ Works well |

### **Conclusion**
Both approaches have their strengths and weaknesses. For **real-time, low-power applications**, the **keypoint-based model** is ideal due to its efficiency. However, for **high-accuracy ASL recognition**, the **transfer learning approach** outperforms the keypoint model significantly.  

A potential **hybrid approach**, combining keypoint extraction with deep learning-based feature extraction, may offer the best of both worlds in the future.

## Model Architectures
- Detailed explanation of the final model(s) used (e.g., CNN, LSTM, Transformers).
- Key hyperparameters (layers, activation functions, learning rates).

## Training Process
- Hardware and software environment.
- Epochs, batch size, training time.
- Any data augmentation or regularization steps used.

## Evaluation and Results
- Metrics (accuracy, F1-score, confusion matrix).
- Comparison of the two approaches if relevant.
- Example outputs or visualizations.

## Usage
- Instructions to install dependencies, set up environment, run the notebooks or scripts.
- Example command lines.

## Future Work
- Plans for expanding, refining, or deploying the system.
- Possible improvements or additional features.

## Contributors
- List or mention everyone who contributed significantly, along with their roles or specific contributions.

## License
- State your chosen license (e.g., MIT, Apache 2.0) and link to the `LICENSE` file if present.

