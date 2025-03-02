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
- *Notebook Reference:* `HandGestureDataExtraction(MediaPipe).ipynb`

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
Here you’ll detail the **two main ways** you tried to solve the problem.

### Approach 1
- **Methodology**: Briefly describe the overall pipeline (e.g., using a custom model, a particular library, or a specific feature-extraction technique).
- **Reasoning**: Why you chose this approach initially.
- **Advantages**: What made it appealing.
- **Challenges**: What you struggled with or why you moved on to a second approach.

### Approach 2
- **Methodology**: How it differs from Approach 1 (e.g., using transfer learning, a different data pipeline, or a more advanced architecture).
- **Reasoning**: Why you pivoted or experimented with this new approach.
- **Advantages** and **Challenges**: Summarize learnings.

## Data Collection and Preprocessing
- Where you sourced data (or how you recorded it).
- Steps taken to clean or organize it.
- Tools/libraries used for data preprocessing (e.g., MediaPipe, OpenCV, etc.).

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

