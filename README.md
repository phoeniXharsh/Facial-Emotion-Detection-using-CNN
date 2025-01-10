---
title: Emotion Detection App
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Emotion Detection App

This **Emotion Detection App** is a machine learning-powered web application that identifies emotions in images using deep learning. Built with **Gradio** and deployed on **Hugging Face Spaces**, the app allows users to upload facial images and get predictions on the emotions they express. The model behind this app is based on the powerful **ResNet50** architecture, a pre-trained model fine-tuned for emotion classification.

## How It Works
The model predicts one of the following emotions from the uploaded images:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

### Key Features:
- The model is built using **transfer learning** with a pre-trained **ResNet50** model, which significantly speeds up training and improves accuracy.
- A custom dataset is used to fine-tune the model for facial emotion detection, with preprocessing techniques such as **data augmentation** to handle imbalances and enhance generalization.
- The app is deployed on **Hugging Face Spaces**, providing an easy-to-use web interface for interaction.

## How to Use
1. **Upload an image**: Simply upload an image of a face, and the app will predict the emotion expressed in the face.
2. **Select a sample image**: You can also choose from a list of sample images provided in the app to see different emotional expressions.

## Model Evaluation:
The **ResNet50** model has been trained and evaluated with several metrics, including:

- **Accuracy**: 62.61% training accuracy, 60.80% validation accuracy.
- **Confusion Matrix**: The predictions show good alignment with the true labels, especially for "Happy" and "Surprise."
- **AUC-ROC Curve**: The model exhibits good performance across multiple classes, with AUC scores ranging from 0.64 to 0.88 for different emotions.

## Run Locally
To run this app locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/emotion-detection-app.git
   cd emotion-detection-app

## How It Works
- The model predicts one of the following emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
- Uses a ResNet50-based pre-trained model.

## Run Locally
1. Clone the repo.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the app with `python app.py`.

## Contributing
Feel free to contribute to this project by reporting issues or submitting pull requests for improvements. If you have suggestions or ideas, please open an issue in the repository.  

## Deployed App
[Click here to access the app](https://huggingface.co/spaces/phoeniXharsh/emotion-detection)