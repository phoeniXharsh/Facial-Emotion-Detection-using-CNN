import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import gradio as gr
import shutil

# Path to the model file
MODEL_PATH = 'Final_Resnet50_Best_model.keras'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' is missing. Please upload the file to the repository.")

# Load the pre-trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels dictionary
emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
index_to_emotion = {v: k for k, v in emotion_labels.items()}

def prepare_image(img_pil):
    """Preprocess the PIL image to fit your model's input requirements."""
    img = img_pil.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

def predict_emotion(image):
    """Predict the emotion from the uploaded image."""
    processed_image = prepare_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = index_to_emotion.get(predicted_class[0], "Unknown Emotion")
    return predicted_emotion

# Define the Gradio interface
BASE_DIR = os.path.abspath("emotion-detection/sample-images/")
sample_images = [
    (os.path.join(BASE_DIR, "sample_img.jpg"), "Sample Image 1"),
    (os.path.join(BASE_DIR, "sample_img2.jpg"), "Sample Image 2"),
    (os.path.join(BASE_DIR, "sample_img3.jpg"), "Sample Image 3"),
    (os.path.join(BASE_DIR, "sample_img4.jpg"), "Sample Image 4"),
]
# Ensure images are copied to the Gradio cache directory
GRADIO_CACHE_DIR = "gradio_cache"
os.makedirs(GRADIO_CACHE_DIR, exist_ok=True)

cached_sample_images = []
for image_path, label in sample_images:
    cached_path = os.path.join(GRADIO_CACHE_DIR, os.path.basename(image_path))
    shutil.copy(image_path, cached_path)
    cached_sample_images.append((cached_path, label))


interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil"),
    outputs="text",
    examples=cached_sample_images,
    title="Emotion Detection",
    description=(
        "Upload/Click an image or select a sample image to detect the emotion."
    ),
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
