import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained CNN model
model_path = r"C:\imp\ml JUPYTER\MY ML PROJECTS(BOOK)\3.Deep learning\ANN\Recreation\DSP\autoencoder_model.h5"  # Update this with your model path
model = load_model(model_path)  # Load the model

def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def denoise_image(model, noisy_image):
    """Denoise the image using the trained model."""
    # Preprocess the image
    noisy_image = noisy_image / 255.0  # Normalize to [0, 1]
    noisy_image = np.expand_dims(noisy_image, axis=0)  # Add batch dimension
    denoised_image = model.predict(noisy_image)[0]  # Denoise the image
    return (denoised_image * 255).astype(np.uint8)  # Convert back to [0, 255]

# Streamlit App Layout
st.title("Image Denoising App with Gaussian Noise")
st.write("Upload an image to add Gaussian noise and denoise it using a CNN model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    original_image = Image.open(uploaded_file)
    original_image = original_image.convert("RGB")
    st.image(original_image, caption="Original Image", use_column_width=True)

    # Convert to NumPy array
    original_image_np = np.array(original_image)

    # Add Gaussian noise
    noisy_image = add_gaussian_noise(original_image_np)
    st.image(noisy_image, caption="Noisy Image", use_column_width=True)

    # Denoise the image
    denoised_image = denoise_image(model, noisy_image)  # Denoise the image

    # Display denoised image
    st.image(denoised_image, caption="Denoised Image", use_column_width=True)

    # Optional: Provide download link for denoised image
    st.download_button("Download Denoised Image", data=denoised_image, file_name="denoised_image.png", mime="image/png")

# Run the Streamlit app
# To run the app, save this code as `app.py` and execute:
# streamlit run app.py
