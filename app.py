import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained CNN model
model_path = r"autoencoder_model.h5"  # Update this with your model path
model = load_model(model_path)  # Load the model

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image."""
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)  # Add the noise to the original image
    return noisy_image

def denoise_image(model, noisy_image):
    """Denoise the image using the trained model."""
    # Resize to 28x28
    noisy_image_resized = cv2.resize(noisy_image, (28, 28))
    noisy_image_resized = cv2.cvtColor(noisy_image_resized, cv2.COLOR_RGB2GRAY)
    noisy_image_resized = np.expand_dims(noisy_image_resized, axis=-1)

    # Preprocess
    noisy_image_resized = noisy_image_resized / 255.0
    noisy_image_resized = np.expand_dims(noisy_image_resized, axis=0)  # Add batch dimension

    # Denoise
    denoised_image = model.predict(noisy_image_resized)[0]
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


