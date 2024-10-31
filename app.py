import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

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
    # Resize to 28x28 for model input
    noisy_image_resized = cv2.resize(noisy_image, (28, 28))
    
    # Preprocess the image
    noisy_image_resized = cv2.cvtColor(noisy_image_resized, cv2.COLOR_RGB2GRAY)
    noisy_image_resized = np.expand_dims(noisy_image_resized, axis=-1)  # Add channel dimension
    noisy_image_resized = noisy_image_resized / 255.0  # Normalize
    noisy_image_resized = np.expand_dims(noisy_image_resized, axis=0)  # Add batch dimension

    # Denoise the image
    denoised_image = model.predict(noisy_image_resized)[0]

    # Convert back to [0, 255] and ensure correct shape
    denoised_image = (denoised_image * 255).astype(np.uint8)
    
    # If your model is outputting a single channel, convert it to RGB
    if denoised_image.shape[-1] == 1:  # Grayscale output
        denoised_image = cv2.cvtColor(denoised_image.reshape(28, 28), cv2.COLOR_GRAY2RGB)
    else:  # Already in RGB format
        denoised_image = denoised_image.reshape(28, 28, 3)  # Ensure it's reshaped to (height, width, channels)

    # Resize back to original dimensions
    original_height, original_width = noisy_image.shape[:2]
    denoised_image = cv2.resize(denoised_image, (original_width, original_height))
    
    return denoised_image


def convert_to_image_bytes(image_np):
    """Convert NumPy array to PNG byte stream."""
    if len(image_np.shape) == 2:  # Grayscale image
        image_np = np.expand_dims(image_np, axis=-1)  # Add channel dimension if needed
    elif len(image_np.shape) == 3 and image_np.shape[2] == 1:  # Single channel
        image_np = np.squeeze(image_np)  # Remove single channel dimension
        
    image_pil = Image.fromarray(image_np)  # Convert NumPy array to PIL Image
    byte_io = io.BytesIO()  # Create a byte stream
    image_pil.save(byte_io, format='PNG')  # Save the image to the byte stream
    byte_io.seek(0)  # Seek to the start of the stream
    return byte_io.getvalue()  # Return the byte data

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

    # Convert denoised image to byte stream for download
    denoised_image_bytes = convert_to_image_bytes(denoised_image)

    # Optional: Provide download link for denoised image
    st.download_button("Download Denoised Image", data=denoised_image_bytes, file_name="denoised_image.png", mime="image/png")

# Run the Streamlit app
# To run the app, save this code as `app.py` and execute:
# streamlit run app.py
