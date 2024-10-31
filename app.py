import streamlit as st
import numpy as np
import cv2
from PIL import Image

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image."""
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def denoise_image(noisy_image):
    """Denoise the image using enhanced Median and Bilateral filters."""
    # Apply median filtering with a larger kernel size
    median_filtered = cv2.medianBlur(noisy_image, 1)  # Increase kernel size to 7

    # Apply bilateral filtering with adjusted parameters
    bilateral_filtered = cv2.bilateralFilter(median_filtered, d=1, sigmaColor=150, sigmaSpace=150)

    return bilateral_filtered

# Streamlit App Layout
st.title("Image Denoising App with Improved Filtering Techniques")
st.write("Upload an image to add Gaussian noise and denoise it using enhanced filtering techniques.")

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
    denoised_image = denoise_image(noisy_image)  # Denoise the image

    # Display denoised image
    st.image(denoised_image, caption="Denoised Image", use_column_width=True)

    # Optional: Provide download link for denoised image
    st.download_button("Download Denoised Image", data=denoised_image.tobytes(), file_name="denoised_image.png", mime="image/png")

# Run the Streamlit app
# To run the app, save this code as `app.py` and execute:
# streamlit run app.py
