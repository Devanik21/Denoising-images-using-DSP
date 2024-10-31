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
    """Denoise the image using Non-Local Means Denoising."""
    # Convert the image to YUV color space
    yuv_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2YUV)
    
    # Denoise the Y channel
    yuv_image[:, :, 0] = cv2.fastNlMeansDenoising(yuv_image[:, :, 0], None, h=30, templateWindowSize=7, searchWindowSize=21)
    
    # Convert back to RGB color space
    denoised_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    return denoised_image

# Streamlit App Layout
st.title("Image Denoising App with Advanced Techniques")
st.write("Upload an image to add Gaussian noise and denoise it using advanced techniques.")

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
