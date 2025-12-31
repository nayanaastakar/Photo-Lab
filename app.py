import streamlit as st
import numpy as np
from PIL import Image
from image_processing import (
    apply_brightness,
    apply_contrast,
    apply_blur,
    analyze_changes
)

st.set_page_config(
    page_title="PhotoLab - Text Based Image Comparison",
    layout="wide"
)

st.title("PhotoLab")
st.write("Upload an image, apply transformations")

# Upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

# Controls
brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
blur_radius = st.slider("Blur Radius", 0.0, 5.0, 0.0, 0.5)

if uploaded_file:
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((300, 300))

    original_array = np.array(img) / 255.0

    # Apply processing
    processed_array = apply_brightness(original_array, brightness)
    processed_array = apply_contrast(processed_array, contrast)

    processed_img = Image.fromarray((processed_array * 255).astype(np.uint8))

    if blur_radius > 0:
        processed_img = apply_blur(processed_img, blur_radius)
        processed_array = np.array(processed_img) / 255.0

    # Analyze text changes
    avg_change, max_change, percent_changed = analyze_changes(
        original_array, processed_array
    )

    # UI layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before (Original)")
        st.image(original_array, clamp=True)

    with col2:
        st.subheader("After (Processed)")
        st.image(processed_array, clamp=True)

    st.markdown("---")

    # TEXT OUTPUT ONLY
    st.subheader("What Changed")

    st.write(f"🔹 **Average pixel intensity change:** {avg_change:.2f}%")
    st.write(f"🔹 **Maximum pixel change:** {max_change:.2f}%")
    st.write(f"🔹 **Pixels affected:** {percent_changed:.2f}%")

    if brightness != 1.0:
        st.write(f"✔ Brightness adjusted by factor **{brightness}**")

    if contrast != 1.0:
        st.write(f"✔ Contrast adjusted by factor **{contrast}**")

    if blur_radius > 0:
        st.write(f"✔ Blur applied with radius **{blur_radius}**")

    if brightness == 1.0 and contrast == 1.0 and blur_radius == 0:
        st.info("No transformation applied to the image.")
