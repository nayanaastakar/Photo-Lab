import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from image_processing import (
    apply_brightness,
    apply_contrast,
    apply_blur,
    apply_sharpness,
    apply_saturation,
    apply_rotation,
    apply_flip,
    apply_grayscale,
    apply_sepia,
    apply_invert,
    apply_vignette,
    compute_histogram,
    analyze_changes,
    pil_to_bytes,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhotoLab - Image Editor & Comparison",
    page_icon="📷",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #4f46e5, #9333ea);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd !important;
    }
    div[data-testid="stSidebar"] label {
        color: #e0e7ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">📷 PhotoLab</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload • Transform • Compare • Download</p>', unsafe_allow_html=True)

# ── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎨 Controls")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"]
    )

    st.markdown("---")

    # -- Basic Adjustments --
    st.subheader("🔆 Basic Adjustments")
    brightness = st.slider("Brightness", 0.0, 3.0, 1.0, 0.05)
    contrast = st.slider("Contrast", 0.0, 3.0, 1.0, 0.05)
    sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
    saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.1)

    st.markdown("---")

    # -- Effects --
    st.subheader("✨ Effects")
    blur_radius = st.slider("Blur Radius", 0.0, 10.0, 0.0, 0.5)
    vignette_strength = st.slider("Vignette", 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")

    # -- Filters --
    st.subheader("🎭 Filters")
    filter_mode = st.selectbox(
        "Color Filter",
        ["None", "Grayscale", "Sepia", "Invert"],
    )

    st.markdown("---")

    # -- Transform --
    st.subheader("🔄 Transform")
    rotation = st.slider("Rotation (°)", -180, 180, 0, 1)
    flip_option = st.selectbox(
        "Flip", ["None", "Horizontal", "Vertical", "Both"]
    )

    st.markdown("---")

    # -- Output --
    st.subheader("💾 Output Settings")
    output_format = st.selectbox("Download Format", ["PNG", "JPEG", "WEBP"])
    if output_format == "JPEG":
        jpeg_quality = st.slider("JPEG Quality", 10, 100, 85, 5)

    # -- Reset --
    st.markdown("---")
    if st.button("🔄 Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload an image from the sidebar to get started!")
    st.stop()

# Load image
img = Image.open(uploaded_file).convert("RGB")
original_width, original_height = img.size

# Resize for display (keep aspect ratio, max 600px wide)
max_display = 600
ratio = min(max_display / original_width, max_display / original_height, 1.0)
display_size = (int(original_width * ratio), int(original_height * ratio))
img = img.resize(display_size, Image.LANCZOS)

original_array = np.array(img) / 255.0

# ── Processing pipeline ─────────────────────────────────────────────────────
processed_array = original_array.copy()

# 1. Brightness & Contrast (numpy)
processed_array = apply_brightness(processed_array, brightness)
processed_array = apply_contrast(processed_array, contrast)

# 2. Color filter (numpy)
if filter_mode == "Grayscale":
    processed_array = apply_grayscale(processed_array)
elif filter_mode == "Sepia":
    processed_array = apply_sepia(processed_array)
elif filter_mode == "Invert":
    processed_array = apply_invert(processed_array)

# 3. Vignette (numpy)
if vignette_strength > 0:
    processed_array = apply_vignette(processed_array, vignette_strength)

# Convert to PIL for remaining ops
processed_img = Image.fromarray((processed_array * 255).astype(np.uint8))

# 4. Blur (PIL)
if blur_radius > 0:
    processed_img = apply_blur(processed_img, blur_radius)

# 5. Sharpness (PIL)
if sharpness != 1.0:
    processed_img = apply_sharpness(processed_img, sharpness)

# 6. Saturation (PIL)
if saturation != 1.0:
    processed_img = apply_saturation(processed_img, saturation)

# 7. Rotation (PIL)
if rotation != 0:
    processed_img = apply_rotation(processed_img, rotation)

# 8. Flip (PIL)
if flip_option != "None":
    processed_img = apply_flip(processed_img, flip_option)

# Final array for analysis
processed_array = np.array(processed_img) / 255.0

# ── Display ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ Original")
    st.image(original_array, clamp=True, width="stretch")

with col2:
    st.subheader("🎨 Processed")
    st.image(processed_array, clamp=True, width="stretch")

    # ── Download button ──────────────────────────────────────────────────
    fmt = output_format
    mime_map = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
    ext_map = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}

    if fmt == "JPEG":
        import io
        buf = io.BytesIO()
        processed_img.save(buf, format="JPEG", quality=jpeg_quality)
        dl_bytes = buf.getvalue()
    else:
        dl_bytes = pil_to_bytes(processed_img, fmt)

    st.download_button(
        label=f"⬇️  Download as {fmt}",
        data=dl_bytes,
        file_name=f"photolab_output.{ext_map[fmt]}",
        mime=mime_map[fmt],
        use_container_width=True,
    )

st.markdown("---")

# ── Analysis section ─────────────────────────────────────────────────────────
st.subheader("📊 Change Analysis")

avg_change, max_change, percent_changed = analyze_changes(
    original_array, processed_array
)

metric_cols = st.columns(3)
with metric_cols[0]:
    st.metric("Avg Pixel Change", f"{avg_change:.2f}%")
with metric_cols[1]:
    st.metric("Max Pixel Change", f"{max_change:.2f}%")
with metric_cols[2]:
    st.metric("Pixels Affected", f"{percent_changed:.2f}%")

# ── Transformation summary ───────────────────────────────────────────────────
st.subheader("📝 Transformation Summary")

applied = []
if brightness != 1.0:
    applied.append(f"✔ Brightness adjusted by factor **{brightness}**")
if contrast != 1.0:
    applied.append(f"✔ Contrast adjusted by factor **{contrast}**")
if sharpness != 1.0:
    applied.append(f"✔ Sharpness adjusted by factor **{sharpness}**")
if saturation != 1.0:
    applied.append(f"✔ Saturation adjusted by factor **{saturation}**")
if blur_radius > 0:
    applied.append(f"✔ Blur applied with radius **{blur_radius}**")
if vignette_strength > 0:
    applied.append(f"✔ Vignette applied at **{vignette_strength:.0%}** strength")
if filter_mode != "None":
    applied.append(f"✔ **{filter_mode}** filter applied")
if rotation != 0:
    applied.append(f"✔ Rotated by **{rotation}°**")
if flip_option != "None":
    applied.append(f"✔ Flipped **{flip_option.lower()}**")

if applied:
    for line in applied:
        st.write(line)
else:
    st.info("No transformations applied. Adjust the controls in the sidebar!")

# ── Histogram comparison ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Histogram Comparison")

hist_col1, hist_col2 = st.columns(2)

orig_hist = compute_histogram(original_array)
proc_hist = compute_histogram(processed_array)

with hist_col1:
    st.caption("Original Histogram")
    orig_df = pd.DataFrame(orig_hist, index=range(256))
    st.area_chart(orig_df, color=["#ef4444", "#22c55e", "#3b82f6"], height=200)

with hist_col2:
    st.caption("Processed Histogram")
    proc_df = pd.DataFrame(proc_hist, index=range(256))
    st.area_chart(proc_df, color=["#ef4444", "#22c55e", "#3b82f6"], height=200)

# ── Image info ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("ℹ️ Image Info")

info_cols = st.columns(4)
with info_cols[0]:
    st.metric("Width", f"{original_width} px")
with info_cols[1]:
    st.metric("Height", f"{original_height} px")
with info_cols[2]:
    st.metric("Display Size", f"{display_size[0]}×{display_size[1]}")
with info_cols[3]:
    st.metric("File Size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")
