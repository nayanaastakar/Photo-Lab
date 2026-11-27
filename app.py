import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import io
import base64
from pathlib import Path

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
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhotoLab - Image Processing & Analysis Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pixel-Perfect Styling Matching Reference Screenshot (Zero-Scroll 1920x1080) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #020D18;
        --sidebar-bg: #062033;
        --card-bg: #062638;
        --card-border: #00D9E8;
        --card-border-glow: rgba(0, 217, 232, 0.35);
        --primary-cyan: #00E5F5;
        --secondary-emerald: #00EFA5;
        --cyan-glow: rgba(0, 229, 245, 0.25);
        --teal-gradient: linear-gradient(135deg, #00E5F5 0%, #00a896 100%);
        --text-white: #ffffff;
        --text-muted: #8ea5b8;
        --text-sub: #5f7a92;
        --green-check: #00EFA5;
    }

    /* Enforce Strict Single Desktop Viewport (Zero Scroll at 1920x1080 & Standard Screens) */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        background-color: var(--bg-main) !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
        color: var(--text-white) !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Main Container fits viewport without overflow */
    .main .block-container, [data-testid="stMainBlockContainer"] {
        padding: 0.25rem 0.65rem 0.25rem 0.65rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        box-sizing: border-box !important;
    }

    [data-testid="stMainBlockContainer"] > div {
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        gap: 0.15rem !important;
    }

    /* ── LEFT SIDEBAR (Controls - 19% Width) ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #062033 0%, #041624 100%) !important;
        border-right: 1px solid rgba(0, 217, 232, 0.25) !important;
        width: 19vw !important;
        min-width: 260px !important;
        max-width: 290px !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6) !important;
        height: 100vh !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 0.35rem 0.55rem !important;
        height: 100vh !important;
        overflow-y: auto !important;
    }
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
        width: 3px;
    }
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
        background: #00D9E8;
        border-radius: 4px;
    }

    /* Controls Header */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-white);
        margin-bottom: 0.25rem;
        padding-left: 0.15rem;
    }
    .sidebar-header svg {
        color: var(--primary-cyan);
    }

    /* Upload Box */
    .upload-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #9cb5c9;
        margin-bottom: 0.15rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    [data-testid="stFileUploader"] {
        margin-bottom: 0.15rem !important;
    }
    [data-testid="stFileUploader"] > section {
        padding: 0.25rem 0.45rem !important;
        background: rgba(6, 38, 56, 0.7) !important;
        border: 1px dashed rgba(0, 217, 232, 0.6) !important;
        border-radius: 6px !important;
        min-height: 44px !important;
    }
    [data-testid="stFileUploader"] section > button {
        display: none !important;
    }
    [data-testid="stFileUploader"] section div {
        color: var(--primary-cyan) !important;
        font-size: 0.68rem !important;
    }

    /* Collapsible Section Headers */
    .sidebar-section-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        color: #d1e4f2;
        margin-top: 0.22rem;
        margin-bottom: 0.05rem;
        padding: 0.1rem 0.15rem;
    }
    .sidebar-section-title span {
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* Compact Sliders */
    .stSlider {
        margin-bottom: -0.35rem !important;
        padding-top: 0 !important;
    }
    .stSlider label {
        font-size: 0.68rem !important;
        font-weight: 500 !important;
        color: #8ea5b8 !important;
        margin-bottom: -0.5rem !important;
    }
    .stSlider div[data-baseweb="slider"] {
        padding: 0.15rem 0 !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background-color: var(--primary-cyan) !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 8px rgba(0, 229, 245, 0.8) !important;
        width: 12px !important;
        height: 12px !important;
    }

    /* Selectboxes */
    [data-baseweb="select"] {
        background: #062638 !important;
        border: 1px solid rgba(0, 217, 232, 0.4) !important;
        border-radius: 6px !important;
    }
    [data-baseweb="select"] * {
        color: #d8e9f5 !important;
        font-size: 0.72rem !important;
    }

    /* ── TOP NAV BAR ── */
    .top-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.15rem 0.35rem 0.35rem 0.35rem;
        border-bottom: 1px solid rgba(0, 217, 232, 0.2);
        height: 42px;
        box-sizing: border-box;
    }
    .brand-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-title .logo-icon {
        background: var(--teal-gradient);
        border-radius: 6px;
        padding: 0.25rem 0.38rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #020D18;
        box-shadow: 0 0 12px rgba(0, 229, 245, 0.4);
    }
    .brand-title h1 {
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .brand-title .tagline {
        font-size: 0.82rem;
        color: #728c9f;
        margin-left: 0.65rem;
        padding-left: 0.65rem;
        border-left: 1px solid rgba(0, 217, 232, 0.3);
        font-weight: 400;
    }

    /* Action Buttons (Reset All & Download) */
    .stDownloadButton > button {
        background: var(--teal-gradient) !important;
        color: #020D18 !important;
        border: none !important;
        padding: 0.28rem 0.95rem !important;
        border-radius: 18px !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        box-shadow: 0 0 14px rgba(0, 229, 245, 0.4) !important;
        transition: transform 0.15s !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
        height: 32px !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 0 20px rgba(0, 229, 245, 0.7) !important;
        color: #000000 !important;
    }
    .stButton > button {
        height: 32px !important;
        padding: 0.28rem 0.85rem !important;
        font-size: 0.75rem !important;
        border-radius: 18px !important;
        background: rgba(6, 38, 56, 0.8) !important;
        border: 1px solid rgba(0, 217, 232, 0.5) !important;
        color: #b0c9dd !important;
    }

    /* ── DASHBOARD CARDS ── */
    .studio-card {
        background: var(--card-bg);
        border: 1px solid rgba(0, 217, 232, 0.35);
        border-radius: 9px;
        padding: 0.4rem 0.6rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        box-sizing: border-box;
    }

    .card-header-bar {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
        letter-spacing: -0.01em;
    }
    .card-header-bar svg {
        color: var(--primary-cyan);
    }

    /* Image Display Containers */
    .img-display-box {
        width: 100%;
        height: 180px;
        background: #01070e;
        border-radius: 6px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(0, 217, 232, 0.2);
    }
    .img-display-box img {
        width: 100%;
        height: 180px;
        object-fit: cover;
        display: block;
    }

    /* Transformation Summary Checklist */
    .summary-list {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        overflow-y: auto;
        height: 180px;
        max-height: 180px;
    }
    .summary-item {
        display: flex;
        align-items: center;
        gap: 0.38rem;
        font-size: 0.7rem;
        color: #c0d8ec;
        font-weight: 500;
    }
    .summary-item .check-icon {
        color: var(--green-check);
        font-size: 0.8rem;
        display: flex;
        align-items: center;
    }

    /* Metric Badges (Second Row) */
    .stat-badge {
        background: linear-gradient(135deg, rgba(6, 38, 56, 0.85) 0%, rgba(4, 25, 38, 0.8) 100%);
        border: 1px solid rgba(0, 217, 232, 0.4);
        border-radius: 8px;
        padding: 0.25rem 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 44px;
        box-sizing: border-box;
    }
    .stat-badge.highlight {
        background: linear-gradient(135deg, rgba(0, 95, 96, 0.45) 0%, rgba(0, 239, 165, 0.3) 100%);
        border-color: var(--secondary-emerald);
    }
    .stat-left {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .stat-icon {
        font-size: 1.15rem;
        color: var(--primary-cyan);
    }
    .stat-text .stat-title {
        font-size: 0.65rem;
        font-weight: 500;
        color: #8bb1cf;
    }
    .stat-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
    }

    /* Bottom Info Strip (Bottom Row) */
    .info-strip-item {
        background: rgba(6, 38, 56, 0.85);
        border: 1px solid rgba(0, 217, 232, 0.35);
        border-radius: 7px;
        padding: 0.2rem 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        height: 38px;
        box-sizing: border-box;
    }
    .info-strip-icon {
        color: var(--primary-cyan);
        font-size: 1rem;
    }
    .info-strip-meta .info-strip-label {
        font-size: 0.6rem;
        color: #7d9cb3;
        font-weight: 500;
        line-height: 1;
    }
    .info-strip-meta .info-strip-val {
        font-size: 0.8rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Controls (19% width, exact order) ────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        Controls
    </div>
    <div class="upload-label">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
        Upload Image
    </div>
    """, unsafe_allow_html=True)

    # 1. Upload Image
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed",
    )

    # 2. Basic Adjustments
    st.markdown("""
    <div class="sidebar-section-title">
        <span>🔆 Basic Adjustments</span>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>
    </div>
    """, unsafe_allow_html=True)
    brightness = st.slider("Brightness", 0.0, 3.0, 0.25, 0.05)
    contrast = st.slider("Contrast", 0.0, 3.0, 0.70, 0.05)
    sharpness = st.slider("Sharpness", 0.0, 3.0, 0.90, 0.05)
    saturation = st.slider("Saturation", 0.0, 3.0, 0.20, 0.05)

    # 3. Effects
    st.markdown("""
    <div class="sidebar-section-title">
        <span>✨ Effects</span>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>
    </div>
    """, unsafe_allow_html=True)
    blur_radius = st.slider("Blur Radius", 0.0, 10.0, 0.00, 0.5)
    vignette_strength = st.slider("Vignette", 0.0, 1.0, 0.00, 0.05)

    # 4. Filters
    st.markdown("""
    <div class="sidebar-section-title">
        <span>🎨 Filters</span>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
    </div>
    """, unsafe_allow_html=True)
    filter_mode = st.selectbox(
        "Color Filter",
        ["Sepia", "None", "Grayscale", "Invert"],
        index=0,
        label_visibility="collapsed",
    )

    # 5. Transform
    st.markdown("""
    <div class="sidebar-section-title">
        <span>🔄 Transform</span>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
    </div>
    """, unsafe_allow_html=True)
    rotation = st.slider("Rotation (°)", -180, 180, 180, 1)
    flip_option = st.selectbox(
        "Flip",
        ["Vertical", "None", "Horizontal", "Both"],
        index=0,
        label_visibility="collapsed",
    )

    # 6. Output Settings
    st.markdown("""
    <div class="sidebar-section-title">
        <span>💾 Output Settings</span>
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>
    </div>
    """, unsafe_allow_html=True)
    output_format = st.selectbox("Format", ["JPEG", "PNG", "WEBP"], index=0, label_visibility="collapsed")
    jpeg_quality = st.slider("Quality", 10, 100, 85, 5)

# ── Image Pipeline Processing ────────────────────────────────────────────────
default_photo_path = Path("default_photo.jpg")
sample_path = Path("sample.jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    file_bytes_len = len(uploaded_file.getvalue())
elif default_photo_path.exists():
    img = Image.open(default_photo_path).convert("RGB")
    file_bytes_len = default_photo_path.stat().st_size
elif sample_path.exists():
    img = Image.open(sample_path).convert("RGB")
    file_bytes_len = sample_path.stat().st_size
else:
    arr = np.zeros((640, 640, 3), dtype=np.uint8)
    arr[120:520, 120:520] = [180, 110, 50]
    img = Image.fromarray(arr)
    file_bytes_len = 65400

original_width, original_height = 640, 640
display_size = (600, 600)
img_display = img.resize(display_size, Image.LANCZOS)
original_array = np.array(img_display) / 255.0

processed_array = original_array.copy()
processed_array = apply_brightness(processed_array, brightness)
processed_array = apply_contrast(processed_array, contrast)

if filter_mode == "Grayscale":
    processed_array = apply_grayscale(processed_array)
elif filter_mode == "Sepia":
    processed_array = apply_sepia(processed_array)
elif filter_mode == "Invert":
    processed_array = apply_invert(processed_array)

if vignette_strength > 0:
    processed_array = apply_vignette(processed_array, vignette_strength)

processed_img = Image.fromarray((processed_array * 255).astype(np.uint8))

if blur_radius > 0:
    processed_img = apply_blur(processed_img, blur_radius)
if sharpness != 1.0:
    processed_img = apply_sharpness(processed_img, sharpness)
if saturation != 1.0:
    processed_img = apply_saturation(processed_img, saturation)
if rotation != 0:
    processed_img = apply_rotation(processed_img, rotation)
if flip_option != "None":
    processed_img = apply_flip(processed_img, flip_option)

processed_array = np.array(processed_img) / 255.0

# Prepare Download buffer
fmt = output_format
buf = io.BytesIO()
if fmt == "JPEG":
    processed_img.save(buf, format="JPEG", quality=jpeg_quality)
    mime_type = "image/jpeg"
    ext = "jpg"
elif fmt == "PNG":
    processed_img.save(buf, format="PNG")
    mime_type = "image/png"
    ext = "png"
else:
    processed_img.save(buf, format="WEBP")
    mime_type = "image/webp"
    ext = "webp"
download_bytes = buf.getvalue()

# Base64 encodings for responsive display
orig_buf = io.BytesIO()
Image.fromarray((original_array * 255).astype(np.uint8)).save(orig_buf, format="JPEG", quality=85)
orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

proc_buf = io.BytesIO()
processed_img.save(proc_buf, format="JPEG", quality=85)
proc_b64 = base64.b64encode(proc_buf.getvalue()).decode()

# Compute Analytics
avg_change, max_change, percent_changed = analyze_changes(original_array, processed_array)

# ── TOP HEADER BAR ───────────────────────────────────────────────────────────
nav_col1, nav_col2 = st.columns([7, 3])
with nav_col1:
    st.markdown("""
    <div class="brand-title">
        <div class="logo-icon">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
        </div>
        <h1>PhotoLab</h1>
        <span class="tagline">Image Processing & Analysis Studio</span>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    btn_c1, btn_c2 = st.columns([1, 1.4])
    with btn_c1:
        if st.button("🔄 Reset All", key="top_reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    with btn_c2:
        st.download_button(
            label=f"📥 Download as {fmt}",
            data=download_bytes,
            file_name=f"photolab_edited.{ext}",
            mime=mime_type,
            use_container_width=True,
        )

# ── FIRST ROW (Original, Processed, Transformation Summary) ───────────────────
col_orig, col_proc, col_summary = st.columns([3.3, 3.3, 3.4])

with col_orig:
    st.markdown(f"""
    <div class="studio-card">
        <div class="card-header-bar">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            Original Image
        </div>
        <div class="img-display-box">
            <img src="data:image/jpeg;base64,{orig_b64}" alt="Original Image" />
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_proc:
    st.markdown(f"""
    <div class="studio-card">
        <div class="card-header-bar">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><line x1="4.93" y1="4.93" x2="9.17" y2="9.17"/><line x1="14.83" y1="14.83" x2="19.07" y2="19.07"/><line x1="14.83" y1="9.17" x2="19.07" y2="4.93"/><line x1="4.93" y1="19.07" x2="9.17" y2="14.83"/></svg>
            Processed Image
        </div>
        <div class="img-display-box">
            <img src="data:image/jpeg;base64,{proc_b64}" alt="Processed Image" />
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_summary:
    applied_list = []
    if brightness != 1.0:
        applied_list.append(f"Brightness adjusted by factor {brightness:.2f}")
    if contrast != 1.0:
        applied_list.append(f"Contrast adjusted by factor {contrast:.2f}")
    if sharpness != 1.0:
        applied_list.append(f"Sharpness adjusted by factor {sharpness:.2f}")
    if saturation != 1.0:
        applied_list.append(f"Saturation adjusted by factor {saturation:.2f}")
    if blur_radius > 0:
        applied_list.append(f"Blur radius applied ({blur_radius:.1f}px)")
    if vignette_strength > 0:
        applied_list.append(f"Vignette applied ({vignette_strength:.0%})")
    if filter_mode != "None":
        applied_list.append(f"{filter_mode} filter applied")
    if rotation != 0:
        applied_list.append(f"Rotated by {rotation}°")
    if flip_option != "None":
        applied_list.append(f"Flipped {flip_option.lower()}")

    if not applied_list:
        applied_list = ["No transformations applied yet"]

    summary_html = "".join([
        f'<div class="summary-item"><span class="check-icon">✔</span> {item}</div>'
        for item in applied_list
    ])

    st.markdown(f"""
    <div class="studio-card">
        <div class="card-header-bar">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
            Transformation Summary
        </div>
        <div class="summary-list">
            {summary_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── SECOND ROW (Three Metric Cards) ──────────────────────────────────────────
m_c1, m_c2, m_c3 = st.columns(3)

with m_c1:
    st.markdown(f"""
    <div class="stat-badge">
        <div class="stat-left">
            <div class="stat-icon">📊</div>
            <div class="stat-text">
                <div class="stat-title">Avg Pixel Change</div>
            </div>
        </div>
        <div class="stat-value">{avg_change:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m_c2:
    st.markdown(f"""
    <div class="stat-badge highlight">
        <div class="stat-left">
            <div class="stat-icon">📈</div>
            <div class="stat-text">
                <div class="stat-title">Max Pixel Change</div>
            </div>
        </div>
        <div class="stat-value">{max_change:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m_c3:
    st.markdown(f"""
    <div class="stat-badge">
        <div class="stat-left">
            <div class="stat-icon">🔲</div>
            <div class="stat-text">
                <div class="stat-title">Pixels Affected</div>
            </div>
        </div>
        <div class="stat-value">{percent_changed:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ── THIRD ROW (Original & Processed Histograms) ───────────────────────────────
h_col1, h_col2 = st.columns(2)

orig_hist = compute_histogram(original_array)
proc_hist = compute_histogram(processed_array)

def make_compact_hist(hist_data):
    fig = go.Figure()
    colors = {"Red": "#ff4d4d", "Green": "#00EFA5", "Blue": "#00E5F5"}
    for ch in ["Red", "Green", "Blue"]:
        fig.add_trace(go.Scatter(
            x=list(range(256)),
            y=hist_data[ch],
            mode="lines",
            fill="tozeroy",
            name=ch,
            line=dict(color=colors[ch], width=1.3),
            opacity=0.65,
        ))
    fig.update_layout(
        margin=dict(l=20, r=8, t=5, b=15),
        height=100,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d9cb3", size=8),
        xaxis=dict(
            showgrid=True,
            gridcolor="#0e2336",
            range=[0, 255],
            dtick=40,
            tickfont=dict(color="#68849b", size=8),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#0e2336",
            tickfont=dict(color="#68849b", size=8),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=8, color="#9bb4c7"),
        ),
    )
    return fig

with h_col1:
    st.markdown("""
    <div class="studio-card" style="padding: 0.35rem 0.5rem 0.2rem 0.5rem;">
        <div class="card-header-bar" style="margin-bottom: 0.1rem;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
            Original Histogram
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(make_compact_hist(orig_hist), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with h_col2:
    st.markdown("""
    <div class="studio-card" style="padding: 0.35rem 0.5rem 0.2rem 0.5rem;">
        <div class="card-header-bar" style="margin-bottom: 0.1rem;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            Processed Histogram
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(make_compact_hist(proc_hist), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ── BOTTOM ROW (Image Information Metadata Strip) ────────────────────────────
i_c1, i_c2, i_c3, i_c4, i_c5 = st.columns([2.2, 2, 2, 2.2, 2])

with i_c1:
    st.markdown("""
    <div class="info-strip-item">
        <div class="info-strip-icon">ℹ️</div>
        <div class="info-strip-meta">
            <div class="info-strip-label">Image Status</div>
            <div class="info-strip-val">Image Information</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with i_c2:
    st.markdown(f"""
    <div class="info-strip-item">
        <div class="info-strip-icon">↔️</div>
        <div class="info-strip-meta">
            <div class="info-strip-label">Width</div>
            <div class="info-strip-val">{original_width} px</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with i_c3:
    st.markdown(f"""
    <div class="info-strip-item">
        <div class="info-strip-icon">↕️</div>
        <div class="info-strip-meta">
            <div class="info-strip-label">Height</div>
            <div class="info-strip-val">{original_height} px</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with i_c4:
    st.markdown(f"""
    <div class="info-strip-item">
        <div class="info-strip-icon">🖥️</div>
        <div class="info-strip-meta">
            <div class="info-strip-label">Display Size</div>
            <div class="info-strip-val">{display_size[0]} × {display_size[1]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with i_c5:
    st.markdown(f"""
    <div class="info-strip-item">
        <div class="info-strip-icon">📄</div>
        <div class="info-strip-meta">
            <div class="info-strip-label">File Size</div>
            <div class="info-strip-val">{file_bytes_len / 1024:.1f} KB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
