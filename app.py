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

# ── Fully Responsive + Midnight Navy & Electric Cyan Styling ──────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #020D18;
        --sidebar-bg: #041829;
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

    /* ── BASE & DESKTOP ZERO-SCROLL VIEWPORT ── */
    html, body, .stApp {
        background-color: var(--bg-main) !important;
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
        color: var(--text-white) !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Hide Streamlit default header decoration */
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        display: none !important;
    }

    /* ── RESPONSIVE DESKTOP (>= 1025px): STRICT ZERO SCROLL ── */
    @media (min-width: 1025px) {
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }

        .main .block-container, [data-testid="stMainBlockContainer"] {
            padding: 0.6rem 0.9rem 0.6rem 0.9rem !important;
            max-width: 100% !important;
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            gap: 0.55rem !important;
            box-sizing: border-box !important;
        }

        [data-testid="stMainBlockContainer"] > div {
            height: auto !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            gap: 0.55rem !important;
        }

        [data-testid="stSidebar"] {
            background: #031422 !important;
            border-right: 1px solid rgba(0, 217, 232, 0.25) !important;
            width: 19vw !important;
            min-width: 260px !important;
            max-width: 290px !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6) !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 0.35rem 0.55rem 0.45rem 0.55rem !important;
            height: 100vh !important;
            overflow-y: hidden !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            box-sizing: border-box !important;
        }

        .img-display-box {
            height: 225px !important;
        }
        .img-display-box img {
            height: 225px !important;
        }
    }

    /* ── RESPONSIVE LAPTOP / TABLET LANDSCAPE (769px to 1024px) ── */
    @media (max-width: 1024px) and (min-width: 769px) {
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            height: auto !important;
            overflow-y: auto !important;
        }
        .main .block-container, [data-testid="stMainBlockContainer"] {
            padding: 0.75rem 1rem !important;
            height: auto !important;
            overflow: visible !important;
            gap: 0.65rem !important;
        }
        [data-testid="stSidebar"] {
            width: 280px !important;
            height: 100vh !important;
            overflow-y: auto !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            height: auto !important;
            overflow-y: auto !important;
        }
        .img-display-box {
            height: 180px !important;
        }
        .img-display-box img {
            height: 180px !important;
        }
        .brand-title h1 {
            font-size: 1.25rem !important;
        }
        .brand-title .tagline {
            display: none !important;
        }
    }

    /* ── RESPONSIVE MOBILE / PORTRAIT TABLET (<= 768px) ── */
    @media (max-width: 768px) {
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            height: auto !important;
            overflow-y: auto !important;
        }
        .main .block-container, [data-testid="stMainBlockContainer"] {
            padding: 0.5rem !important;
            height: auto !important;
            overflow: visible !important;
            gap: 0.75rem !important;
        }
        [data-testid="stSidebar"] {
            width: 100% !important;
            height: auto !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            height: auto !important;
            overflow-y: visible !important;
        }
        .brand-title h1 {
            font-size: 1.15rem !important;
        }
        .brand-title .tagline {
            display: none !important;
        }
        .img-display-box {
            height: 180px !important;
        }
        .img-display-box img {
            height: 180px !important;
        }
        .stat-badge {
            margin-bottom: 0.4rem !important;
        }
        .info-strip-item {
            margin-bottom: 0.35rem !important;
        }
    }

    /* ── SIDEBAR CONTROLS STYLING (Card boxes, clean sliders, no red text) ── */
    [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
    }

    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-white);
        margin-top: 0 !important;
        margin-bottom: 0.35rem;
        padding-top: 0 !important;
        padding-left: 0.15rem;
    }
    .sidebar-header svg {
        color: var(--primary-cyan);
    }

    .sidebar-card-box {
        background: rgba(6, 38, 56, 0.85);
        border: 1px solid rgba(0, 217, 232, 0.35);
        border-radius: 9px;
        padding: 0.35rem 0.5rem;
        margin-bottom: 0.25rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    }

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
        margin-bottom: 0 !important;
    }
    [data-testid="stFileUploader"] > section {
        padding: 0.25rem 0.45rem !important;
        background: rgba(4, 25, 38, 0.7) !important;
        border: 1px dashed rgba(0, 217, 232, 0.6) !important;
        border-radius: 6px !important;
        min-height: 40px !important;
    }
    [data-testid="stFileUploader"] section > button {
        display: none !important;
    }
    [data-testid="stFileUploader"] section div {
        color: var(--primary-cyan) !important;
        font-size: 0.68rem !important;
    }

    .sidebar-section-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.76rem;
        font-weight: 600;
        color: #d1e4f2;
        margin-bottom: 0.15rem;
    }
    .sidebar-section-title span {
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }

    .slider-row-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.68rem;
        font-weight: 500;
        color: #8ea5b8;
        margin-bottom: -0.1rem;
        margin-top: 0.05rem;
    }
    .slider-row-header .val {
        color: #ffffff;
        font-weight: 600;
    }

    /* Suppress default Streamlit tick numbers and floating red bubbles */
    [data-testid="stSliderTickBar"] {
        display: none !important;
    }
    [data-testid="stSliderThumbValue"] {
        display: none !important;
    }

    .stSlider {
        margin-bottom: -0.25rem !important;
        padding-top: 0 !important;
    }
    .stSlider label[data-testid="stWidgetLabel"] {
        display: none !important;
    }
    .stSlider div[data-baseweb="slider"] {
        padding: 0.08rem 0 !important;
    }

    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
        background: #11364d !important;
        height: 3px !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background-color: var(--primary-cyan)"],
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="rgb("] {
        background: var(--primary-cyan) !important;
        height: 3px !important;
    }

    div[data-testid="stSlider"] [role="slider"] {
        background-color: #00E5F5 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 10px rgba(0, 229, 245, 0.9) !important;
        width: 12px !important;
        height: 12px !important;
    }

    [data-baseweb="select"] {
        background: #041a29 !important;
        border: 1px solid rgba(0, 217, 232, 0.4) !important;
        border-radius: 6px !important;
        min-height: 28px !important;
    }
    [data-baseweb="select"] * {
        color: #d8e9f5 !important;
        font-size: 0.72rem !important;
    }

    /* ── TOP NAV BAR ── */
    [data-testid="stLayoutWrapper"]:first-child {
        overflow: hidden !important;
    }
    .top-navbar-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(0, 217, 232, 0.2);
        box-sizing: border-box;
    }
    .brand-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .brand-title .logo-icon {
        background: var(--teal-gradient);
        border-radius: 6px;
        padding: 0.28rem 0.42rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #020D18;
        box-shadow: 0 0 14px rgba(0, 229, 245, 0.4);
    }
    .brand-title h1 {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .brand-title .tagline {
        font-size: 0.88rem;
        color: #728c9f;
        margin-left: 0.75rem;
        padding-left: 0.75rem;
        border-left: 1px solid rgba(0, 217, 232, 0.3);
        font-weight: 400;
    }

    .stDownloadButton > button {
        background: var(--teal-gradient) !important;
        color: #020D18 !important;
        border: none !important;
        padding: 0.35rem 1.15rem !important;
        border-radius: 20px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        box-shadow: 0 0 16px rgba(0, 229, 245, 0.4) !important;
        transition: transform 0.15s !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.4rem !important;
        height: 36px !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 0 22px rgba(0, 229, 245, 0.75) !important;
        color: #000000 !important;
    }
    .stButton > button {
        height: 36px !important;
        padding: 0.35rem 1rem !important;
        font-size: 0.82rem !important;
        border-radius: 20px !important;
        background: rgba(6, 38, 56, 0.8) !important;
        border: 1px solid rgba(0, 217, 232, 0.5) !important;
        color: #b0c9dd !important;
    }

    /* ── DASHBOARD CARDS ── */
    .studio-card {
        background: var(--card-bg);
        border: 1px solid rgba(0, 217, 232, 0.35);
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        box-sizing: border-box;
    }

    .card-header-bar {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.85rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }
    .card-header-bar svg {
        color: var(--primary-cyan);
    }

    .img-display-box {
        width: 100%;
        background: #01070e;
        border-radius: 7px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(0, 217, 232, 0.2);
    }
    .img-display-box img {
        width: 100%;
        object-fit: cover;
        display: block;
    }

    /* Metric Badges (Second Row) */
    .stat-badge {
        background: linear-gradient(135deg, rgba(6, 38, 56, 0.85) 0%, rgba(4, 25, 38, 0.8) 100%);
        border: 1px solid rgba(0, 217, 232, 0.4);
        border-radius: 9px;
        padding: 0.4rem 0.95rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 52px;
        box-sizing: border-box;
    }
    .stat-badge.highlight {
        background: linear-gradient(135deg, rgba(0, 95, 96, 0.45) 0%, rgba(0, 239, 165, 0.3) 100%);
        border-color: var(--secondary-emerald);
    }
    .stat-left {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .stat-icon {
        font-size: 1.35rem;
        color: var(--primary-cyan);
    }
    .stat-text .stat-title {
        font-size: 0.72rem;
        font-weight: 500;
        color: #8bb1cf;
    }
    .stat-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
    }

    /* Bottom Info Strip (Bottom Row) */
    .info-strip-item {
        background: rgba(6, 38, 56, 0.85);
        border: 1px solid rgba(0, 217, 232, 0.35);
        border-radius: 8px;
        padding: 0.35rem 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        height: 48px;
        box-sizing: border-box;
    }
    .info-strip-icon {
        color: var(--primary-cyan);
        font-size: 1.15rem;
    }
    .info-strip-meta .info-strip-label {
        font-size: 0.65rem;
        color: #7d9cb3;
        font-weight: 500;
        line-height: 1.1;
    }
    .info-strip-meta .info-strip-val {
        font-size: 0.92rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Controls (19% width, exact Image 2 layout, responsive) ───────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        Controls
    </div>
    """, unsafe_allow_html=True)

    # 1. Upload Card Box
    st.markdown("""
    <div class="sidebar-card-box">
        <div class="upload-label">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            Upload Image
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Basic Adjustments Card Box
    st.markdown("""
    <div class="sidebar-card-box">
        <div class="sidebar-section-title">
            <span>🔆 Basic Adjustments</span>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>
        </div>
    """, unsafe_allow_html=True)
    
    b_val = st.session_state.get("brightness", 0.25)
    st.markdown(f'<div class="slider-row-header"><span>Brightness</span><span class="val">{b_val:.2f}</span></div>', unsafe_allow_html=True)
    brightness = st.slider("Brightness", 0.0, 3.0, 0.25, 0.05, key="brightness", label_visibility="collapsed")

    c_val = st.session_state.get("contrast", 0.70)
    st.markdown(f'<div class="slider-row-header"><span>Contrast</span><span class="val">{c_val:.2f}</span></div>', unsafe_allow_html=True)
    contrast = st.slider("Contrast", 0.0, 3.0, 0.70, 0.05, key="contrast", label_visibility="collapsed")

    s_val = st.session_state.get("sharpness", 0.90)
    st.markdown(f'<div class="slider-row-header"><span>Sharpness</span><span class="val">{s_val:.2f}</span></div>', unsafe_allow_html=True)
    sharpness = st.slider("Sharpness", 0.0, 3.0, 0.90, 0.05, key="sharpness", label_visibility="collapsed")

    sat_val = st.session_state.get("saturation", 0.20)
    st.markdown(f'<div class="slider-row-header"><span>Saturation</span><span class="val">{sat_val:.2f}</span></div>', unsafe_allow_html=True)
    saturation = st.slider("Saturation", 0.0, 3.0, 0.20, 0.05, key="saturation", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Effects Card Box
    st.markdown("""
    <div class="sidebar-card-box">
        <div class="sidebar-section-title">
            <span>✨ Effects</span>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>
        </div>
    """, unsafe_allow_html=True)
    blur_val = st.session_state.get("blur_radius", 0.0)
    st.markdown(f'<div class="slider-row-header"><span>Blur Radius</span><span class="val">{blur_val:.2f}</span></div>', unsafe_allow_html=True)
    blur_radius = st.slider("Blur Radius", 0.0, 10.0, 0.00, 0.5, key="blur_radius", label_visibility="collapsed")

    vig_val = st.session_state.get("vignette", 0.0)
    st.markdown(f'<div class="slider-row-header"><span>Vignette</span><span class="val">{vig_val:.2f}</span></div>', unsafe_allow_html=True)
    vignette_strength = st.slider("Vignette", 0.0, 1.0, 0.00, 0.05, key="vignette", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. Filters Card Box
    st.markdown("""
    <div class="sidebar-card-box">
        <div class="sidebar-section-title">
            <span>🎨 Filters</span>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div class="slider-row-header" style="margin-bottom: 0.15rem;"><span>Color Filter</span></div>
    """, unsafe_allow_html=True)
    filter_mode = st.selectbox(
        "Color Filter",
        ["Sepia", "None", "Grayscale", "Invert"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 5. Transform Card Box
    st.markdown("""
    <div class="sidebar-card-box">
        <div class="sidebar-section-title">
            <span>🔄 Transform</span>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
    """, unsafe_allow_html=True)
    rot_val = st.session_state.get("rotation", 180)
    st.markdown(f'<div class="slider-row-header"><span>Rotation (°)</span><span class="val">{rot_val}</span></div>', unsafe_allow_html=True)
    rotation = st.slider("Rotation (°)", -180, 180, 180, 1, key="rotation", label_visibility="collapsed")

    st.markdown('<div class="slider-row-header" style="margin-bottom: 0.15rem; margin-top: 0.2rem;"><span>Flip</span></div>', unsafe_allow_html=True)
    flip_option = st.selectbox(
        "Flip",
        ["Vertical", "None", "Horizontal", "Both"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 6. Output Settings Card Box
    st.markdown("""
    <div class="sidebar-card-box">
        <div class="sidebar-section-title">
            <span>💾 Output Settings</span>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>
        </div>
        <div class="slider-row-header" style="margin-bottom: 0.15rem;"><span>Format</span></div>
    """, unsafe_allow_html=True)
    output_format = st.selectbox("Format", ["JPEG", "PNG", "WEBP"], index=0, label_visibility="collapsed")
    q_val = st.session_state.get("quality", 85)
    st.markdown(f'<div class="slider-row-header"><span>Quality</span><span class="val">{q_val}</span></div>', unsafe_allow_html=True)
    jpeg_quality = st.slider("Quality", 10, 100, 85, 5, key="quality", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

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
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
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

# ── FIRST ROW (Original & Processed Images) ───────────────────────────────────
col_orig, col_proc = st.columns([1, 1])

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
            line=dict(color=colors[ch], width=1.4),
            opacity=0.65,
        ))
    fig.update_layout(
        margin=dict(l=25, r=10, t=5, b=15),
        height=145,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d9cb3", size=9),
        xaxis=dict(
            showgrid=True,
            gridcolor="#0e2336",
            range=[0, 255],
            dtick=40,
            tickfont=dict(color="#68849b", size=9),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#0e2336",
            tickfont=dict(color="#68849b", size=9),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=9, color="#9bb4c7"),
        ),
    )
    return fig

with h_col1:
    st.markdown("""
    <div class="studio-card" style="padding: 0.45rem 0.65rem 0.25rem 0.65rem;">
        <div class="card-header-bar" style="margin-bottom: 0.15rem;">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
            Original Histogram
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(make_compact_hist(orig_hist), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with h_col2:
    st.markdown("""
    <div class="studio-card" style="padding: 0.45rem 0.65rem 0.25rem 0.65rem;">
        <div class="card-header-bar" style="margin-bottom: 0.15rem;">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
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
