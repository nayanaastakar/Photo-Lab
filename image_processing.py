import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io


def apply_brightness(img_array, factor):
    return np.clip(img_array * factor, 0, 1)


def apply_contrast(img_array, factor):
    mean = np.mean(img_array, axis=(0, 1), keepdims=True)
    return np.clip((img_array - mean) * factor + mean, 0, 1)


def apply_blur(pil_img, radius):
    return pil_img.filter(ImageFilter.GaussianBlur(radius))


def apply_sharpness(pil_img, factor):
    """Apply sharpness enhancement. factor=1.0 is original, >1 sharpens, <1 softens."""
    enhancer = ImageEnhance.Sharpness(pil_img)
    return enhancer.enhance(factor)


def apply_saturation(pil_img, factor):
    """Apply color saturation. factor=1.0 is original, 0=grayscale, >1=vivid."""
    enhancer = ImageEnhance.Color(pil_img)
    return enhancer.enhance(factor)


def apply_rotation(pil_img, angle):
    """Rotate image by given angle (degrees). Uses bicubic resampling with white fill."""
    return pil_img.rotate(-angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))


def apply_flip(pil_img, direction):
    """Flip image. direction: 'horizontal', 'vertical', or 'both'."""
    if direction == "Horizontal":
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == "Vertical":
        return pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    elif direction == "Both":
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    return pil_img


def apply_grayscale(img_array):
    """Convert to grayscale using luminosity method, returns 3-channel array."""
    gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    return np.stack([gray, gray, gray], axis=-1)


def apply_sepia(img_array):
    """Apply sepia tone filter."""
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia = img_array @ sepia_filter.T
    return np.clip(sepia, 0, 1)


def apply_invert(img_array):
    """Invert image colors."""
    return 1.0 - img_array


def apply_vignette(img_array, strength=0.5):
    """Apply vignette effect (darkened edges)."""
    rows, cols = img_array.shape[:2]
    X = np.arange(cols) - cols / 2
    Y = np.arange(rows) - rows / 2
    X, Y = np.meshgrid(X, Y)
    radius = np.sqrt(X**2 + Y**2)
    max_radius = np.sqrt((cols / 2)**2 + (rows / 2)**2)
    vignette = 1 - strength * (radius / max_radius)**2
    vignette = np.clip(vignette, 0, 1)
    return np.clip(img_array * vignette[..., np.newaxis], 0, 1)


def compute_histogram(img_array):
    """Compute histogram data for R, G, B channels. Returns dict of channel arrays."""
    img_uint8 = (img_array * 255).astype(np.uint8)
    histograms = {}
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        hist, _ = np.histogram(img_uint8[:, :, i], bins=256, range=(0, 256))
        histograms[color] = hist
    return histograms


def analyze_changes(original, processed):
    diff = np.abs(original - processed)
    avg_change = np.mean(diff) * 100
    max_change = np.max(diff) * 100
    affected_pixels = np.sum(diff > 0.05)
    total_pixels = diff.size
    percent_affected = (affected_pixels / total_pixels) * 100
    return avg_change, max_change, percent_affected


def pil_to_bytes(pil_img, fmt="PNG"):
    """Convert PIL image to bytes for download."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()
