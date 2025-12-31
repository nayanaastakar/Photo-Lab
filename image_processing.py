import numpy as np
from PIL import Image, ImageFilter

def apply_brightness(img_array, factor):
    return np.clip(img_array * factor, 0, 1)

def apply_contrast(img_array, factor):
    mean = np.mean(img_array, axis=(0,1), keepdims=True)
    return np.clip((img_array - mean) * factor + mean, 0, 1)

def apply_blur(pil_img, radius):
    return pil_img.filter(ImageFilter.GaussianBlur(radius))

def analyze_changes(original, processed):
    diff = np.abs(original - processed)

    avg_change = np.mean(diff) * 100
    max_change = np.max(diff) * 100
    affected_pixels = np.sum(diff > 0.05)
    total_pixels = diff.size
    percent_affected = (affected_pixels / total_pixels) * 100

    return avg_change, max_change, percent_affected
