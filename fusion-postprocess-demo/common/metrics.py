import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def computepsnr(imgA, imgB):
    # higher is better, measures raw pixel error in db scale
    return psnr(imgA, imgB, data_range=1.0)

def computessim(imgA, imgB):
    # higher is better (max 1.0), measures structural/perceptual similarity
    # channel_axis tells it color images are HWC not grayscale
    return ssim(imgA, imgB, data_range=1.0, channel_axis=2)

def edgeenergy(image):
    # rough proxy for "how sharp does this look", sum of gradient magnitude
    # useful for sharpening tests where we dont have a clean reference to compare against
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    gy, gx = np.gradient(gray)
    return np.mean(np.sqrt(gx**2 + gy**2))