import numpy as np
import cv2

def globaltonemap(image, gamma=1.0):
    # one curve for the whole image, simple gamma, gamma<1 brightens shadows
    image = np.clip(image, 0.0, 1.0)
    return np.power(image, gamma, dtype=np.float32)

def localtonemap(image, sigma=15.0, basegamma=0.7):
    # split into base (blurry, low freq) and detail (whats left)
    # only squash the base so shadows/highlights get fixed but texture stays sharp
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    base = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=sigma)
    detail = image - base
    baseSquashed = np.power(np.clip(base, 0.0, 1.0), basegamma)
    result = baseSquashed + detail
    return np.clip(result, 0.0, 1.0).astype(np.float32)