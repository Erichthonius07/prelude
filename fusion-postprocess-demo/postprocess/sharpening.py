import numpy as np
import cv2

def unsharpmask(image, radius=1.5, amount=1.0, threshold=0.0):
    # blur it, find the difference (thats the detail/edges), add extra copies back
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    blurred = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=radius)
    detail = image - blurred
    if threshold > 0.0:
        # dont amplify tiny detail, thats usually just noise not real edges
        mask = np.abs(detail) >= threshold
        detail = detail * mask
    sharpened = image + amount * detail
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)

if __name__ == "__main__":
    # bare smoke check, just confirms it runs and shape/range stay sane
    img = np.random.uniform(0,1,size=(32,32,3)).astype(np.float32)
    out = unsharpmask(img, radius=1.5, amount=1.0, threshold=0.02)
    print("shape:", out.shape, "min:", out.min(), "max:", out.max())