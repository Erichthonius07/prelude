import numpy as np

def naivemeanfusion(frames):
    # frames shape (N,H,W,C) float32 0 to 1, just average across the burst axis
    if frames.ndim != 4:
        raise ValueError(f"expected NHWC, got {frames.shape}")
    if frames.shape[0] < 1:
        raise ValueError("need at least 1 frame")
    return np.mean(frames, axis=0, dtype=np.float32)

def trimmedmeanfusion(frames, trimfrac=0.2):
    # sorts each pixel across frames, chops off extremes, averages the rest
    # this is how we dodge ghosting without needing role2 confidence scores
    n = frames.shape[0]
    trimcount = int(np.floor(n * trimfrac / 2))
    if n - 2 * trimcount < 1:
        raise ValueError(f"trimfrac {trimfrac} leaves 0 frames for n={n}, lower it or use longer burst")
    sortedframes = np.sort(frames, axis=0)
    if trimcount == 0:
        return np.mean(sortedframes, axis=0, dtype=np.float32)
    trimmed = sortedframes[trimcount:n-trimcount]
    return np.mean(trimmed, axis=0, dtype=np.float32)

if __name__ == "__main__":
    # bare smoke check, just confirms the functions run and shapes are sane
    burst = np.random.uniform(0,1,size=(6,32,32,3)).astype(np.float32)
    print("naive shape:", naivemeanfusion(burst).shape)
    print("trimmed shape:", trimmedmeanfusion(burst, trimfrac=0.4).shape)