# training/inference.py
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import DnCNN

def estimate_noise_std(img_tensor: torch.Tensor) -> float:
    """
    Immerkaer's Fast Noise Variance Estimation.
    Mathematically estimates the standard deviation of noise in an image 
    without needing a clean ground-truth reference.
    """
    # 1. Convert image to grayscale for structural analysis
    gray = (0.2989 * img_tensor[:, 0:1, :, :] + 
            0.5870 * img_tensor[:, 1:2, :, :] + 
            0.1140 * img_tensor[:, 2:3, :, :])
    
    # 2. Immerkaer's 3x3 Laplacian kernel to isolate high-frequency grain
    kernel = torch.tensor([
        [ 1.0, -2.0,  1.0], 
        [-2.0,  4.0, -2.0], 
        [ 1.0, -2.0,  1.0]
    ], device=img_tensor.device).view(1, 1, 3, 3)
    
    # 3. Apply the filter and calculate the standard deviation (sigma)
    laplacian = F.conv2d(gray, kernel, padding=0)
    sigma = torch.sum(torch.abs(laplacian)) * math.sqrt(0.5 * math.pi) / (6 * laplacian.numel())
    
    # Convert from [0.0, 1.0] scale to standard [0, 255] pixel scale
    return sigma.item() * 255.0

def run_inference_with_analysis(image_path: str, model_path: str, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running analytical inference on: {device}")
    
    model = DnCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load and prepare image
    img = Image.open(image_path).convert("RGB")
    noisy_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    noisy_tensor = noisy_tensor.to(device)

    # Process through the AI
    with torch.no_grad():
        denoised_tensor = torch.clamp(model(noisy_tensor), 0.0, 1.0)
        
        # Calculate the exact noise the AI extracted (Original - Clean = Noise)
        extracted_noise_tensor = torch.clamp(noisy_tensor - denoised_tensor + 0.5, 0.0, 1.0)

    # Calculate blind noise levels using Immerkaer's formula
    noise_lvl_orig = estimate_noise_std(noisy_tensor)
    noise_lvl_clean = estimate_noise_std(denoised_tensor)
    reduction_percent = ((noise_lvl_orig - noise_lvl_clean) / noise_lvl_orig) * 100

    print("-" * 40)
    print("🔬 NOISE REDUCTION ANALYSIS")
    print(f"Original Noise Level (σ): {noise_lvl_orig:.2f}")
    print(f"Denoised Noise Level (σ): {noise_lvl_clean:.2f}")
    print(f"Total Noise Reduced:      {reduction_percent:.1f}%")
    print("-" * 40)

    # Convert tensors back to visual images
    noisy_img = (noisy_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    denoised_img = (denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # The noise map is shifted by 128 (0.5) so negative noise is visible as dark gray
    noise_map_img = (extracted_noise_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Generate a visual analytical report
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(noisy_img)
    axes[0].set_title(f"Original (Noise σ: {noise_lvl_orig:.2f})")
    
    axes[1].imshow(denoised_img)
    axes[1].set_title(f"Denoised (Noise σ: {noise_lvl_clean:.2f})")
    
    axes[2].imshow(noise_map_img)
    axes[2].set_title(f"Extracted AI Noise Map\n(Reduction: {reduction_percent:.1f}%)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    run_inference_with_analysis(
        image_path="noisy_zee.jpeg", 
        model_path="dncnn_best_med.pth",        
        output_path="denoise_analysis_report_zee_med.png"
    )