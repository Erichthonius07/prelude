# training/train.py
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SIDDMediumDataset
from leakage_guard import get_safe_training_pairs
from model import DnCNN
from results_client import ResultsClient

logger = logging.getLogger("TrainLoop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) between two PyTorch tensors."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0  # Cap for perfect identical images to avoid division by zero
    return 20.0 * math.log10(1.0 / math.sqrt(mse))

def train(
    data_dir: str, 
    api_url: str, 
    run_id: str, 
    epochs: int = 20, 
    batch_size: int = 16, 
    lr: float = 1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on {device}...")
    
    # Initialize the automated Results Service messenger (§5.6)
    client = ResultsClient(api_url, run_id, "denoise_cnn")
    
    # 1. Locate all raw image candidates from the SIDD directory structure
    base_path = Path(data_dir)
    raw_pairs = []
    for scene_dir in base_path.glob("Scene_Instances/*"):
        if not scene_dir.is_dir(): 
            continue
        noisy_files = sorted(list(scene_dir.glob("NOISY_*.PNG")))
        gt_files = sorted(list(scene_dir.glob("GT_*.PNG")))
        for n_path, g_path in zip(noisy_files, gt_files):
            image_id = f"sidd_{scene_dir.name}_{n_path.stem}"
            raw_pairs.append((image_id, n_path, g_path))

    # 2. Programmatically filter out held-out test images using the guard API (§8)
    safe_pairs = get_safe_training_pairs(raw_pairs, api_url)
    
    # 3. Create a strict 90/10 split for training vs. validation
    split_idx = int(0.9 * len(safe_pairs))
    train_loader = DataLoader(
        SIDDMediumDataset(safe_pairs[:split_idx], patch_size=128, is_train=True), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        # Validation uses larger, deterministic crops to ensure consistent benchmarking
        SIDDMediumDataset(safe_pairs[split_idx:], patch_size=256, is_train=False), 
        batch_size=1, 
        shuffle=False
    )

    # 4. Initialize model, optimizer, and specific loss function
    model = DnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # L1 Loss is explicitly chosen and justified over MSE (L2) to preserve sharp edges
    criterion = nn.L1Loss() 

    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss_accum = 0.0
        
        # --- TRAINING PASS ---
        for _, noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            
            optimizer.zero_grad()
            denoised_pred = model(noisy)
            loss = criterion(denoised_pred, gt)
            loss.backward()
            optimizer.step()
            
            epoch_loss_accum += loss.item()
            global_step += 1
            
        avg_train_loss = epoch_loss_accum / len(train_loader)
        
        # --- VALIDATION & SCORING PASS ---
        model.eval()
        per_image_scores = []
        with torch.no_grad():
            for img_ids, noisy, gt in val_loader:
                img_id_str = img_ids[0]  # Unpack batch of 1
                noisy, gt = noisy.to(device), gt.to(device)
                
                # Clamp outputs to valid pixel range [0.0, 1.0] to prevent overflow during metric calculation
                denoised = torch.clamp(model(noisy), 0.0, 1.0)
                
                score_psnr = calculate_psnr(denoised, gt)
                
                # Format exactly as the training metric schema requires (§5.6)
                per_image_scores.append({
                    "imageId": img_id_str,
                    "epoch": epoch,
                    "globalStep": global_step,
                    "metricName": "psnr",
                    "value": round(score_psnr, 4),
                    "split": "validation"
                })
        
        # 5. Push metrics to Results Service (§5.6)
        loss_curves = [{
            "curve": "train_loss",
            "points": [{"step": global_step, "value": round(avg_train_loss, 6)}]
        }]
        
        try:
            client.submit_training_metrics(
                per_image_scores=per_image_scores, 
                loss_curves=loss_curves
            )
            logger.info(f"Epoch {epoch} finished. Validation scores uploaded successfully.")
        except Exception as e:
            logger.error(f"Failed to submit metrics for Epoch {epoch}: {e}")

if __name__ == "__main__":
    train(
        data_dir="./data/SIDD_Medium_Srgb_Patches",
        api_url="http://localhost:8080",
        run_id="phase3-denoise-sidd-l1"
    )