# training/train.py
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SIDDMediumDataset # Renaming this class isn't strictly necessary
from leakage_guard import get_safe_training_pairs
from model import DnCNN
from results_client import ResultsClient

logger = logging.getLogger("TrainLoop")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
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
    
    client = ResultsClient(api_url, run_id, "denoise_cnn")
    
    # 1. UPDATED: Target the SIDD Small 'Data' directory structure directly
    base_path = Path(data_dir)
    raw_pairs = []
    for scene_dir in base_path.iterdir():
        if not scene_dir.is_dir(): 
            continue
            
        # Target the Noisy and GT sRGB PNG files explicitly mentioned in the ReadMe
        noisy_files = sorted(list(scene_dir.rglob("*NOISY*.PNG")) + list(scene_dir.rglob("*NOISY*.png")))
        gt_files = sorted(list(scene_dir.rglob("*GT*.PNG")) + list(scene_dir.rglob("*GT*.png")))
        
        for n_path, g_path in zip(noisy_files, gt_files):
            # The directory name (e.g., 0001_001_S6_...) acts as the unique scene instance ID
            image_id = f"sidd_{scene_dir.name}"
            raw_pairs.append((image_id, n_path, g_path))

    if not raw_pairs:
        raise FileNotFoundError(f"No PNG image pairs found in {data_dir}. Check your extracted dataset structure.")

    safe_pairs = get_safe_training_pairs(raw_pairs, api_url)
    split_idx = int(0.9 * len(safe_pairs))
    
    train_loader = DataLoader(
        SIDDMediumDataset(safe_pairs[:split_idx], patch_size=128, is_train=True), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        SIDDMediumDataset(safe_pairs[split_idx:], patch_size=256, is_train=False), 
        batch_size=1, 
        shuffle=False
    )

    model = DnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() 

    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss_accum = 0.0
        
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
        
        model.eval()
        per_image_scores = []
        with torch.no_grad():
            for img_ids, noisy, gt in val_loader:
                img_id_str = img_ids[0]
                noisy, gt = noisy.to(device), gt.to(device)
                denoised = torch.clamp(model(noisy), 0.0, 1.0)
                score_psnr = calculate_psnr(denoised, gt)
                
                per_image_scores.append({
                    "imageId": img_id_str,
                    "epoch": epoch,
                    "globalStep": global_step,
                    "metricName": "psnr",
                    "value": round(score_psnr, 4),
                    "split": "validation"
                })
        
        loss_curves = [{
            "curve": "train_loss",
            "points": [{"step": global_step, "value": round(avg_train_loss, 6)}]
        }]
        
        try:
            client.submit_training_metrics(per_image_scores=per_image_scores, loss_curves=loss_curves)
            logger.info(f"Epoch {epoch} finished. Validation scores uploaded successfully.")
        except Exception as e:
            logger.error(f"Failed to submit metrics for Epoch {epoch}: {e}")

if __name__ == "__main__":
    # Target the new directory name
    train(
        data_dir="./data/Data",
        api_url="http://localhost:8080",
        run_id="phase3-denoise-sidd-small-l1"
    )