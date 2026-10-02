import hashlib
import requests
from pathlib import Path
from typing import List, Tuple

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_safe_training_pairs(
    candidates: List[Tuple[str, Path, Path]], 
    api_base_url: str
) -> List[Tuple[str, Path, Path]]:
    # candidates: list of (imageId, noisy_path, clean_path)
    payload = [{"imageId": c[0], "sha256": compute_sha256(c[1])} for c in candidates]
    
    url = f"{api_base_url}/api/v1/testset/check"
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    
    held_out_ids = {item["imageId"] for item in resp.json().get("heldout", [])}
    
    safe_pairs = [c for c in candidates if c[0] not in held_out_ids]
    print(f"Leakage Guard: Removed {len(held_out_ids)} held-out test images.")
    return safe_pairs