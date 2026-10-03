# training/export_handoff.py
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

def generate_role5_manifest(
    model_filename: str = "dncnn_baseline.onnx", 
    run_id: str = "phase3-denoise-sidd-medium-l1", 
    domain: str = "srgb"
):
    # 1. Setup the requested directory structure
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    source_model = Path(model_filename)
    if not source_model.exists():
        raise FileNotFoundError(f"{model_filename} not found. Run export_onnx.py first.")
        
    dest_model = checkpoint_dir / source_model.name
    shutil.copy(source_model, dest_model)
    
    # 2. Calculate the SHA-256 Hash
    sha256 = hashlib.sha256()
    with open(dest_model, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    model_hash = sha256.hexdigest()
    
    # 3. Extract the current Git commit hash
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown_no_git_repository"
        
    # 4. Construct the required metadata dictionary
    manifest = {
        "checkpoint_path": f"data/checkpoints/{dest_model.name}",
        "sha256_hash": model_hash,
        "git_commit": git_commit,
        "run_id": run_id,
        "input_domain": domain,
        "handoff_notes": "Smoke-test only. Model trained on 8-bit sRGB. Pending linear fine-tuning."
    }
    
    # 5. Save the manifest for Role 5
    manifest_path = checkpoint_dir / "role5_handoff.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"✅ Handoff package generated successfully in {checkpoint_dir}/")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    generate_role5_manifest()