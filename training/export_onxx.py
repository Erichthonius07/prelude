# training/export_onnx.py
from pathlib import Path
import torch
from model import DnCNN

def export_to_onnx(
    checkpoint_path: str = "dncnn_best_med.pth",
    output_onnx_path: str = "dncnn_baseline.onnx",
    opset_version: int = 13
):
    # Export on CPU to avoid baking CUDA runtime metadata into the static graph
    device = torch.device("cpu")
    
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Make sure you have trained the model "
            f"or renamed your best epoch file (e.g., 'dncnn_epoch_19.pth') to '{checkpoint_path}'."
        )

    # 1. Initialize architecture and load trained weights
    model = DnCNN().to(device)
    model.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=True))
    model.eval()

    # 2. Create representative dummy input tensor [Batch=1, Channels=3, H=256, W=256]
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32, device=device)

    # 3. Export to ONNX with dynamic spatial/batch axes for downstream flexibility
    print(f"Tracing model from '{checkpoint_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    )
    print(f"Export complete. Model saved to '{output_onnx_path}'.")

if __name__ == "__main__":
    export_to_onnx()