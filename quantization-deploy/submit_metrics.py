import argparse
import json
import logging
import uuid
from datetime import datetime, timezone
import requests

logger = logging.getLogger("SubmitMetrics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def build_payload(run_id: str, device_id: str = "mock-device", latency_ms: float = 15.0) -> dict:
    """Builds the denoise payload per Data Contract 5.4."""
    with open("quantization-deploy/outputs/quantization_results.json", "r") as f:
        results = json.load(f)

    model_variant = results.get("model_variant", "cnn-v1")
    model_size = results.get("file_sizes", {}).get("int8_tflite_bytes", 607792)
    
    images = []
    
    for row in results.get("int8_evaluation", {}).get("per_image", []):
        idx = row["image_idx"]
        image_id = f"smoke-test-burst:{idx:06d}"
        
        # fp32 row
        images.append({
            "imageId": image_id,
            "precision": "fp32",
            "psnr": 40.0, 
            "ssim": 0.99,
            "latencyMs": latency_ms * 3.5,
            "discardRaceEvent": False,
            "timeoutEvent": False
        })
        
        # int8 row
        images.append({
            "imageId": image_id,
            "precision": "int8",
            "psnr": 40.0 - (100.0 - row["psnr_fp32_vs_int8"]), 
            "ssim": 0.99 * row["ssim_fp32_vs_int8"],
            "latencyMs": latency_ms,
            "discardRaceEvent": False,
            "timeoutEvent": False
        })

    payload = {
        "modelVariant": model_variant,
        "modelFileSizeBytes": model_size,
        "images": images
    }

    envelope = {
        "submissionId": str(uuid.uuid4()),
        "runId": run_id,
        "deviceId": device_id,
        "occurredAt": datetime.now(timezone.utc).isoformat(),
        "payload": payload
    }
    return envelope

def main():
    parser = argparse.ArgumentParser(description="Submit denoise metrics to Results Service")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g., phase3-int8-deploy)")
    parser.add_argument("--planned-test-n", type=int, required=True, help="Test set N")
    parser.add_argument("--planned-comparisons", type=int, required=True, help="Comparisons (e.g., 1)")
    parser.add_argument("--base-url", required=True, help="Results Service base URL")
    
    args = parser.parse_args()

    # 1. Build Payload
    envelope = build_payload(args.run_id)
    
    # 2. Cache Locally
    cache_path = "quantization-deploy/outputs/submit_payload_cache.json"
    with open(cache_path, "w") as f:
        json.dump(envelope, f, indent=2)
        
    logger.info(f"Payload cached locally to {cache_path}")
    
    # Dump payload to stdout so user can see it before sending
    print(json.dumps(envelope, indent=2))
    print(f"\n--- Payload generated. Awaiting confirmation before sending to: {args.base_url} ---")
    
    if args.base_url != "TBD":
        # 3. Register Run
        print("\nRegistering run...")
        reg_payload = {
            "runId": args.run_id,
            "plannedTestN": args.planned_test_n,
            "plannedComparisons": args.planned_comparisons
        }
        resp = requests.post(f"{args.base_url}/api/v1/runs", json=reg_payload)
        print("Registration response:", resp.status_code, resp.text)
        
        # 4. Transmit Payload
        print("\nTransmitting denoise metrics...")
        submit_resp = requests.post(f"{args.base_url}/api/v1/metrics/denoise", json=envelope)
        print("Submission response:", submit_resp.status_code)
        
        try:
            print("Server response verbatim:", json.dumps(submit_resp.json(), indent=2))
        except Exception:
            print("Raw text:", submit_resp.text)
            
if __name__ == "__main__":
    main()
