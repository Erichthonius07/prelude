import argparse
import json
import logging
import uuid
from datetime import datetime, timezone
import requests
import sys

logger = logging.getLogger("SubmitMetrics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def fetch_manifest(base_url, run_id):
    try:
        resp = requests.get(f"{base_url}/api/v1/runs/{run_id}/manifest", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("imageIds", [])
        logger.warning(f"Failed to fetch manifest, status: {resp.status_code}")
        return None
    except requests.RequestException as e:
        logger.warning(f"Error fetching manifest: {e}")
        return None

def fetch_run(base_url, run_id):
    try:
        resp = requests.get(f"{base_url}/api/v1/runs/{run_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException:
        return None

def build_payload(run_id: str, metrics_path: str, latency_path: str, device_id: str = "primary-device") -> dict:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    with open(latency_path, "r") as f:
        latency = json.load(f)
        
    model_variant = metrics.get("model_variant")
    model_size = metrics.get("file_sizes", {}).get("int8_tflite_bytes")
    dataset = metrics.get("dataset", "unknown")
    input_domain = metrics.get("input_domain", "unknown")
    input_size = metrics.get("input_size", "unknown")
    
    if not model_variant or not model_size:
        raise ValueError("Missing model_variant or modelFileSizeBytes in metrics JSON")
        
    latency_lookup = {}
    for ev in latency.get("events", []):
        key = (ev["imageId"], ev["precision"])
        latency_lookup[key] = {
            "latencyMs": ev["latencyMs"],
            "discardRaceEvent": ev.get("discardRaceEvent", False),
            "timeoutEvent": ev.get("timeoutEvent", False)
        }
        
    images = []
    
    for row in metrics.get("per_image", []):
        image_id = row["imageId"]
        precision = row["precision"]
        
        psnr = row.get("psnr_gt")
        ssim = row.get("ssim_gt")
        if psnr is None or ssim is None:
            raise ValueError(f"Missing psnr_gt or ssim_gt for {image_id} {precision}")
            
        key = (image_id, precision)
        lat = latency_lookup.get(key)
        if lat is None:
            raise ValueError(f"Missing latency data for {image_id} {precision}")
            
        images.append({
            "imageId": image_id,
            "precision": precision,
            "psnr": float(psnr), 
            "ssim": float(ssim),
            "latencyMs": float(lat["latencyMs"]),
            "discardRaceEvent": bool(lat["discardRaceEvent"]),
            "timeoutEvent": bool(lat["timeoutEvent"])
        })
        
    payload = {
        "modelVariant": model_variant,
        "modelFileSizeBytes": model_size,
        "dataset": dataset,
        "inputDomain": input_domain,
        "inputSize": input_size,
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

def run_cli(args_list=None):
    parser = argparse.ArgumentParser(description="Submit denoise metrics to Results Service")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g., phase3-int8-deploy)")
    parser.add_argument("--planned-test-n", type=int, required=True, help="Test set N")
    parser.add_argument("--planned-comparisons", type=int, required=True, help="Comparisons (e.g., 1)")
    parser.add_argument("--base-url", required=True, help="Results Service base URL")
    parser.add_argument("--metrics-json", required=True, help="Metrics JSON file with PSNR/SSIM vs GT")
    parser.add_argument("--latency-json", required=True, help="Latency JSON file from device")
    parser.add_argument("--send", action="store_true", help="Actually send the payload")
    
    args = parser.parse_args(args_list)
    
    envelope = build_payload(args.run_id, args.metrics_json, args.latency_json)
    images = envelope["payload"]["images"]
    
    input_domain = envelope["payload"].get("inputDomain", "")
    if args.send and input_domain.lower() == "srgb" and not args.run_id.startswith("smoke-"):
        print(f"Error: --send is only allowed for smoke tests (run-id must start with 'smoke-') when input_domain is sRGB.")
        return 1

    manifest_image_ids = fetch_manifest(args.base_url, args.run_id)
    manifest_verified = False
    if manifest_image_ids is None:
        print("manifest NOT VERIFIED")
        if args.send:
            print("Refusing to --send without a verified manifest.")
            return 1
    else:
        manifest_verified = True
        manifest_set = set(manifest_image_ids)
        submitted_set = set(img["imageId"] for img in images)
        
        if not submitted_set.issubset(manifest_set):
            diff = submitted_set - manifest_set
            print(f"Error: Submitted imageIds not in manifest: {diff}")
            return 1
            
        if len(submitted_set) < len(manifest_set):
            print(f"Warning: Partial coverage. Submitted {len(submitted_set)} images, manifest has {len(manifest_set)}.")
            
        if args.send:
            if len(submitted_set) != len(manifest_set):
                print("Error: --send requires full coverage.")
                return 1
            
            for img_id in manifest_set:
                precisions = [img["precision"] for img in images if img["imageId"] == img_id]
                if "fp32" not in precisions or "int8" not in precisions:
                    print(f"Error: --send requires both fp32 and int8 rows for {img_id}")
                    return 1
                    
    run_info = fetch_run(args.base_url, args.run_id)
    should_register = True
    if run_info:
        print(f"Run {args.run_id} already exists.")
        should_register = False
        if manifest_verified and run_info.get("plannedTestN") != len(manifest_image_ids):
            print(f"Error: run plannedTestN ({run_info.get('plannedTestN')}) != manifest count ({len(manifest_image_ids)})")
            return 1

    if args.send:
        if should_register:
            print("\nRegistering run...")
            reg_payload = {
                "runId": args.run_id,
                "plannedTestN": args.planned_test_n,
                "plannedComparisons": args.planned_comparisons
            }
            resp = requests.post(f"{args.base_url}/api/v1/runs", json=reg_payload)
            if resp.status_code not in (200, 201):
                print(f"Failed to register run: {resp.status_code} {resp.text}")
                return 1
                
        print("\nTransmitting denoise metrics...")
        submit_resp = requests.post(f"{args.base_url}/api/v1/metrics/denoise", json=envelope)
        print("Submission response:", submit_resp.status_code)
    else:
        print("\n--- Dry run complete. Did not send. ---")
    return 0

def main():
    sys.exit(run_cli())

if __name__ == "__main__":
    main()
