"""
results.py -- metric submissions for Role 1's Results Service, with local-JSON fallback.

Every submission carries PER-IMAGE scores (bootstrap needs them) and a client-made
submissionId so a retried upload is idempotent (proposal 4.1).

Until the Results Service is live, submissions are written to results_local/ as JSON;
Role 1's backfill script pushes them later. post_pending() uploads them once it is up.

Wire shape is Role 1's ingestion envelope (docs/data-contract.md §3, §11):
POST <url>/api/v1/metrics/fusion | /postprocess, runId = a registered evaluation run.
"""
import json, time, urllib.request, uuid
from pathlib import Path

SCHEMA_VERSION = "role3-1.0.0"
LOCAL_DIR = Path("results_local")


def make_submission(module, variant, per_image, dataset_id, device_id="desktop", extra=None):
    """per_image: list of dicts like {"imageId": ..., "psnr": ..., "ssim": ...}."""
    if not per_image:
        raise ValueError("per-image scores are required; an aggregate-only submission can't be bootstrapped")
    ids = [p["imageId"] for p in per_image]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate imageId in per-image scores")
    if module == "FusionModule":
        strategy = {"confidence": "confidence_weighted"}.get(variant.removeprefix("fusion_"), variant.removeprefix("fusion_"))
        payload = {"images": [{"imageId": p["imageId"], "scores": [{"strategy": strategy, "ssim": p["ssim"], "psnr": p["psnr"]}]} for p in per_image]}
    else:
        payload = {"images": [{"imageId": p["imageId"], "ssimBeforePostprocess": p["ssim_before"], "ssimAfterPostprocess": p["ssim"],
                               "ssimContribution": p["ssim_delta"]} for p in per_image]}
    return {
        "submissionId": str(uuid.uuid4()),
        "runId": dataset_id,              # MUST be a run registered with POST /api/v1/runs
        "deviceId": device_id,
        "appVersion": SCHEMA_VERSION,
        "occurredAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "payload": payload,
        # local-only fields below; post() sends just the envelope above
        "stage": "fusion" if module == "FusionModule" else "postprocess",
        "variant": variant,               # e.g. "fusion_naive", "fusion_learned", "postprocess"
        "n": len(per_image),
        "extra": extra or {},
    }


def save_local(sub, local_dir=LOCAL_DIR):
    local_dir = Path(local_dir); local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / f"{sub['variant']}_{sub['submissionId']}.json"
    path.write_text(json.dumps(sub, indent=1))
    return path


def post(sub, url, timeout=10):
    envelope = {k: sub[k] for k in ("submissionId", "runId", "deviceId", "appVersion", "occurredAt", "payload")}
    req = urllib.request.Request(f"{url.rstrip('/')}/api/v1/metrics/{sub['stage']}", data=json.dumps(envelope).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def submit(sub, url=None, local_dir=LOCAL_DIR):
    """Always caches locally first; uploads too when url is given. Returns (local_path, uploaded)."""
    path = save_local(sub, local_dir)
    if not url:
        return path, False
    try:
        post(sub, url)
        sent = path.with_suffix(".sent")
        path.rename(sent)
        return sent, True
    except Exception as e:                # keep the local copy, upload later with post_pending
        print(f"  upload failed ({type(e).__name__}: {e}); kept {path}")
        return path, False


def post_pending(url, local_dir=LOCAL_DIR):
    """Upload every cached .json that hasn't been sent. Safe to re-run (same submissionId)."""
    sent = 0
    for p in sorted(Path(local_dir).glob("*.json")):
        sub = json.loads(p.read_text())
        post(sub, url)
        p.rename(p.with_suffix(".sent")); sent += 1
    return sent
