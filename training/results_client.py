# training/results_client.py
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("ResultsClient")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ResultsServiceError(Exception):
    """Custom exception raised when the Results Service rejects a request."""
    def __init__(self, status_code: int, code: str, message: str, details: Any = None):
        super().__init__(f"HTTP {status_code} [{code}]: {message} (Details: {details})")
        self.status_code = status_code
        self.code = code
        self.details = details


class ResultsClient:
    def __init__(
        self,
        base_url: str,
        run_id: str,
        model_name: str,
        device_id: Optional[str] = None,
        timeout_seconds: int = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.run_id = run_id
        self.model_name = model_name
        self.device_id = device_id
        self.timeout = timeout_seconds

        # Configure resilient HTTP session with automated connection retries
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def register_run(
        self,
        planned_test_n: int,
        planned_comparisons: int,
        description: Optional[str] = None,
        phase: Optional[str] = "Phase 3",
        created_by: Optional[str] = "Role 4",
        ci_level: float = 0.95,
        bootstrap_resamples: int = 10000,
    ) -> Dict[str, Any]:
        """Registers an evaluation campaign run before pushing metrics (§4)."""
        url = f"{self.base_url}/api/v1/runs"
        payload = {
            "runId": self.run_id,
            "plannedTestN": planned_test_n,
            "plannedComparisons": planned_comparisons,
            "description": description,
            "phase": phase,
            "createdBy": created_by,
            "ciLevel": ci_level,
            "bootstrapResamples": bootstrap_resamples,
        }

        resp = self.session.post(url, json=payload, timeout=self.timeout)
        if resp.status_code not in (200, 201):
            self._handle_error_response(resp)
        return resp.json()

    def submit_training_metrics(
        self,
        per_image_scores: List[Dict[str, Any]],
        loss_curves: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Submits per-image validation/training scores and optional loss curves (§5.6)."""
        # 1. Client-side contract validation
        if not per_image_scores:
            raise ValueError(
                "Data Contract §2.4 Violation: Aggregate-only submissions are rejected. "
                "'perImageScores' cannot be empty."
            )

        allowed_splits = {"train", "validation"}
        for idx, row in enumerate(per_image_scores):
            missing = {"imageId", "epoch", "metricName", "value", "split"} - set(row.keys())
            if missing:
                raise ValueError(f"Row {idx} missing required fields: {missing}")
            if row["split"] not in allowed_splits:
                raise ValueError(
                    f"Row {idx} has invalid split '{row['split']}'. "
                    f"Must be 'train' or 'validation' (§5.6). Held-out test data is forbidden."
                )

        # 2. Assemble the payload according to §5.6
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "perImageScores": per_image_scores,
        }
        if loss_curves:
            payload["lossCurves"] = loss_curves

        # 3. Assemble the envelope (§2.1)
        # Unique submissionId serves as the idempotency key for this batch
        submission_id = str(uuid.uuid4())
        envelope = {
            "submissionId": submission_id,
            "runId": self.run_id,
            "deviceId": self.device_id,
            "occurredAt": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }

        url = f"{self.base_url}/api/v1/metrics/training"

        # 4. Transmission
        resp = self.session.post(url, json=envelope, timeout=self.timeout)

        # 201: Successfully inserted; 200: Idempotent replay accepted
        if resp.status_code in (200, 201):
            data = resp.json()
            if data.get("duplicate"):
                logger.warning("Server confirmed idempotent replay (duplicate submission).")
            return data

        self._handle_error_response(resp)

    def _handle_error_response(self, resp: requests.Response) -> None:
        try:
            err_data = resp.json()
            code = err_data.get("code", "UNKNOWN_ERROR")
            message = err_data.get("message", resp.text)
            details = err_data.get("details")
        except Exception:
            code = "RAW_HTTP_ERROR"
            message = resp.text
            details = None

        logger.error(f"Results Service Error: {code} - {message}")
        raise ResultsServiceError(resp.status_code, code, message, details)