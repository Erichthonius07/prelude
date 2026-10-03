import unittest
from unittest.mock import patch, mock_open, MagicMock
from submit_metrics import build_payload, run_cli
import json

class TestSubmitMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics_data = {
            "model_variant": "cnn-v1",
            "file_sizes": {"int8_tflite_bytes": 1024},
            "dataset": "sidd",
            "input_domain": "sRGB",
            "input_size": "256x256",
            "per_image": [
                {"imageId": "img1", "precision": "fp32", "psnr_gt": 40.0, "ssim_gt": 0.99},
                {"imageId": "img1", "precision": "int8", "psnr_gt": 39.5, "ssim_gt": 0.98}
            ]
        }
        
        self.latency_data = {
            "metadata": {
                "numThreads": 4,
                "delegate": "XNNPACK requested"
            },
            "events": [
                {"imageId": "img1", "precision": "fp32", "latencyMs": 150.0, "discardRaceEvent": False, "timeoutEvent": False},
                {"imageId": "img1", "precision": "int8", "latencyMs": 50.0, "discardRaceEvent": False, "timeoutEvent": False}
            ]
        }
        
        self.mock_files = {
            "metrics.json": json.dumps(self.metrics_data),
            "latency.json": json.dumps(self.latency_data)
        }
        
        self.base_args = [
            "--run-id", "smoke-test", 
            "--planned-test-n", "1", 
            "--planned-comparisons", "1", 
            "--base-url", "http://mock",
            "--metrics-json", "metrics.json",
            "--latency-json", "latency.json"
        ]

    def mock_open_impl(self, filename, *args, **kwargs):
        if filename in self.mock_files:
            return mock_open(read_data=self.mock_files[filename])()
        return mock_open(read_data="{}")()

    def test_value_round_trip(self):
        with patch("builtins.open", side_effect=self.mock_open_impl):
            envelope = build_payload("smoke-test", "metrics.json", "latency.json", device_id="test-device")
            
        payload = envelope["payload"]
        self.assertEqual(payload["modelVariant"], "cnn-v1")
        self.assertEqual(payload["inputDomain"], "sRGB")
        self.assertNotIn("metadata", payload, "Local metadata must not be leaked into wire payload")
        
        images = payload["images"]
        self.assertEqual(len(images), 2)
        
        fp32_img = next(img for img in images if img["precision"] == "fp32")
        self.assertEqual(fp32_img["psnr"], 40.0)
        self.assertEqual(fp32_img["ssim"], 0.99)
        self.assertEqual(fp32_img["latencyMs"], 150.0)

    @patch("submit_metrics.requests.get")
    def test_subset_of_manifest(self, mock_get):
        # imageIds not in manifest
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"imageIds": ["other_img"]}
        
        with patch("builtins.open", side_effect=self.mock_open_impl):
            ret = run_cli(self.base_args + ["--send"])
            self.assertEqual(ret, 1)

    @patch("submit_metrics.requests.get")
    def test_partial_coverage_warns_and_refuses_send(self, mock_get):
        # manifest has more images
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"imageIds": ["img1", "img2"]}
        
        with patch("builtins.open", side_effect=self.mock_open_impl):
            ret = run_cli(self.base_args + ["--send"])
            self.assertEqual(ret, 1)

    @patch("submit_metrics.requests.get")
    def test_both_precisions_required(self, mock_get):
        # Remove int8 from metrics
        self.metrics_data["per_image"].pop()
        self.mock_files["metrics.json"] = json.dumps(self.metrics_data)
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"imageIds": ["img1"]}
        
        with patch("builtins.open", side_effect=self.mock_open_impl):
            ret = run_cli(self.base_args + ["--send"])
            self.assertEqual(ret, 1)

    def test_srgb_requires_smoke_prefix(self):
        with patch("builtins.open", side_effect=self.mock_open_impl):
            args = list(self.base_args)
            args[1] = "prod-run" # not smoke-
            ret = run_cli(args + ["--send"])
            self.assertEqual(ret, 1)

    @patch("submit_metrics.requests.get")
    def test_manifest_unreachable_refuses_send(self, mock_get):
        mock_get.return_value.status_code = 404
        
        with patch("builtins.open", side_effect=self.mock_open_impl):
            ret = run_cli(self.base_args + ["--send"])
            self.assertEqual(ret, 1)

    @patch("submit_metrics.requests.get")
    def test_existing_run_different_planned_test_n(self, mock_get):
        def mock_requests_get(url, *args, **kwargs):
            m = MagicMock()
            m.status_code = 200
            if "manifest" in url:
                m.json.return_value = {"imageIds": ["img1"]}
            else:
                m.json.return_value = {"plannedTestN": 99} # differs from 1
            return m
            
        mock_get.side_effect = mock_requests_get
        
        with patch("builtins.open", side_effect=self.mock_open_impl):
            ret = run_cli(self.base_args + ["--send"])
            self.assertEqual(ret, 1)

    @patch("submit_metrics.requests.post")
    @patch("submit_metrics.requests.get")
    def test_dry_run_is_default(self, mock_get, mock_post):
        mock_get.return_value.status_code = 404 # unreachable
        with patch("builtins.open", side_effect=self.mock_open_impl):
            ret = run_cli(self.base_args)
            self.assertEqual(ret, 0)
            mock_post.assert_not_called()

if __name__ == '__main__':
    unittest.main()
