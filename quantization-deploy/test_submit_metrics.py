import unittest
from submit_metrics import build_payload
import uuid
import datetime

class TestSubmitMetrics(unittest.TestCase):
    def test_payload_schema_compliance(self):
        # 1. Build payload
        envelope = build_payload("test-run-id", device_id="test-device", latency_ms=20.0)
        
        # 2. Validate Envelope
        self.assertIn("submissionId", envelope)
        self.assertTrue(isinstance(uuid.UUID(envelope["submissionId"]), uuid.UUID))
        
        self.assertEqual(envelope["runId"], "test-run-id")
        self.assertEqual(envelope["deviceId"], "test-device")
        
        # Validate timestamp format
        datetime.datetime.fromisoformat(envelope["occurredAt"])
        
        # 3. Validate Denoise Payload (§5.4)
        payload = envelope["payload"]
        self.assertIn("modelVariant", payload)
        self.assertIsInstance(payload["modelVariant"], str)
        self.assertIn("modelFileSizeBytes", payload)
        self.assertIsInstance(payload["modelFileSizeBytes"], int)
        
        # 4. Validate images array
        self.assertIn("images", payload)
        self.assertGreater(len(payload["images"]), 0)
        
        for img in payload["images"]:
            self.assertIn("imageId", img)
            self.assertIsInstance(img["imageId"], str)
            
            self.assertIn("precision", img)
            self.assertIn(img["precision"], ["fp32", "int8"])
            
            self.assertIn("psnr", img)
            self.assertIsInstance(img["psnr"], float)
            
            self.assertIn("ssim", img)
            self.assertIsInstance(img["ssim"], float)
            
            self.assertIn("latencyMs", img)
            self.assertIsInstance(img["latencyMs"], float)
            
            self.assertIn("discardRaceEvent", img)
            self.assertIsInstance(img["discardRaceEvent"], bool)
            
            self.assertIn("timeoutEvent", img)
            self.assertIsInstance(img["timeoutEvent"], bool)

if __name__ == '__main__':
    unittest.main()
