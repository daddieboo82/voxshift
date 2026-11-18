"""Offline deployment-contract test. Run with python -m unittest discover -s tests."""
import unittest
from pathlib import Path


class GpuDeploymentTests(unittest.TestCase):
    def test_gpu_override_requests_gpu_and_cuda_model(self):
        contents = (Path(__file__).resolve().parents[1] / "deploy" / "compose.gpu.yaml").read_text()
        self.assertIn("VOXSHIFT_DEVICE: cuda", contents)
        self.assertIn("driver: nvidia", contents)
        self.assertIn("capabilities: [gpu]", contents)
        self.assertIn("whl/cu124", contents)


if __name__ == "__main__":
    unittest.main()
