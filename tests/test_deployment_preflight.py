"""No-network configuration checks: python -m unittest discover -s tests."""
import tempfile
import unittest
from pathlib import Path
from scripts.deployment_preflight import parse_env, validate


class DeploymentPreflightTests(unittest.TestCase):
    def test_example_is_rejected(self):
        sample = {
            "VOXSHIFT_DOMAIN": "voice.example.com",
            "VOXSHIFT_API_KEY": "REPLACE_WITH_A_RANDOM_32_PLUS_CHARACTER_SECRET",
            "VOXSHIFT_DEVICE": "cpu",
        }
        self.assertGreaterEqual(len(validate(sample)), 2)

    def test_valid_cpu_environment(self):
        values = {
            "VOXSHIFT_DOMAIN": "voice.music-platform.org",
            "VOXSHIFT_API_KEY": "VA2fju7rjQw3hL8DZ9bM0Kt4N6pc7YTz1anX",
            "VOXSHIFT_DEVICE": "cpu",
        }
        self.assertEqual(validate(values), [])

    def test_cuda_cpu_wheels_mismatch(self):
        values = {
            "VOXSHIFT_DOMAIN": "voice.music-platform.org",
            "VOXSHIFT_API_KEY": "VA2fju7rjQw3hL8DZ9bM0Kt4N6pc7YTz1anX",
            "VOXSHIFT_DEVICE": "cuda",
            "VOXSHIFT_TORCH_INDEX": "https://download.pytorch.org/whl/cpu",
        }
        self.assertTrue(any("CPU" in error for error in validate(values)))

    def test_invalid_env_line_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Path(temp) / ".env"
            env.write_text("VOXSHIFT_DOMAIN=voice.music-platform.org\nBROKENLINE\n")
            with self.assertRaises(ValueError):
                parse_env(env)


if __name__ == "__main__":
    unittest.main()
