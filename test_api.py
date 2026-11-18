import io
import unittest
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient

from app import create_app


def make_wav(seconds):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * (16000 * seconds))
    return output.getvalue()


class StandInEngine:
    def synthesize(self, text, reference):
        assert text == "Studio voice check" and reference == make_wav(3)
        return make_wav(1)

    def convert(self, source, reference):
        assert reference == make_wav(3)
        return source


class ProviderAPI(unittest.TestCase):
    def test_auth_consent_validation_and_wav(self):
        key = "test-key-" + "x" * 32
        headers = {"Authorization": f"Bearer {key}"}
        source = make_wav(1)
        files = {"source": ("source.wav", source), "reference": ("target.wav", make_wav(3))}
        with patch.dict("os.environ", {"VOXSHIFT_API_KEY": key}), TestClient(create_app(StandInEngine)) as client:
            self.assertEqual(client.get("/healthz").json(), {"status": "ready"})
            self.assertEqual(client.get("/v1/capabilities").status_code, 401)
            self.assertEqual(client.get("/v1/capabilities", headers=headers).json(),
                             {"modes": ["speech", "convert", "sing"]})
            self.assertEqual(client.post("/v1/convert", files=files).status_code, 401)
            self.assertEqual(client.post("/v1/convert", files=files, headers=headers).status_code, 422)
            rejected = client.post("/v1/convert", files={**files, "reference": ("bad.wav", b"invalid")},
                                   headers=headers, data={"consent": "confirmed"})
            self.assertEqual(rejected.status_code, 422)
            converted = client.post("/v1/convert", files=files, headers=headers, data={"consent": "confirmed"})
            self.assertEqual(converted.status_code, 200)
            self.assertEqual(converted.headers["content-type"], "audio/wav")
            self.assertEqual(converted.headers["cache-control"], "no-store")
            self.assertEqual(converted.content, source)
            speech = client.post("/v1/speech", files={"reference": files["reference"]},
                                 headers=headers, data={"consent": "confirmed", "text": "Studio voice check"})
            self.assertEqual(speech.status_code, 200)
            self.assertEqual(speech.content, source)
            blank = client.post("/v1/speech", files={"reference": files["reference"]},
                                headers=headers, data={"consent": "confirmed", "text": " "})
            self.assertEqual(blank.status_code, 422)

    def test_background_job_returns_before_inference_and_is_owner_scoped(self):
        started = Event()
        release = Event()
        key = "test-key-" + "x" * 32
        owner = {"Authorization": f"Bearer {key}", "X-VoxShift-Owner": "artist-one"}

        class BlockingEngine(StandInEngine):
            def synthesize(self, text, reference):
                started.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Test gate expired")
                return super().synthesize(text, reference)

        with patch.dict("os.environ", {"VOXSHIFT_API_KEY": key}), TestClient(create_app(BlockingEngine)) as client:
            request = {
                "headers": owner,
                "files": {"reference": ("reference.wav", make_wav(3))},
                "data": {"consent": "confirmed", "text": "Studio voice check"},
            }
            accepted = client.post("/v1/jobs/speech", **request)
            self.assertEqual(accepted.status_code, 202)
            path = "/v1/jobs/" + accepted.json()["jobId"]
            try:
                self.assertTrue(started.wait(timeout=2))
                self.assertEqual(client.get(path, headers=owner).json()["status"], "running")
                self.assertEqual(client.get(path, headers={**owner, "X-VoxShift-Owner": "artist-two"}).status_code, 404)
                self.assertEqual(client.get(path).status_code, 401)
                self.assertEqual(client.post("/v1/jobs/speech", **request).status_code, 503)
            finally:
                release.set()
            deadline = monotonic() + 2
            while monotonic() < deadline:
                result = client.get(path, headers=owner)
                if result.headers.get("content-type") == "audio/wav":
                    break
                sleep(0.01)
            self.assertEqual(result.content, make_wav(1))

    def test_singing_job_dispatches_to_melody_engine(self):
        seen = Event()
        key = "test-key-" + "x" * 32
        owner = {"Authorization": f"Bearer {key}", "X-VoxShift-Owner": "artist-one"}

        class SingingEngine(StandInEngine):
            def convert_singing(self, source, reference):
                seen.set()
                return super().convert(source, reference)

        with patch.dict("os.environ", {"VOXSHIFT_API_KEY": key}), TestClient(create_app(SingingEngine)) as client:
            accepted = client.post("/v1/jobs/sing", headers=owner, files={
                "source": ("source.wav", make_wav(1)),
                "reference": ("reference.wav", make_wav(3)),
            }, data={"consent": "confirmed"})
            self.assertEqual(accepted.status_code, 202)
            self.assertTrue(seen.wait(timeout=2))


if __name__ == "__main__":
    unittest.main()
