"""Offline tests for VoxShift vocal polish; no neural weights are required."""
import io
import unittest
import wave

import numpy as np

from vocal_enhance import enhance_vocal


def make_wav(samples, rate=24000, channels=1):
    samples = np.asarray(samples, dtype=np.float64)
    audio = np.rint(np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(audio.tobytes())
    return output.getvalue()


class VocalEnhanceTests(unittest.TestCase):
    def test_preserves_length_rate_and_clean_output(self):
        rate = 24000
        time = np.arange(rate) / rate
        source = .18 * np.sin(2 * np.pi * 180 * time)
        result = enhance_vocal(make_wav(source))
        with wave.open(io.BytesIO(result), "rb") as stream:
            self.assertEqual(stream.getframerate(), rate)
            self.assertEqual(stream.getnframes(), len(source))
            self.assertEqual(stream.getnchannels(), 1)
            audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        self.assertTrue(np.any(audio))
        self.assertLessEqual(np.max(np.abs(audio.astype(np.int32))), 32767)

    def test_silent_take_not_amplified(self):
        silence = make_wav(np.zeros(24000))
        self.assertEqual(enhance_vocal(silence), silence)

    def test_stereo_output_keeps_two_channels(self):
        time = np.arange(8000) / 8000
        samples = np.column_stack((.11 * np.sin(2 * np.pi * 200 * time),
                                   .05 * np.sin(2 * np.pi * 200 * time)))
        out = enhance_vocal(make_wav(samples, 8000, channels=2))
        with wave.open(io.BytesIO(out), "rb") as stream:
            self.assertEqual(stream.getnchannels(), 2)
            self.assertEqual(stream.getnframes(), len(samples))

    def test_rejects_corrupted_audio(self):
        with self.assertRaises((ValueError, EOFError, wave.Error)):
            enhance_vocal(b"garbage")


if __name__ == "__main__":
    unittest.main()
