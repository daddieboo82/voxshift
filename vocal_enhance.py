"""Conservative vocal polish for clean, isolated PCM WAV takes.

This is deterministic audio processing, not AI voice generation or restoration.
It preserves the source voice and does not imitate a reference speaker.
"""
import io
import wave

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

MIN_LEVEL = 1e-7


def enhance_vocal(data: bytes) -> bytes:
    """High-pass rumble, tame peaks, and bring a clean vocal to a safe level."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.getnframes()
        if width != 2 or channels not in (1, 2) or not 8000 <= sample_rate <= 48000:
            raise ValueError("Expected 16-bit mono/stereo PCM WAV")
        original = wav.readframes(frames)
        if len(original) != frames * channels * width:
            raise ValueError("Truncated PCM data")

    samples = np.frombuffer(original, dtype="<i2").astype(np.float64)
    samples = samples.reshape(-1, channels) / 32768.0
    if not len(samples) or not np.all(np.isfinite(samples)):
        raise ValueError("Missing or invalid samples")

    if np.max(np.abs(samples)) < 1e-5:
        return data  # Do not amplify empty or silent takes.

    # Low-frequency rumble cut, without eliminating the body of deep voices.
    highpass_hz = min(70.0, sample_rate * .01)
    sos = butter(2, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, samples, axis=0)

    # Link stereo gain reduction across channels to preserve stereo balance.
    envelope_power = np.mean(filtered * filtered, axis=1)
    window = max(1, int(sample_rate * .025))
    rms = np.sqrt(np.maximum(0.0, uniform_filter1d(envelope_power, window)))
    threshold = 10 ** (-19 / 20)
    above = np.maximum(rms / threshold, 1.0)
    gain = above ** (1 / 2.7 - 1.0)
    # Smooth compressor action rather than reacting instantly to every sample.
    smooth_window = max(1, int(sample_rate * .012))
    gain = uniform_filter1d(gain, smooth_window)
    processed = filtered * gain[:, None]

    # Gentle gain-up, capped to avoid excessive amplification of low-level noise.
    active = rms > 10 ** (-43 / 20)
    if np.any(active):
        active_level = np.sqrt(np.mean(np.square(processed[active])))
        desired = 10 ** (-17 / 20)
        makeup = min(10 ** (6 / 20), desired / max(active_level, MIN_LEVEL))
        processed *= makeup

    # Peak guard prevents clipping and preserves PCM compatibility.
    peak = np.max(np.abs(processed))
    if peak > .92:
        processed *= .92 / peak
    pcm = np.rint(np.clip(processed, -1.0, 32767 / 32768) * 32768).astype("<i2")

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((channels, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm.tobytes())
    return output.getvalue()
