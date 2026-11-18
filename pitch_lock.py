"""Source-contour correction for monophonic converted vocals."""
import io
import math
import wave

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly


def read_mono_pcm(data):
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        if wav.getsampwidth() != 2:
            raise ValueError("Pitch lock expects 16-bit PCM")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        return samples.reshape(-1, channels).mean(axis=1).astype(np.float32) / 32768, rate


def track_pitch(samples, rate):
    divisor = math.gcd(rate, 12000)
    gained = resample_poly(samples, 12000 // divisor, rate // divisor)
    pitch, voiced, probability = librosa.pyin(
        gained, fmin=75, fmax=850, sr=12000, frame_length=512, hop_length=120,
    )
    return np.where(voiced & (probability >= .1), pitch, np.nan)


def match_phrase_dynamics(source, source_rate, rendered, rate):
    """Restore relative phrase levels while retaining the converted take's overall level."""
    def envelope(samples, sample_rate):
        hop = max(1, round(sample_rate * .02))
        starts = np.arange(0, len(samples), hop)
        counts = np.minimum(hop, len(samples) - starts)
        energy = np.add.reduceat(samples.astype(np.float64) ** 2, starts) / counts
        return (starts + counts / 2) / sample_rate, np.sqrt(energy)

    source_time, original = envelope(source, source_rate)
    times, converted = envelope(rendered, rate)
    original = np.interp(times, source_time, original)
    active = ((original > max(.001, original.max() * .04))
              & (converted > max(.001, converted.max() * .04)))
    if np.count_nonzero(active) < 5:
        return rendered
    delta = 20 * np.log10(np.maximum(original, 1e-6) / np.maximum(converted, 1e-6))
    correction = np.where(active, np.clip(delta - np.median(delta[active]), -6, 6), 0)
    gain_db = np.interp(np.arange(len(rendered)) / rate, times,
                        gaussian_filter1d(correction, sigma=2))
    adjusted = rendered * 10 ** (gain_db / 20)
    return adjusted / max(1, np.max(np.abs(adjusted)) / (32767 / 32768))


def match_source_pitch(source_data, converted_data):
    source, source_rate = read_mono_pcm(source_data)
    converted, rate = read_mono_pcm(converted_data)
    target_length = round(len(source) * rate / source_rate)
    if abs(len(converted) - target_length) > rate * .12:
        raise ValueError("Converted take duration changed too much for pitch lock")
    if len(converted) < target_length:
        converted = np.pad(converted, (0, target_length - len(converted)))
    else:
        converted = converted[:target_length]

    source_pitch = track_pitch(source, source_rate)
    converted_pitch = track_pitch(converted, rate)
    length = min(len(source_pitch), len(converted_pitch))
    ratio = source_pitch[:length] / converted_pitch[:length]
    valid = np.isfinite(ratio) & (ratio >= .7) & (ratio <= 1.42)
    source_voiced = np.isfinite(source_pitch[:length])
    minimum_frames = max(5, np.count_nonzero(source_voiced) * .45)
    if np.count_nonzero(valid) < minimum_frames:
        # Conversion can transpose a low singing voice toward the reference register.
        valid = np.isfinite(ratio) & (ratio >= .5) & (ratio <= 2)
    if np.count_nonzero(valid) < minimum_frames:
        raise ValueError("Too few aligned voiced notes could be measured")
    frame_log_ratio = np.where(valid, np.log(np.clip(ratio, .5, 2)), 0)
    smoothed_mask = gaussian_filter1d(valid.astype(np.float64), sigma=1.2)
    smoothed_log_ratio = gaussian_filter1d(frame_log_ratio, sigma=1.2)
    smoothed_log_ratio /= np.maximum(.01, smoothed_mask)
    frames = np.arange(length) * .01
    index = np.arange(target_length)
    time = index / rate
    shift = np.exp(np.interp(time, frames, smoothed_log_ratio))
    wet = np.clip(np.interp(time, frames, smoothed_mask) * 1.25 - .25, 0, 1)
    wet *= np.clip(np.abs(shift - 1) / .008, 0, 1)
    sweep = rate * .03
    phase = np.mod(.5 + np.cumsum((1 - shift) / sweep), 1)
    alt = np.mod(phase + .5, 1)
    fade = .5 - .5 * np.cos(2 * np.pi * phase)
    wet *= np.clip(np.minimum(index, target_length - index - 1) / sweep, 0, 1)
    shifted = np.interp(index - (phase - .5) * sweep, index, converted) * fade
    shifted += np.interp(index - (alt - .5) * sweep, index, converted) * (1 - fade)
    result = converted + wet * (shifted - converted)
    result = match_phrase_dynamics(source, source_rate, result, rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(np.rint(np.clip(result, -1, 32767 / 32768) * 32768).astype("<i2").tobytes())
    return buffer.getvalue()
