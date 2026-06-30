"""RVC chunked voice conversion.

Splits long audio into ~30s chunks for RVC inference to manage memory usage.
Each chunk is converted independently and concatenated with crossfade blending.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch


def chunked_voice_conversion(
    source_audio: np.ndarray,
    sr: int,
    voice_name: str,
    output_path: str = "output.wav",
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Convert *source_audio* using RVC in memory-safe chunks.

    Parameters
    ----------
    source_audio : np.ndarray
        Input audio samples (float32 or int16).
    sr : int
        Sample rate of input audio.
    voice_name : str
        Name of trained voice in voices/ directory.
    output_path : str
        Output WAV file path.
    chunk_seconds : float
        Chunk duration in seconds. Default 30s for RVC.
    overlap_seconds : float
        Overlap between chunks for crossfade blending.

    Returns
    -------
    tuple[np.ndarray, int]
        (converted_audio, sample_rate)
    """
    from core.rvc_engine import load_voice, convert_audio as rvc_convert

    _start = time.perf_counter()

    # Normalize input
    if source_audio.dtype != np.float32:
        source_audio = source_audio.astype(np.float32) / 32768.0

    n_samples = len(source_audio)
    chunk_size = int(chunk_seconds * sr)
    overlap_size = int(overlap_seconds * sr)

    # Short-circuit if audio fits in one chunk
    if n_samples <= chunk_size:
        voice_info = load_voice(str(Path("voices") / voice_name))
        converted, target_sr = rvc_convert(source_audio, sr, voice_info)
        elapsed = time.perf_counter() - _start
        print(f"[chunked_convert] Single chunk in {elapsed:.1f}s")
        return converted, target_sr

    # Chunked processing with crossfade overlap
    hop_size = chunk_size - overlap_size
    parts: list[np.ndarray] = []
    pos = 0
    from core.rvc_assets import detect_device, empty_cache as rvc_empty_cache

    device = detect_device()
    voice_info = load_voice(str(Path("voices") / voice_name))

    while pos < n_samples:
        end = min(pos + chunk_size, n_samples)
        chunk = source_audio[pos:end]

        converted_chunk, target_sr = rvc_convert(chunk, sr, voice_info)

        rvc_empty_cache(device)

        parts.append(converted_chunk)
        pos += hop_size

    # Concatenate with crossfade blending for overlaps
    if len(parts) == 1:
        converted = parts[0]
    else:
        converted = _crossfade_concat(parts, overlap_size, target_sr or sr)

    elapsed = time.perf_counter() - _start
    print(f"[chunked_convert] {len(parts)} chunks in {elapsed:.1f}s")

    from scipy.io import wavfile
    wavfile.write(output_path, int(target_sr), converted.astype(np.float32))

    return converted, target_sr


def _crossfade_concat(parts: list[np.ndarray], overlap_size: int, sr: int) -> np.ndarray:
    """Concatenate audio parts with crossfade blending at overlaps."""
    if len(parts) <= 1:
        return np.concatenate(parts)

    # Build fade curves for overlap regions
    fade = np.linspace(0, 1, overlap_size, dtype=np.float32)

    result_parts = [parts[0]]
    for i in range(1, len(parts)):
        prev = result_parts[-1]
        curr = parts[i]

        if len(curr) <= overlap_size:
            # Current chunk is shorter than overlap — just append
            result_parts.append(curr)
            continue

        # Crossfade the overlap region
        blended = prev[-overlap_size:] * (1 - fade[:min(overlap_size, len(prev))]) + \
                  curr[:overlap_size] * fade[:min(overlap_size, len(prev))]

        # Replace end of previous with blended, then append rest of current
        result_parts[-1] = np.concatenate([prev[:-overlap_size], blended])
        result_parts.append(curr[overlap_size:])

    return np.concatenate(result_parts)
