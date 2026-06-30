"""Audio encoding utilities: numpy → bytes in various formats."""

from __future__ import annotations

import io
import shutil
import subprocess
from typing import Literal

import numpy as np
import soundfile as sf

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

_CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

_FILE_EXTENSIONS: dict[str, str] = {
    "mp3": "mp3",
    "opus": "opus",
    "aac": "aac",
    "flac": "flac",
    "wav": "wav",
    "pcm": "pcm",
}


def content_type(fmt: AudioFormat) -> str:
    return _CONTENT_TYPES.get(fmt, "application/octet-stream")


def file_extension(fmt: AudioFormat) -> str:
    return _FILE_EXTENSIONS.get(fmt, "bin")


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def encode_audio(samples: np.ndarray, sample_rate: int, fmt: AudioFormat) -> bytes:
    """Encode a float32/ int16 numpy 1-D array into *fmt* bytes.

    *samples* is a 1-D float32 numpy array (values in [-1, 1]).
    Returns the encoded bytes.
    """
    # Ensure float32 1-D
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)

    if fmt == "pcm":
        # Raw 16-bit little-endian samples, no header.
        return np.clip(audio * 32767, -32768, 32767).astype("<i2").tobytes()

    if fmt in ("wav", "flac"):
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format=_sf_format(fmt))
        return buf.getvalue()

    # Compressed formats (mp3, opus, aac) — encode to WAV first, then use ffmpeg.
    if not _has_ffmpeg():
        # Fallback: return WAV if ffmpeg is not installed.
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        return buf.getvalue()

    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sample_rate, format="WAV")
    wav_bytes = wav_buf.getvalue()

    return _ffmpeg_encode(wav_bytes, sample_rate, fmt)


def _sf_format(fmt: AudioFormat) -> str:
    if fmt == "wav":
        return "WAV"
    if fmt == "flac":
        return "FLAC"
    return "WAV"


def _ffmpeg_encode(wav_bytes: bytes, sample_rate: int, fmt: AudioFormat) -> bytes:
    """Convert WAV bytes → target format via ffmpeg subprocess."""
    codec_map = {
        "mp3": ["-codec:a", "libmp3lame", "-b:a", "128k"],
        "opus": ["-codec:a", "libopus", "-b:a", "128k"],
        "aac": ["-codec:a", "aac", "-b:a", "128k"],
    }
    container_map = {
        "mp3": "mp3",
        "opus": "ogg",
        "aac": "adts",
    }

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "wav",
        "-ar", str(sample_rate),
        "-i", "pipe:0",
        "-f", container_map.get(fmt, fmt),
        *codec_map.get(fmt, []),
        "pipe:1",
    ]

    result = subprocess.run(
        cmd,
        input=wav_bytes,
        capture_output=True,
    )
    return result.stdout
