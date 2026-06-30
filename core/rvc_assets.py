"""RVC shared model downloads and management."""

import os
import torch
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


def detect_device(preferred: Optional[str] = None) -> torch.device:
    """Detect the best available compute device.

    Priority: MPS (Apple Silicon) → CUDA (NVIDIA) → XPU (Intel) → CPU.

    On Intel XPU systems, device selection can be controlled via environment
    variables::

        ONEAPI_DEVICE_SELECTOR=level_zero:N   # select device N (0, 1, ...)
        XPU_VISIBLE_DEVICES=N                  # alternative / additive

    Args:
        preferred: Force a specific device type ('cuda', 'mps', 'xpu', 'cpu').
    """
    if preferred:
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if preferred == "cpu":
            return torch.device("cpu")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_count = torch.xpu.device_count()
        xpu_dev = os.environ.get("ONEAPI_DEVICE_SELECTOR", os.environ.get("XPU_VISIBLE_DEVICES", "0"))
        if isinstance(xpu_dev, str):
            if "level_zero:" in xpu_dev:
                xpu_dev = xpu_dev.rsplit("level_zero:", 1)[-1].split(",")[0]
            xpu_dev = xpu_dev.split(",")[0]
            try:
                idx = int(xpu_dev)
                if 0 <= idx < device_count:
                    return torch.device(f"xpu:{idx}")
            except ValueError:
                pass
        return torch.device("xpu:0")
    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    """Empty framework-specific GPU cache."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()

BASE_DIR = Path(__file__).parent.parent / "rvc_models"
HF_REPO = "lj1995/VoiceConversionWebUI"


def _ensure_dir() -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR


def get_vocoder_path(sr: int = 40000) -> Path:
    """Download and return path to HiFi-GAN vocoder matching the target sample rate.

    Args:
        sr: Target sample rate.  If 40000, downloads G40k.pth; if 48000,
            downloads G48k.pth; if 32000, downloads G32k.pth.
    """
    filename = f"G{sr // 1000}k.pth"
    target = _ensure_dir() / filename
    if not target.exists():
        print(f"[RVC] Downloading vocoder {filename} (~73 MB)...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"pretrained/{filename}",
            local_dir=str(BASE_DIR),
        )
    return target


def get_hubert_path() -> str:
    """Return HuggingFace model ID for HuBERT content encoder."""
    return "facebook/hubert-base-ls960"


def get_pretrained_generator(path: Optional[str] = None) -> Path:
    """Download and return path to pre-trained RVC v2 generator.

    Args:
        path: Specific file name in pretrained_v2/, e.g. 'f0G48k.pth'.
              Defaults to f0G48k.pth (pitch-aware, 48kHz).
    """
    filename = path or "f0G48k.pth"
    target = _ensure_dir() / filename
    if not target.exists():
        print(f"[RVC] Downloading pre-trained model {filename}...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"pretrained_v2/{filename}",
            local_dir=str(BASE_DIR),
        )
    return target


def get_rmvpe_path() -> Path:
    """Download and return path to RMVPE pitch estimator (optional)."""
    target = _ensure_dir() / "rmvpe.pt"
    if not target.exists():
        print("[RVC] Downloading RMVPE model (~181 MB)...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename="rmvpe.pt",
            local_dir=str(BASE_DIR),
        )
    return target


def ensure_all_models(sr: int = 40000) -> dict:
    """Download all required shared models. Returns path mapping."""
    return {
        "hubert": get_hubert_path(),
        "vocoder": str(get_vocoder_path(sr)),
        "pretrained": str(get_pretrained_generator()),
    }
