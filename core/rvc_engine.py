"""RVC inference engine for voice conversion."""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.io import wavfile

logger = logging.getLogger(__name__)

_vocoder_cache: dict[int, torch.nn.Module] = {}


def _load_vocoder(sr: int, device: torch.device) -> torch.nn.Module:
    """Load (and cache) the HiFi-GAN vocoder for the given sample rate."""
    if sr in _vocoder_cache:
        vocoder = _vocoder_cache[sr]
        vocoder.to(device)
        return vocoder

    from core.rvc_assets import get_vocoder_path
    from core.rvc_models.vocoder import Generator

    vocoder_path = get_vocoder_path(sr)
    ckpt = torch.load(vocoder_path, map_location=device, weights_only=False)

    # G40k.pth has top-level "model" key with nested dec.* keys for the vocoder
    full_sd = ckpt.get("weight", ckpt.get("model", ckpt))
    dec_sd = {k: v for k, v in full_sd.items() if k.startswith("dec.")}

    n_mel = 192 if sr == 40000 else 80
    vocoder = Generator(n_mel_channels=n_mel)
    vocoder.load_state_dict(dec_sd, strict=True)

    # Fuse weight_norm (weight_v + weight_g → .weight) to avoid numerical
    # issues during inference.  Checkpoints store decomposed weight-norm
    # parameters; leaving them unfused can produce saturated tanh output.
    vocoder.remove_weight_norm()

    vocoder.eval()
    vocoder.to(device)
    _vocoder_cache[sr] = vocoder
    return vocoder


def load_voice(
    voice_dir: str,
    device_select: Optional[str] = None,
) -> dict:
    """Load a trained voice model from disk.

    Args:
        voice_dir: Path to voice directory containing v1.pth, config.json, etc.
        device_select: Force device type ('cuda', 'mps', 'xpu', 'cpu').

    Returns:
        Dict with loaded model info and paths.
    """
    voice_path = Path(voice_dir)
    config_path = voice_path / "config.json"
    pth_path = voice_path / "v1.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {voice_dir}")
    if not pth_path.exists():
        raise FileNotFoundError(f"No v1.pth found in {voice_dir}")

    with open(config_path) as f:
        config = json.load(f)

    from core.rvc_assets import detect_device

    device = detect_device(preferred=device_select)

    # Load checkpoint first to infer actual architecture (config.json may be wrong)
    ckpt = torch.load(pth_path, map_location=device)
    sd = ckpt.get("weight", ckpt.get("model", None)) or ckpt

    from core.rvc_models import build_rvc_model_from_ckpt

    net_g = build_rvc_model_from_ckpt(sd, config).to(device)
    missing, unexpected = net_g.load_state_dict(sd, strict=False)
    if missing:
        print(f"[RVC] Warning: {len(missing)} missing keys in model weights")

    net_g.eval()

    return {
        "config": config,
        "device": device,
        "net_g": net_g,
        "pth_path": str(pth_path),
    }


def convert_audio(
    source_audio: np.ndarray,
    sr: int,
    voice_info: dict,
    transpose: int = 0,
    f0_method: str = "harvest",
) -> Tuple[np.ndarray, int]:
    """Convert audio using loaded RVC model.

    Args:
        source_audio: Input audio samples (float32 or int16).
        sr: Sample rate of input audio.
        voice_info: Dict from load_voice().
        transpose: Pitch shift in semitones.
        f0_method: F0 extraction method ('harvest', 'pm').

    Returns:
        (converted_audio, sample_rate) tuple.
    """
    import faiss

    device = voice_info["device"]
    config = voice_info["config"]
    net_g = voice_info["net_g"]
    target_sr = config.get("sample_rate", 40000)
    hop_length = config.get("hop_length", 300)

    if source_audio.dtype != np.float32:
        source_audio = source_audio.astype(np.float32) / 32768.0

    print(f"[DEBUG] input audio shape={source_audio.shape}, range=[{float(source_audio.min()):.4f}, {float(source_audio.max()):.4f}], mean={float(source_audio.mean()):.4f}, std={float(source_audio.std()):.4f}")

    if sr != target_sr:
        import librosa
        source_audio = librosa.resample(source_audio, orig_sr=sr, target_sr=target_sr)

    from core.rvc_assets import get_hubert_path
    hubert_path = str(get_hubert_path())
    content_features = _extract_content(hubert_path, source_audio, target_sr, device)

    print(f"[DEBUG] HuBERT features shape={content_features.shape}, range=[{float(content_features.min()):.4f}, {float(content_features.max()):.4f}], mean={float(content_features.mean()):.4f}, std={float(content_features.std()):.4f}")

    f0 = _extract_f0(source_audio, target_sr, hop_length, f0_method)
    if transpose != 0:
        f0 *= 2 ** (transpose / 12)

    print(f"[DEBUG] F0 shape={f0.shape}, nonzero={int((f0 != 0).sum())}/{len(f0)}, range=[{float(f0.min()):.0f}, {float(f0.max()):.0f}]")

    index_path = Path(voice_info["pth_path"]).parent / "v1.index"
    if index_path.exists():
        content_features = _retrieve_features(content_features, str(index_path))

    # Align lengths: HuBERT and F0 may have different frame counts
    min_len = min(len(content_features), len(f0))
    content_features = content_features[:min_len]
    f0 = f0[:min_len]

    print(f"[DEBUG] after alignment: content={content_features.shape}, f0={f0.shape}")

    converted = _infer(net_g, content_features, f0, device, target_sr)

    return converted, target_sr


def _extract_content(hubert_path: str, audio: np.ndarray, sr: int, device) -> np.ndarray:
    """Extract HuBERT content features from audio."""
    from core.hubert_manager import HuBERTManager

    hubert = HuBERTManager.get(hubert_path, device=device)
    content = hubert.content(audio, sr=sr)
    return content


def _extract_f0(
    audio: np.ndarray, sr: int, hop_length: int, method: str = "harvest"
) -> np.ndarray:
    """Extract F0 (fundamental frequency) from audio."""
    import pyworld as pw

    frame_period = hop_length / sr * 1000
    f0, _ = pw.harvest(audio.astype(np.float64), sr, frame_period=frame_period)

    f0_coarse = np.zeros(len(f0), dtype=np.int32)
    for i in range(len(f0)):
        if f0[i] != 0:
            f0_coarse[i] = int(np.round(12 * np.log2(max(f0[i], 50) / 75))) + 1

    return f0_coarse


def _retrieve_features(content: np.ndarray, index_path: str) -> np.ndarray:
    """Retrieve similar features from FAISS index for quality improvement."""
    import faiss

    index = faiss.read_index(index_path)
    d = content.shape[1]
    query = content.reshape(-1, d).astype(np.float32)

    n_search = min(256, len(query))
    if n_search > 0:
        _, indices = index.search(query[:n_search], 1)
        retrieved = index.reconstruct_n(int(indices.flatten()[0]), int(n_search))
        blend_ratio = 0.5
        blended = query[:n_search] * (1 - blend_ratio) + retrieved * blend_ratio
        if len(query) > n_search:
            return np.concatenate([blended, query[n_search:]])
        return blended

    return content


def _infer(
    net_g,
    content: np.ndarray,
    f0: np.ndarray,
    device: torch.device,
    sr: int = 40000,
) -> np.ndarray:
    """Run RVC model inference: content + F0 → mel → waveform.

    SynthesizerTrn.infer() returns mel/cond at T_c×160 frames (upsampled by
    the generator's internal upsample layers).  The HiFi-GAN vocoder also
    upsamples by 160x, so we downsample the mel/cond to the correct frame
    count (target_audio_samples / 160) before feeding to the vocoder to avoid
    double-upsampling.
    """
    c = torch.FloatTensor(content).unsqueeze(0).to(device)
    f0_tensor = torch.LongTensor(f0).unsqueeze(0).to(device)

    with torch.no_grad():
        mel, cond = net_g.infer(c, f0=f0_tensor)  # [B, 192, T_c*160], [B, 256, T_c*160]

        print(f"[DEBUG] mel (pre-downsample) shape={mel.shape}, range=[{float(mel.min()):.4f}, {float(mel.max()):.4f}], mean={float(mel.mean()):.4f}, std={float(mel.std()):.4f}")
        print(f"[DEBUG] cond (pre-downsample) shape={cond.shape}, range=[{float(cond.min()):.4f}, {float(cond.max()):.4f}], mean={float(cond.mean()):.4f}, std={float(cond.std()):.4f}")

        # Vocoder upsamples by product(upsample_rates).  For G40k.pth: [8,5,2,2] = 160.
        vocoder = _load_vocoder(sr, device)
        vocoder_upsample_factor = 8 * 5 * 2 * 2  # 160

        # Target audio length: content has T_c frames at HuBERT rate (~50fps).
        # Audio at sr=40k for same duration = T_c * (sr / hubert_fps).
        hubert_fps = 50
        target_audio_samples = int(len(content) * sr / hubert_fps)
        target_mel_frames = max(1, target_audio_samples // vocoder_upsample_factor)

        print(f"[DEBUG] target_audio_samples={target_audio_samples}, target_mel_frames={target_mel_frames}")

        # Downsample mel and cond from T_c*160 to target_mel_frames.
        # interpolate works on the last dimension (T) for [B, C, T] input.
        mel_down = torch.nn.functional.interpolate(
            mel, size=target_mel_frames, mode="linear", align_corners=False
        )
        cond_down = torch.nn.functional.interpolate(
            cond, size=target_mel_frames, mode="linear", align_corners=False
        )

        print(f"[DEBUG] mel (post-downsample) shape={mel_down.shape}")
        print(f"[DEBUG] cond (post-downsample) shape={cond_down.shape}")

        waveform = vocoder.infer(mel_down, cond=cond_down)  # [B, 1, T_audio]

    wav_np = waveform[0, 0].cpu().numpy()
    print(f"[DEBUG] waveform shape={wav_np.shape}, range=[{float(wav_np.min()):.4f}, {float(wav_np.max()):.4f}], mean={float(wav_np.mean()):.4f}, std={float(wav_np.std()):.4f}")

    return wav_np


def unload_voice(voice_info: dict):
    """Unload voice model and free GPU memory."""
    if "net_g" in voice_info:
        del voice_info["net_g"]
    from core.rvc_assets import empty_cache

    device = voice_info.get("device")
    if device:
        empty_cache(device)