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
        # Use torchaudio for resampling (replaces librosa to avoid numba/llvmlite dep)
        import torch
        import torchaudio
        tensor = torch.from_numpy(source_audio).unsqueeze(0)  # [1, T]
        source_audio = torchaudio.functional.resample(
            tensor, orig_freq=sr, new_freq=target_sr,
        ).squeeze().numpy()

    from core.rvc_assets import get_hubert_path
    hubert_path = str(get_hubert_path())
    content_features = _extract_content(hubert_path, source_audio, target_sr, device)

    print(f"[DEBUG] HuBERT features shape={content_features.shape}, range=[{float(content_features.min()):.4f}, {float(content_features.max()):.4f}], mean={float(content_features.mean()):.4f}, std={float(content_features.std()):.4f}")

    f0 = _extract_f0(source_audio, target_sr, hop_length, f0_method)
    if transpose != 0:
        f0 = (f0 * (2 ** (transpose / 12))).astype(np.int32)

    print(f"[DEBUG] F0 shape={f0.shape}, nonzero={int((f0 != 0).sum())}/{len(f0)}, range=[{float(f0.min()):.0f}, {float(f0.max()):.0f}]")

    index_path = Path(voice_info["pth_path"]).parent / "v1.index"
    if index_path.exists():
        content_features = _retrieve_features(content_features, str(index_path))

    # Align lengths: resample HuBERT content features to match F0 frame rate.
    # HuBERT runs at ~50fps, F0/mel at ~133fps — same as training pipeline.
    if len(content_features) != len(f0):
        c_torch = torch.FloatTensor(content_features).T.unsqueeze(0)  # [1, 768, T_c]
        content_features = torch.nn.functional.interpolate(
            c_torch, size=len(f0), mode="linear", align_corners=False
        ).squeeze(0).transpose(0, 1).cpu().numpy()  # [T_m, 768]

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


def _retrieve_features(content: np.ndarray, index_path: str, k: int = 3, blend_ratio: float = 0.7) -> np.ndarray:
    """Retrieve similar features from FAISS index for quality improvement.

    Processes ALL content frames (not just the first 256) by blending
    each frame's query feature with its k-nearest neighbors from the
    training index.  Higher ``blend_ratio`` = more training-speaker
    timbre; lower = more source content preservation.

    Args:
        content: Source content features [T, 768].
        index_path: Path to FAISS index file.
        k: Number of nearest neighbors to average (default 3).
        blend_ratio: Weight of retrieved features (0.0 = no retrieval,
            1.0 = fully replaced by training features, default 0.7).

    Returns:
        Blended content features with same shape as input.
    """
    import faiss

    index = faiss.read_index(index_path)
    d = content.shape[1]
    query = content.reshape(-1, d).astype(np.float32)

    if len(query) == 0:
        return content

    # Search k-nearest neighbors for every frame
    # I shape: [T, k] — k neighbor indices for each of T query frames
    D, I = index.search(query, k)
    T, _ = I.shape

    # Reconstruct all k neighbors for each frame and average them
    retrieved = np.zeros_like(query)
    for t in range(T):
        vecs = np.array([index.reconstruct(int(I[t, j])) for j in range(k)])
        retrieved[t] = vecs.mean(axis=0)

    # Blend original with retrieved
    blended = query * (1.0 - blend_ratio) + retrieved * blend_ratio

    return blended


def _infer(
    net_g,
    content: np.ndarray,
    f0: np.ndarray,
    device: torch.device,
    sr: int = 40000,
) -> np.ndarray:
    """Run RVC model inference: content + F0 → mel (+ audio).

    End-to-end mode (``net_g.use_decoder=True``, per config.json ``use_decoder:`` True):
        The generator's embedded HiFi-GAN ``dec`` produces audio directly from
        the predicted mel.  No external vocoder is loaded.  This is the only
        working inference path after the G40k.pth forensic investigation (see
        ``docs/VOCODER_INVESTIGATION.md``).

    Legacy mel-only mode (``net_g.use_decoder=False``):
        Returns ``(mel, cond)`` — mean the caller picks up an external vocoder.
        This path is preserved for the superseded ``am_jefferson/v1.pth``
        checkpoint but is expected to produce saturated audio.
    """
    c = torch.FloatTensor(content).unsqueeze(0).to(device)
    f0_tensor = torch.LongTensor(f0).unsqueeze(0).to(device)

    with torch.no_grad():
        mel, secondary = net_g.infer(c, f0=f0_tensor)

        print(f"[DEBUG] mel shape={mel.shape}, range=[{float(mel.min()):.4f}, {float(mel.max()):.4f}], "
              f"mean={float(mel.mean()):.4f}, std={float(mel.std()):.4f}")

        if getattr(net_g, "use_decoder", False) and secondary is not None:
            # ─── End-to-end path: secondary is the audio output ───
            audio = secondary  # [B, 1, T_audio]
            print(f"[DEBUG] end-to-end audio: shape={audio.shape}, "
                  f"mean={float(audio.mean()):.4f}, std={float(audio.std()):.4f}, "
                  f"sat={float((audio.abs() > 0.99).float().mean()) * 100:.1f}%")
            wav_np = audio[0, 0].cpu().numpy()
        else:
            # ─── Legacy mel-only path: external G40k.pth vocoder ───
            # NOTE: saturated output; kept only for regression testing.
            # See docs/VOCODER_INVESTIGATION.md.
            cond = secondary
            print(f"[DEBUG] legacy cond shape={cond.shape}")
            vocoder = _load_vocoder(sr, device)
            audio = vocoder.infer(mel, cond=cond)
            wav_np = audio[0, 0].cpu().numpy()

    print(f"[DEBUG] final waveform: shape={wav_np.shape}, "
          f"range=[{float(wav_np.min()):.4f}, {float(wav_np.max()):.4f}], "
          f"mean={float(wav_np.mean()):.4f}, std={float(wav_np.std()):.4f}")

    return wav_np


def unload_voice(voice_info: dict):
    """Unload voice model and free GPU memory."""
    if "net_g" in voice_info:
        del voice_info["net_g"]
    from core.rvc_assets import empty_cache

    device = voice_info.get("device")
    if device:
        empty_cache(device)