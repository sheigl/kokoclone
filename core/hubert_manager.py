"""HuBERT feature extractor wrapper for RVC."""

import torch


class HuBERTManager:
    """Manages loading and inference of HuBERT content encoder.

    Uses transformers library (same as training) to ensure 768-dim output
    matches the trained model's expected input dimension.
    """

    _instance = None
    _model = None
    _device = None

    @classmethod
    def get(cls, hubert_path: str, device=None):
        if cls._model is None or device != cls._device:
            cls._load_model(hubert_path, device)
        return cls._instance

    @classmethod
    def _load_model(cls, hubert_path: str, device=None):
        from core.rvc_assets import detect_device

        if device is None:
            device = detect_device()

        from transformers import AutoModel

        model = AutoModel.from_pretrained(hubert_path).to(device).eval()

        cls._model = model
        cls._device = device
        cls._instance = cls(model, device)

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def content(self, audio=None, sr=16000, wav_path=None):
        """Extract HuBERT content features.

        Args:
            audio: numpy array of audio samples (float32 or int16).
            sr: Sample rate of input audio.
            wav_path: Optional path to WAV file (alternative to audio tensor).

        Returns:
            Content features tensor [T, 768].
        """
        import numpy as np

        if wav_path is not None:
            import librosa

            audio, sr = librosa.load(wav_path, sr=sr)

        # Normalize to [-1, 1]
        if audio.dtype == np.int16 or audio.dtype == np.int32:
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

        # Resample to 16kHz (HuBERT expects this)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        x = torch.FloatTensor(audio).to(self.device)
        while x.dim() < 2:
            x = x.unsqueeze(0)

        features = self.model(input_values=x).last_hidden_state  # [B, T, 768]
        content = features.squeeze(0).cpu().numpy()

        return content
