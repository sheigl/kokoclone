"""Test XTTS v2 → RVC pipeline quality on CPU.
Usage: python tools/test_xtts_rvc.py [voice_name]
       voice_name: 'am_jefferson' (default) or 'am_welch'
"""

import sys, os
voice_name = sys.argv[1] if len(sys.argv) > 1 else "am_jefferson"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Monkey-patch: transformers 5.x removed isin_mps_friendly; torch.isin works on MPS natively now.
import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
    import torch
    transformers.pytorch_utils.isin_mps_friendly = torch.isin

from TTS.api import TTS
import soundfile as sf
import numpy as np
from core.rvc_engine import load_voice, convert_audio

TEXT = (
    "Doors and corners, kid. That's where they get you. "
    "Humans are too big for the doors they build themselves. "
    "They have to leave them open, or hinge them just right."
)

# Reference speaker wav for XTTS (voice cloning).
ref_dir = os.path.expanduser(f"~/code/kokoclone-train/training_temp/{voice_name}/1_16k_wavs")
REF_WAV = os.path.join(ref_dir, "0_0.wav")
if not os.path.exists(REF_WAV):
    files = sorted(os.listdir(ref_dir))
    REF_WAV = os.path.join(ref_dir, files[0]) if files else None

# ─── Step 1: XTTS generates source audio on CPU ───
print(f"[1/3] Loading XTTS v2 on CPU (voice={voice_name})...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

print("[2/3] Generating speech...")
os.makedirs("/tmp/xtts_test", exist_ok=True)
out_path = "/tmp/xtts_test/source.wav"
print(f"  Reference: {REF_WAV}")
tts.tts_to_file(text=TEXT, speaker_wav=REF_WAV, language="en", file_path=out_path)
source, sr = sf.read(out_path, dtype="float32")
print(f"  Source: {len(source)} samples @ {sr}Hz, {len(source)/sr:.2f}s")

# ─── Step 2: RVC conversion on CPU ───
print("[3/3] Running RVC conversion on CPU...")
voice_dir = os.path.expanduser(f"~/code/kokoclone-train/voices/{voice_name}")
voice = load_voice(voice_dir, device_select="cpu")
converted, out_sr = convert_audio(source, sr, voice)

out_wav = f"tools/test_xtts_rvc_{voice_name}.wav"
sf.write(out_wav, converted, out_sr)
print(f"\nDone → {out_wav}")
print(f"  Output: {len(converted)} samples @ {out_sr}Hz, {len(converted)/out_sr:.2f}s")
print(f"  Range: [{converted.min():.4f}, {converted.max():.4f}]")
print(f"  Std:   {converted.std():.4f}")
print(f"  Sat:   {(np.abs(converted) > 0.99).mean() * 100:.1f}%")
