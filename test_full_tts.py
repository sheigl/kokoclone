#!/usr/bin/env python3
"""End-to-end TTS + RVC voice conversion test (minimal memory footprint)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import torch

# 1. Generate speech with Kokoro + apply am_jefferson voice conversion
print("[1/2] Generating TTS with am_jefferson voice conversion...")
from core.cloner import KokoClone

kc = KokoClone()  # Auto-detect device (XPU if available)

output = kc.generate(
    text="Hello, this is a test of the Jefferson voice model.",
    lang="en",
    voice_name="am_jefferson",
    output_path="test_output_jefferson.wav",
    speed=1.0,
)
print(f"  Saved: test_output_jefferson.wav")

# Free Kokoro + RVC models from VRAM before reading stats
del kc
if hasattr(torch.xpu, 'empty_cache'):
    torch.xpu.empty_cache()
elif hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

# 2. Load and verify the output stats
print("[2/2] Verifying output...")
audio, sr = sf.read("test_output_jefferson.wav")
print(f"  Audio: {len(audio)} samples @ {sr} Hz")
print(f"  Stats: mean={float(audio.mean()):+.4f}, std={float(audio.std()):.4f}")
print(f"         min={float(audio.min()):+.4f}, max={float(audio.max()):.4f}")

if abs(float(audio.mean())) < 0.5 and float(audio.std()) > 0.01:
    print("\n✅ Pipeline looks healthy!")
else:
    print("\n⚠️ Check output — may still have issues.")
