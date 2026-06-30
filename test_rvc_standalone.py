"""Standalone RVC inference test script.

This script tests the RVC voice conversion pipeline in isolation from Kokoro TTS,
using the same model code but with detailed debug output at every step.

Usage:
    cd /home/sheigl/code/kokoclone
    python test_rvc_standalone.py

This will:
    1. Load the am_jefferson voice model
    2. Load one of the training audio files as input (40kHz)
    3. Extract HuBERT content features and F0
    4. Run the SynthesizerTrn generator
    5. Test the vocoder with and without conditioning
    6. Save all intermediate outputs for inspection
"""

import sys
from pathlib import Path

# Add kokoclone to path
sys.path.insert(0, "/home/sheigl/code/kokoclone")
sys.path.insert(0, "/home/sheigl/code/kokoclone-train")

import numpy as np
import torch
from scipy.io import wavfile

print("=" * 70)
print("RVC Standalone Inference Test")
print("=" * 70)

# ── Configuration ──────────────────────────────────────────────────────────
VOICE_DIR = "/home/sheigl/code/kokoclone/voices/am_jefferson"
TEST_AUDIO = "/home/sheigl/code/kokoclone-train/training_temp/am_jefferson/0_gt_wavs/0_36.wav"
OUTPUT_DIR = "/home/sheigl/code/kokoclone/test_rvc_outputs"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ── Device detection ───────────────────────────────────────────────────────
if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Using XPU device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device")
else:
    device = torch.device("cpu")
    print(f"Using CPU device")

# ── Load voice model ───────────────────────────────────────────────────────
from core.rvc_engine import load_voice, _load_vocoder, _extract_content, _extract_f0

print("\n[1] Loading voice model...")
voice_info = load_voice(VOICE_DIR, device_select=str(device))
net_g = voice_info["net_g"]
config = voice_info["config"]
print(f"  Config: {config}")
print(f"  Model params: {sum(p.numel() for p in net_g.parameters()):,}")

# ── Load test audio ────────────────────────────────────────────────────────
print(f"\n[2] Loading test audio: {TEST_AUDIO}")
sr, audio = wavfile.read(TEST_AUDIO)
print(f"  Sample rate: {sr} Hz")
print(f"  Audio shape: {audio.shape}")
print(f"  Audio dtype: {audio.dtype}")

# Convert to float32 in [-1, 1]
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.int32:
    audio = audio.astype(np.float32) / 2147483648.0

# Use only first 5 seconds to keep it fast
target_sr = config.get("sample_rate", 40000)
max_samples = target_sr * 5
if len(audio) > max_samples:
    audio = audio[:max_samples]
    print(f"  Truncated to {len(audio)} samples ({len(audio)/target_sr:.1f}s)")

print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
print(f"  Audio mean/std: {audio.mean():.4f} / {audio.std():.4f}")

# Resample if needed
if sr != target_sr:
    import librosa
    print(f"  Resampling {sr} -> {target_sr}")
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

# Save input for reference
wavfile.write(f"{OUTPUT_DIR}/01_input.wav", target_sr, audio.astype(np.float32))
print(f"  Saved input audio to 01_input.wav")

# ── Extract features ───────────────────────────────────────────────────────
print("\n[3] Extracting HuBERT content features...")
from core.rvc_assets import get_hubert_path
hubert_path = str(get_hubert_path())
content = _extract_content(hubert_path, audio, target_sr, device)
print(f"  Content shape: {content.shape}")
print(f"  Content range: [{content.min():.4f}, {content.max():.4f}]")
print(f"  Content mean/std: {content.mean():.4f} / {content.std():.4f}")

# Save content features
np.save(f"{OUTPUT_DIR}/02_content_features.npy", content)

print("\n[4] Extracting F0...")
hop_length = config.get("hop_length", 300)
f0 = _extract_f0(audio, target_sr, hop_length, "harvest")
print(f"  F0 shape: {f0.shape}")
print(f"  F0 unique values (first 20): {np.unique(f0)[:20]}")
print(f"  F0 nonzero count: {(f0 > 0).sum()} / {len(f0)}")

# Save F0
np.save(f"{OUTPUT_DIR}/03_f0.npy", f0)

# Align lengths
min_len = min(len(content), len(f0))
content = content[:min_len]
f0 = f0[:min_len]
print(f"\n[5] Aligned lengths: {min_len} frames")

# ── Test 1: Run model inference (mel + cond) ───────────────────────────────
print("\n" + "=" * 70)
print("TEST 1: SynthesizerTrn inference (mel + cond)")
print("=" * 70)

c_tensor = torch.FloatTensor(content).unsqueeze(0).to(device)
f0_tensor = torch.LongTensor(f0).unsqueeze(0).to(device)

print(f"  Input content: {c_tensor.shape}")
print(f"  Input F0: {f0_tensor.shape}")

with torch.no_grad():
    mel, cond = net_g.infer(c_tensor, f0=f0_tensor)

print(f"\n  Output mel: {mel.shape}")
print(f"    Range: [{mel.min():.4f}, {mel.max():.4f}]")
print(f"    Mean/std: {mel.mean():.4f} / {mel.std():.4f}")

print(f"\n  Output cond: {cond.shape}")
print(f"    Range: [{cond.min():.4f}, {cond.max():.4f}]")
print(f"    Mean/std: {cond.mean():.4f} / {cond.std():.4f}")

# Save mel
np.save(f"{OUTPUT_DIR}/04_mel.npy", mel[0].cpu().numpy())
np.save(f"{OUTPUT_DIR}/05_cond.npy", cond[0].cpu().numpy())

# ── Test 2: Vocoder without conditioning ───────────────────────────────────
print("\n" + "=" * 70)
print("TEST 2: Vocoder WITHOUT conditioning")
print("=" * 70)

vocoder = _load_vocoder(target_sr, device)
print(f"  Vocoder params: {sum(p.numel() for p in vocoder.parameters()):,}")

with torch.no_grad():
    wav_no_cond = vocoder.infer(mel, cond=None)

print(f"\n  Output waveform: {wav_no_cond.shape}")
print(f"    Range: [{wav_no_cond.min():.4f}, {wav_no_cond.max():.4f}]")
print(f"    Mean/std: {wav_no_cond.mean():.4f} / {wav_no_cond.std():.4f}")

wav_no_cond_np = wav_no_cond[0, 0].cpu().numpy()
wavfile.write(f"{OUTPUT_DIR}/06_output_no_cond.wav", target_sr, wav_no_cond_np.astype(np.float32))
print(f"  Saved to 06_output_no_cond.wav")

# ── Test 3: Vocoder WITH conditioning ──────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 3: Vocoder WITH conditioning")
print("=" * 70)

with torch.no_grad():
    wav_with_cond = vocoder.infer(mel, cond=cond)

print(f"\n  Output waveform: {wav_with_cond.shape}")
print(f"    Range: [{wav_with_cond.min():.4f}, {wav_with_cond.max():.4f}]")
print(f"    Mean/std: {wav_with_cond.mean():.4f} / {wav_with_cond.std():.4f}")

wav_with_cond_np = wav_with_cond[0, 0].cpu().numpy()
wavfile.write(f"{OUTPUT_DIR}/07_output_with_cond.wav", target_sr, wav_with_cond_np.astype(np.float32))
print(f"  Saved to 07_output_with_cond.wav")

# ── Test 4: Model without F0 ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 4: SynthesizerTrn WITHOUT F0 (monophonic)")
print("=" * 70)

with torch.no_grad():
    mel_no_f0, cond_no_f0 = net_g.infer(c_tensor, f0=None)

print(f"  Output mel: {mel_no_f0.shape}")
print(f"    Range: [{mel_no_f0.min():.4f}, {mel_no_f0.max():.4f}]")
print(f"    Mean/std: {mel_no_f0.mean():.4f} / {mel_no_f0.std():.4f}")

with torch.no_grad():
    wav_no_f0 = vocoder.infer(mel_no_f0, cond=cond_no_f0)

print(f"\n  Output waveform: {wav_no_f0.shape}")
print(f"    Range: [{wav_no_f0.min():.4f}, {wav_no_f0.max():.4f}]")
print(f"    Mean/std: {wav_no_f0.mean():.4f} / {wav_no_f0.std():.4f}")

wav_no_f0_np = wav_no_f0[0, 0].cpu().numpy()
wavfile.write(f"{OUTPUT_DIR}/08_output_no_f0.wav", target_sr, wav_no_f0_np.astype(np.float32))
print(f"  Saved to 08_output_no_f0.wav")

# ── Test 5: Compare with random mel ────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 5: Vocoder with random Gaussian mel (sanity check)")
print("=" * 70)

random_mel = torch.randn_like(mel) * 0.5
with torch.no_grad():
    wav_random = vocoder.infer(random_mel, cond=None)

print(f"  Random mel range: [{random_mel.min():.4f}, {random_mel.max():.4f}]")
print(f"  Output waveform: {wav_random.shape}")
print(f"    Range: [{wav_random.min():.4f}, {wav_random.max():.4f}]")
print(f"    Mean/std: {wav_random.mean():.4f} / {wav_random.std():.4f}")

wav_random_np = wav_random[0, 0].cpu().numpy()
wavfile.write(f"{OUTPUT_DIR}/09_output_random_mel.wav", target_sr, wav_random_np.astype(np.float32))
print(f"  Saved to 09_output_random_mel.wav")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"All outputs saved to: {OUTPUT_DIR}/")
print("\nFiles:")
print("  01_input.wav           - Original input audio (5s)")
print("  02_content_features.npy - HuBERT features")
print("  03_f0.npy              - F0 pitch data")
print("  04_mel.npy             - Generated mel spectrogram")
print("  05_cond.npy            - Vocoder conditioning features")
print("  06_output_no_cond.wav  - Vocoder without conditioning")
print("  07_output_with_cond.wav - Vocoder with conditioning")
print("  08_output_no_f0.wav    - Without F0 conditioning")
print("  09_output_random_mel.wav - Random mel sanity check")

print("\nKey diagnostics:")
print(f"  - Mel range: [{mel.min():.4f}, {mel.max():.4f}]")
print(f"  - Mel should be roughly in [-3, 5] range (typical for RVC)")
print(f"  - If output is saturated (mean near ±1.0), vocoder is broken")
print(f"  - If output is quiet/flat, model weights may be random/untrained")
print(f"  - Random mel test should produce noisy/static output")
