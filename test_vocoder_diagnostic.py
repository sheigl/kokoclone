"""Diagnose vocoder loading issues.

Checks:
  1. Inspect G40k.pth checkpoint keys
  2. Compare with our Generator model keys
  3. Test weight loading mismatches
  4. Try inference on CPU vs XPU
  5. Check upsample rates match
"""

import sys
sys.path.insert(0, "/home/sheigl/code/kokoclone")

import torch
from pathlib import Path

print("=" * 70)
print("VOCODER DIAGNOSTIC")
print("=" * 70)

# ── Device setup ───────────────────────────────────────────────────────────
if torch.xpu.is_available():
    xpu_device = torch.device("xpu:0")
else:
    xpu_device = None

cpu_device = torch.device("cpu")

# ── Load checkpoint ────────────────────────────────────────────────────────
from core.rvc_assets import get_vocoder_path

vocoder_path = get_vocoder_path(40000)
print(f"\n[1] Loading vocoder checkpoint: {vocoder_path}")

ckpt = torch.load(vocoder_path, map_location="cpu", weights_only=False)
print(f"  Top-level keys: {list(ckpt.keys())}")

# Get the actual state dict
full_sd = ckpt.get("weight", ckpt.get("model", ckpt))
print(f"  State dict type: {type(full_sd)}")
if isinstance(full_sd, dict):
    print(f"  State dict keys ({len(full_sd)}):")
    for k in sorted(full_sd.keys()):
        v = full_sd[k]
        if hasattr(v, 'shape'):
            print(f"    {k}: {v.shape}")
        else:
            print(f"    {k}: {type(v)}")

# ── Extract decoder keys ───────────────────────────────────────────────────
print("\n[2] Extracting dec.* keys:")
dec_sd = {k: v for k, v in full_sd.items() if k.startswith("dec.")}
print(f"  Found {len(dec_sd)} dec.* keys")

# Also check for other prefixes
other_prefixes = {}
for k in full_sd.keys():
    prefix = k.split(".")[0]
    if prefix not in other_prefixes:
        other_prefixes[prefix] = []
    other_prefixes[prefix].append(k)

print(f"  Key prefixes found:")
for prefix, keys in sorted(other_prefixes.items()):
    print(f"    {prefix}: {len(keys)} keys")

# ── Build our Generator model ──────────────────────────────────────────────
from core.rvc_models.vocoder import Generator

print("\n[3] Building our Generator model:")
gen = Generator(n_mel_channels=192)
our_keys = list(gen.state_dict().keys())
print(f"  Our keys ({len(our_keys)}):")
for k in sorted(our_keys):
    v = gen.state_dict()[k]
    print(f"    {k}: {v.shape}")

# ── Check key matching ────────────────────────────────────────────────────
print("\n[4] Key matching analysis:")

# Try loading with our keys directly (without dec. prefix)
gen_sd = gen.state_dict()
missing_keys = []
unexpected_keys = []
shape_mismatches = []

for k, v in gen_sd.items():
    if k not in dec_sd:
        missing_keys.append(k)
    elif dec_sd[k].shape != v.shape:
        shape_mismatches.append((k, dec_sd[k].shape, v.shape))

print(f"  Missing keys (in our model but not in checkpoint): {len(missing_keys)}")
for k in missing_keys:
    print(f"    {k}: {gen_sd[k].shape}")

print(f"\n  Shape mismatches: {len(shape_mismatches)}")
for k, ckpt_shape, our_shape in shape_mismatches:
    print(f"    {k}: checkpoint={ckpt_shape}, ours={our_shape}")

# ── Try loading with strict=False ──────────────────────────────────────────
print("\n[5] Loading weights (strict=False):")
missing, unexpected = gen.load_state_dict(dec_sd, strict=False)
print(f"  Missing: {len(missing)} keys")
for k in missing:
    print(f"    {k}")
print(f"  Unexpected: {len(unexpected)} keys")
for k in unexpected[:5]:
    print(f"    {k}")

# ── Test inference on CPU ─────────────────────────────────────────────────
print("\n[6] Testing vocoder inference on CPU:")
gen_cpu = Generator(n_mel_channels=192)
missing, _ = gen_cpu.load_state_dict(dec_sd, strict=False)
gen_cpu.eval()

# Create test mel: [B, C, T]
test_mel = torch.randn(1, 192, 100)  # Short test
print(f"  Input mel shape: {test_mel.shape}")

with torch.no_grad():
    output_cpu = gen_cpu(test_mel)

print(f"  Output shape: {output_cpu.shape}")
print(f"  Output range: [{output_cpu.min():.4f}, {output_cpu.max():.4f}]")
print(f"  Output mean/std: {output_cpu.mean():.4f} / {output_cpu.std():.4f}")

# ── Test inference on XPU (if available) ───────────────────────────────────
if xpu_device is not None:
    print("\n[7] Testing vocoder inference on XPU:")
    gen_xpu = Generator(n_mel_channels=192).to(xpu_device)
    missing, _ = gen_xpu.load_state_dict(dec_sd, strict=False)
    gen_xpu.eval()

    test_mel_xpu = torch.randn(1, 192, 100).to(xpu_device)

    with torch.no_grad():
        output_xpu = gen_xpu(test_mel_xpu)

    print(f"  Output shape: {output_xpu.shape}")
    print(f"  Output range: [{output_xpu.min():.4f}, {output_xpu.max():.4f}]")
    print(f"  Output mean/std: {output_xpu.mean():.4f} / {output_xpu.std():.4f}")

# ── Check upsample rates ───────────────────────────────────────────────────
print("\n[8] Upsample rate analysis:")
print(f"  Our default rates: [8, 5, 2, 2], product = {8*5*2*2}")
print(f"  Input T=100 → Output T should be ~{100 * 8 * 5 * 2 * 2} (if rates are correct)")

# Check actual kernel sizes from checkpoint
for k, v in dec_sd.items():
    if "ups." in k and ".weight" in k:
        print(f"  {k}: shape={v.shape}")

print("\n[9] Summary:")
if missing_keys or shape_mismatches:
    print("  ⚠️  KEY MISMATCH DETECTED - vocoder weights may be loaded incorrectly")
else:
    print("  ✓ Keys match, issue may be elsewhere (XPU bug, weight norm, etc.)")
