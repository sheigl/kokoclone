"""Tests for Intel XPU (GPU) support in KokoClone.

These tests verify the XPU device detection and memory management patterns
across the codebase using mocked conditions, since actual XPU hardware is
not available in this test environment.
"""

import sys
from unittest.mock import MagicMock, patch

# Import real torch first so it's in sys.modules before we mock other deps.
# This ensures chunked_convert.py and other modules get the real torch module.
import torch  # noqa: E402

# Mock heavy dependencies that aren't needed for XPU logic tests.
# Note: torch is NOT mocked here because we need it for device detection tests.
for _mod in (
    "torchaudio",
    "soundfile",
    "huggingface_hub",
    "kanade_tokenizer",
    "kokoro_onnx",
    "kokoro_onnx.config",
    "misaki",
    "misaki.espeak",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Provide concrete values for kokoro_onnx.config constants used at import time
sys.modules["kokoro_onnx.config"].MAX_PHONEME_LENGTH = 512
sys.modules["kokoro_onnx.config"].SAMPLE_RATE = 24000

import pytest


# ============================================================================
# Device auto-detection tests (core/cloner.py)
# ============================================================================

class TestDeviceAutoDetection:
    """Test KokoClone._detect_device() XPU detection logic."""

    def test_detect_device_returns_xpu_when_available(self):
        """XPU is returned when available and no CUDA/MPS present."""
        from core.cloner import KokoClone

        mock_dev = MagicMock()
        mock_dev.type = "xpu"

        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("torch.xpu") as mock_xpu,
            patch.object(torch, "device", return_value=mock_dev),
        ):
            mock_xpu.is_available.return_value = True
            device = KokoClone._detect_device()
            assert device.type == "xpu"

    def test_detect_device_priority_mps_over_cuda_over_xpu(self):
        """MPS > CUDA > XPU priority chain is correct."""
        from core.cloner import KokoClone

        mock_dev = MagicMock()
        mock_dev.type = "mps"

        with (
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch("torch.xpu") as mock_xpu,
            patch.object(torch, "device", return_value=mock_dev),
        ):
            mock_xpu.is_available.return_value = True
            device = KokoClone._detect_device()
            assert device.type == "mps"

    def test_detect_device_priority_cuda_over_xpu(self):
        """CUDA > XPU when MPS is not available."""
        from core.cloner import KokoClone

        mock_dev = MagicMock()
        mock_dev.type = "cuda"

        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch("torch.xpu") as mock_xpu,
            patch.object(torch, "device", return_value=mock_dev),
        ):
            mock_xpu.is_available.return_value = True
            device = KokoClone._detect_device()
            assert device.type == "cuda"

    def test_detect_device_falls_back_to_cpu_when_no_gpu(self):
        """Returns cpu when no GPU is available."""
        from core.cloner import KokoClone

        mock_dev = MagicMock()
        mock_dev.type = "cpu"

        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("torch.xpu") as mock_xpu,
            patch.object(torch, "device", return_value=mock_dev),
        ):
            mock_xpu.is_available.return_value = False
            device = KokoClone._detect_device()
            assert device.type == "cpu"

    def test_detect_device_hasattr_guard_when_xpu_module_absent(self):
        """No crash when torch.xpu module does not exist."""
        from core.cloner import KokoClone

        original_hasattr = hasattr

        def mock_hasattr(obj, name):
            if obj is torch and name == "xpu":
                return False
            return original_hasattr(obj, name)

        mock_dev = MagicMock()
        mock_dev.type = "cpu"

        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch("builtins.hasattr", side_effect=mock_hasattr),
            patch.object(torch, "device", return_value=mock_dev),
        ):
            device = KokoClone._detect_device()
            assert device.type == "cpu"


# ============================================================================
# Chunked convert VRAM detection tests (core/chunked_convert.py)
# ============================================================================

class TestChunkedConvertXPU:
    """Test XPU cache clearing in chunked_voice_conversion."""

    def test_xpu_cache_clearing_in_chunk_loop(self):
        """torch.xpu.empty_cache() is called during chunked processing on XPU."""
        import numpy as np

        source_audio = np.random.randn(24000 * 40).astype(np.float32)
        mock_voice_info = {"model": MagicMock(), "config": {}, "faiss_index": None, "spk_id": 0}
        mock_device = MagicMock()
        mock_device.type = "xpu"

        with (
            patch("torch.xpu") as mock_xpu,
            patch("core.rvc_assets.detect_device", return_value=mock_device),
            patch("core.rvc_engine.load_voice", return_value=mock_voice_info),
            patch(
                "core.rvc_engine.convert_audio",
                return_value=(np.random.randn(24000 * 30).astype(np.float32), 24000),
            ),
            patch("scipy.io.wavfile.write"),
        ):
            mock_xpu.empty_cache = MagicMock()

            from core.chunked_convert import chunked_voice_conversion

            chunked_voice_conversion(
                source_audio=source_audio,
                sr=24000,
                voice_name="test_voice",
                output_path="/tmp/test_output.wav",
                chunk_seconds=30.0,
                overlap_seconds=1.0,
            )

            mock_xpu.empty_cache.assert_called()

    def test_cuda_cache_clearing_still_works(self):
        """CUDA cache clearing still uses torch.cuda.empty_cache (no regression)."""
        import numpy as np

        source_audio = np.random.randn(24000 * 40).astype(np.float32)
        mock_voice_info = {"model": MagicMock(), "config": {}, "faiss_index": None, "spk_id": 0}
        mock_device = MagicMock()
        mock_device.type = "cuda"

        with (
            patch("torch.cuda") as mock_cuda,
            patch("core.rvc_assets.detect_device", return_value=mock_device),
            patch("core.rvc_engine.load_voice", return_value=mock_voice_info),
            patch(
                "core.rvc_engine.convert_audio",
                return_value=(np.random.randn(24000 * 30).astype(np.float32), 24000),
            ),
            patch("scipy.io.wavfile.write"),
        ):
            mock_cuda.empty_cache = MagicMock()

            from core.chunked_convert import chunked_voice_conversion

            chunked_voice_conversion(
                source_audio=source_audio,
                sr=24000,
                voice_name="test_voice",
                output_path="/tmp/test_output.wav",
                chunk_seconds=30.0,
                overlap_seconds=1.0,
            )

            mock_cuda.empty_cache.assert_called()


# ============================================================================
# pyproject.toml XPU extra validation
# ============================================================================

class TestPyprojectXPU:
    """Validate pyproject.toml XPU configuration."""

    def test_xpu_extra_defined(self):
        """xpu extra is defined in optional-dependencies."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        assert "xpu" in config["project"]["optional-dependencies"]
        xpu_deps = config["project"]["optional-dependencies"]["xpu"]
        assert any("torch" in dep for dep in xpu_deps)

    def test_xpu_in_conflicts(self):
        """xpu is listed in tool.uv.conflicts."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        conflicts = config["tool"]["uv"]["conflicts"]
        found_xpu_conflict = False
        for conflict_group in conflicts:
            extras_in_group = [c.get("extra") for c in conflict_group if isinstance(c, dict)]
            if "xpu" in extras_in_group:
                found_xpu_conflict = True
                assert "gpu" in extras_in_group
                break

        assert found_xpu_conflict, "xpu not found in any conflict group"

    def test_pytorch_xpu_index_defined(self):
        """pytorch-xpu index is defined in tool.uv.index."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        indices = config["tool"]["uv"]["index"]
        xpu_index = [idx for idx in indices if idx.get("name") == "pytorch-xpu"]
        assert len(xpu_index) == 1, "pytorch-xpu index not found"
        assert xpu_index[0]["url"] == "https://download.pytorch.org/whl/xpu"

    def test_xpu_source_configured(self):
        """torch source includes pytorch-xpu for xpu extra."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        torch_sources = config["tool"]["uv"]["sources"]["torch"]
        xpu_source = [s for s in torch_sources if s.get("index") == "pytorch-xpu"]
        assert len(xpu_source) >= 1, "No pytorch-xpu source found for torch"
        assert xpu_source[0].get("extra") == "xpu"
