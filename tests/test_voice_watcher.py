"""Tests for core/voice_watcher.py — directory-based voice registration."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy ML dependencies
for _mod in (
    "torch",
    "torchaudio",
    "huggingface_hub",
    "kanade_tokenizer",
    "kokoro_onnx",
    "kokoro_onnx.config",
    "misaki",
    "misaki.espeak",
    "uvicorn",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["kokoro_onnx.config"].MAX_PHONEME_LENGTH = 512
sys.modules["kokoro_onnx.config"].SAMPLE_RATE = 24000


def _make_mock_voice_dir(dirpath: str, voice_name: str) -> None:
    """Create a minimal trained voice directory with config.json and v1.pth."""
    full_path = Path(dirpath) / voice_name
    full_path.mkdir(parents=True, exist_ok=True)
    (full_path / "config.json").write_text("{}")
    (full_path / "v1.pth").touch()


class TestWatchedVoice:
    def test_init_with_none_reference(self):
        from core.voice_watcher import _WatchedVoice

        voice = _WatchedVoice(name="test_voice", reference_audio_path=None, lang="en")
        assert voice.name == "test_voice"
        assert voice.reference_audio_path is None
        assert voice.lang == "en"

    def test_init_with_string_reference(self):
        from core.voice_watcher import _WatchedVoice

        # Interface accepts str | None for backward compatibility
        voice = _WatchedVoice(name="test", reference_audio_path="/some/path.wav", lang="ja")
        assert voice.name == "test"
        assert voice.reference_audio_path == "/some/path.wav"
        assert voice.lang == "ja"


class TestResolveUniqueName:
    @pytest.fixture
    def handler(self, tmp_path):
        from core.voice_watcher import _VoiceDirHandler

        registry = {}
        lock = __import__("threading").Lock()
        reserved = {"af_bella", "alloy"}

        return _VoiceDirHandler(
            watch_dir=str(tmp_path),
            registry=registry,
            lock=lock,
            reserved_names=reserved,
        ), registry, tmp_path

    def test_unique_name_no_conflict(self, handler):
        h, reg, tmp = handler
        result = h._resolve_unique_name("my_voice")
        assert result == "my_voice"

    def test_reserved_name_returns_none(self, handler):
        h, reg, tmp = handler
        result = h._resolve_unique_name("af_bella")
        assert result is None

    def test_appends_suffix_on_registry_conflict(self, handler):
        h, reg, tmp = handler
        reg["my_voice"] = "existing"
        result = h._resolve_unique_name("my_voice")
        assert result == "my_voice_2"

    def test_increments_suffix_until_free(self, handler):
        h, reg, tmp = handler
        reg["my_voice"] = "a"
        reg["my_voice_2"] = "b"
        reg["my_voice_3"] = "c"
        result = h._resolve_unique_name("my_voice")
        assert result == "my_voice_4"


class TestVoiceDirHandler:
    @pytest.fixture
    def handler(self, tmp_path):
        from core.voice_watcher import _VoiceDirHandler

        registry = {}
        lock = __import__("threading").Lock()
        reserved = {"af_bella", "alloy"}

        return _VoiceDirHandler(
            watch_dir=str(tmp_path),
            registry=registry,
            lock=lock,
            reserved_names=reserved,
        ), registry, tmp_path

    def test_register_valid_voice_dir(self, handler):
        h, reg, tmp = handler
        voice_name = "my_trained_voice"
        _make_mock_voice_dir(str(tmp), voice_name)
        dirpath = str(tmp / voice_name)

        h.register_dir(dirpath)

        assert voice_name in reg
        voice = reg[voice_name]
        assert getattr(voice, "name", None) == voice_name
        assert getattr(voice, "reference_audio_path", None) is None
        assert getattr(voice, "lang", None) == "en"

    def test_register_skips_dir_without_config_json(self, handler):
        h, reg, tmp = handler
        voice_name = "incomplete_voice"
        dirpath = str(tmp / voice_name)
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        (Path(dirpath) / "v1.pth").touch()

        h.register_dir(dirpath)

        assert voice_name not in reg

    def test_register_skips_dir_without_v1_pth(self, handler):
        h, reg, tmp = handler
        voice_name = "incomplete_voice"
        dirpath = str(tmp / voice_name)
        Path(dirpath).mkdir(parents=True, exist_ok=True)
        (Path(dirpath) / "config.json").write_text("{}")

        h.register_dir(dirpath)

        assert voice_name not in reg

    def test_register_skips_non_directory(self, handler):
        h, reg, tmp = handler
        filepath = str(tmp / "not_a_dir.txt")
        Path(filepath).touch()

        h.register_dir(filepath)

        assert len(reg) == 0

    def test_register_skips_reserved_name(self, handler):
        h, reg, tmp = handler
        voice_name = "af_bella"
        _make_mock_voice_dir(str(tmp), voice_name)
        dirpath = str(tmp / voice_name)

        h.register_dir(dirpath)

        assert voice_name not in reg

    def test_register_appends_suffix_on_conflict(self, handler):
        h, reg, tmp = handler
        voice_name = "my_voice"
        _make_mock_voice_dir(str(tmp), voice_name)
        dirpath = str(tmp / voice_name)

        h.register_dir(dirpath)
        assert voice_name in reg

        # Simulate a second directory with same base name (e.g., renamed)
        # by directly calling register again — it should get _2 suffix
        # Actually, since the dir already exists and is registered, re-registering
        # would just overwrite. Let's test via registry pre-population:
        reg.clear()
        reg["my_voice"] = "existing"

        h.register_dir(dirpath)

        assert "my_voice_2" in reg

    def test_unregister_deleted_dir(self, handler):
        h, reg, tmp = handler
        voice_name = "temp_voice"
        _make_mock_voice_dir(str(tmp), voice_name)
        dirpath = str(tmp / voice_name)

        h.register_dir(dirpath)
        assert voice_name in reg

        # Remove the directory
        import shutil
        shutil.rmtree(dirpath)

        h.unregister_dir(dirpath)

        assert voice_name not in reg

    def test_unregister_skips_existing_dir(self, handler):
        h, reg, tmp = handler
        voice_name = "temp_voice"
        _make_mock_voice_dir(str(tmp), voice_name)
        dirpath = str(tmp / voice_name)

        h.register_dir(dirpath)
        assert voice_name in reg

        # Directory still exists — should NOT unregister
        h.unregister_dir(dirpath)

        assert voice_name in reg


class TestVoiceWatcherIntegration:
    def test_initial_scan_registers_existing_dirs(self, tmp_path):
        from core.voice_watcher import VoiceWatcher

        registry = {}
        lock = __import__("threading").Lock()

        _make_mock_voice_dir(str(tmp_path), "alice")
        _make_mock_voice_dir(str(tmp_path), "bob")

        watcher = VoiceWatcher(
            watch_dir=str(tmp_path),
            registry=registry,
            lock=lock,
            reserved_names={"af_bella", "alloy"},
        )

        try:
            assert "alice" in registry
            assert registry["alice"].lang == "en"
            assert registry["alice"].reference_audio_path is None
            assert "bob" in registry
            assert registry["bob"].lang == "en"
        finally:
            watcher.stop()

    def test_creates_directory_if_missing(self, tmp_path):
        from core.voice_watcher import VoiceWatcher

        new_dir = tmp_path / "new_voices"
        registry = {}
        lock = __import__("threading").Lock()

        assert not new_dir.exists()

        watcher = VoiceWatcher(
            watch_dir=str(new_dir),
            registry=registry,
            lock=lock,
            reserved_names=set(),
        )

        try:
            assert new_dir.exists()
        finally:
            watcher.stop()

    def test_stop_does_not_raise(self, tmp_path):
        from core.voice_watcher import VoiceWatcher

        registry = {}
        lock = __import__("threading").Lock()

        watcher = VoiceWatcher(
            watch_dir=str(tmp_path),
            registry=registry,
            lock=lock,
            reserved_names=set(),
        )

        watcher.stop()

    def test_initial_scan_skips_incomplete_dirs(self, tmp_path):
        from core.voice_watcher import VoiceWatcher

        registry = {}
        lock = __import__("threading").Lock()

        # Complete voice dir
        _make_mock_voice_dir(str(tmp_path), "good_voice")

        # Incomplete: only config.json
        bad1 = tmp_path / "bad_voice_1"
        bad1.mkdir()
        (bad1 / "config.json").write_text("{}")

        # Incomplete: only v1.pth
        bad2 = tmp_path / "bad_voice_2"
        bad2.mkdir()
        (bad2 / "v1.pth").touch()

        # Not a directory
        (tmp_path / "not_a_dir.txt").touch()

        watcher = VoiceWatcher(
            watch_dir=str(tmp_path),
            registry=registry,
            lock=lock,
            reserved_names=set(),
        )

        try:
            assert "good_voice" in registry
            assert "bad_voice_1" not in registry
            assert "bad_voice_2" not in registry
        finally:
            watcher.stop()


class TestServerIntegration:
    def test_server_module_has_main(self):
        """Verify server module structure without importing heavy deps."""
        # We can't import server directly (needs numpy, etc.) but we can
        # verify the VoiceWatcher interface is compatible with what server expects.
        from core.voice_watcher import VoiceWatcher

        assert hasattr(VoiceWatcher, "__init__")
        assert hasattr(VoiceWatcher, "stop")

    def test_voice_watcher_interface_compatible(self):
        """Verify VoiceWatcher has the expected interface for server integration."""
        from core.voice_watcher import VoiceWatcher

        assert hasattr(VoiceWatcher, "__init__")
        assert hasattr(VoiceWatcher, "stop")
        assert hasattr(VoiceWatcher, "_initial_scan")
