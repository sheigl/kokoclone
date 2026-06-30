"""Tests for the OpenAI-compatible API server."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf  # noqa: F401 — imported first so test_xpu_support doesn't mock it

# Mock heavy ML dependencies that aren't needed for API tests.
# NOTE: soundfile is NOT mocked — the API tests use it for real audio encoding.
for _mod in (
    "huggingface_hub",
    "kanade_tokenizer",
    "kokoro_onnx",
    "kokoro_onnx.config",
    "kokoclone_train",
    "kokoclone_train.core.training_watcher",
    "misaki",
    "misaki.espeak",
    "uvicorn",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["kokoro_onnx.config"].MAX_PHONEME_LENGTH = 512
sys.modules["kokoro_onnx.config"].SAMPLE_RATE = 24000

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked KokoClone."""
    mock_cloner = MagicMock()
    mock_cloner.tts.return_value = (np.zeros(24000, dtype=np.float32), 24000)
    mock_cloner.generate = MagicMock()
    mock_cloner.convert = MagicMock()

    with patch("server.get_cloner", return_value=mock_cloner):
        from server import app

        with TestClient(app) as c:
            yield c, mock_cloner


class TestHealth:
    def test_health_returns_healthy(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestModels:
    def test_list_models(self, client):
        c, _ = client
        resp = c.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "kokoro" in ids
        assert "tts-1" in ids
        assert "tts-1-hd" in ids

    def test_retrieve_model(self, client):
        c, _ = client
        resp = c.get("/v1/models/kokoro")
        assert resp.status_code == 200
        assert resp.json()["id"] == "kokoro"

    def test_retrieve_model_not_found(self, client):
        c, _ = client
        resp = c.get("/v1/models/nonexistent")
        assert resp.status_code == 404


class TestVoices:
    def test_list_voices(self, client):
        c, _ = client
        resp = c.get("/v1/audio/voices")
        assert resp.status_code == 200
        voices = resp.json()["voices"]
        assert len(voices) > 0
        ids = [v["id"] for v in voices]
        assert "af_bella" in ids


class TestSpeech:
    def test_speech_returns_audio(self, client):
        c, mock_cloner = client
        resp = c.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Hello world",
                "voice": "af_bella",
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert len(resp.content) > 0
        mock_cloner.tts.assert_called_once()

    def test_speech_with_openai_voice_alias(self, client):
        c, mock_cloner = client
        resp = c.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Hello world",
                "voice": "echo",  # maps to af_bella
                "response_format": "pcm",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"
        # Verify voice was resolved from echo → af_bella
        call_kwargs = mock_cloner.tts.call_args
        assert call_kwargs.kwargs["voice"] == "af_bella"

    def test_speech_pcm_format(self, client):
        c, _ = client
        resp = c.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Test",
                "voice": "af_heart",
                "response_format": "pcm",
            },
        )
        assert resp.status_code == 200
        # PCM is raw int16 — should be even number of bytes
        assert len(resp.content) % 2 == 0

    def test_speech_missing_input(self, client):
        c, _ = client
        resp = c.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "voice": "af_bella"},
        )
        assert resp.status_code == 422  # Pydantic validation error

    def test_speech_invalid_format(self, client):
        c, _ = client
        resp = c.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Test",
                "voice": "af_bella",
                "response_format": "ogg",
            },
        )
        assert resp.status_code == 422

    def test_speech_speed_validation(self, client):
        c, _ = client
        # Speed too high (> 4.0)
        resp = c.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Test",
                "voice": "af_bella",
                "speed": 10.0,
            },
        )
        assert resp.status_code == 422


class TestClone:
    def test_clone_returns_audio(self, client):
        c, mock_cloner = client

        def fake_generate(text, lang, voice_name=None, output_path="output.wav", speed=1.0):
            sf.write(output_path, np.zeros(24000, dtype=np.float32), 24000)

        mock_cloner.generate.side_effect = fake_generate

        with patch("server._is_trained_voice", return_value=True):
            resp = c.post(
                "/v1/audio/clone",
                json={
                    "input": "Hello cloned voice",
                    "lang": "en",
                    "voice_name": "my_voice",
                    "response_format": "wav",
                },
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        mock_cloner.generate.assert_called_once()

    def test_clone_requires_trained_voice(self, client):
        c, _ = client

        with patch("server._is_trained_voice", return_value=False):
            resp = c.post(
                "/v1/audio/clone",
                json={
                    "input": "Hello",
                    "lang": "en",
                    "voice_name": "unknown_voice",
                    "response_format": "wav",
                },
            )
        assert resp.status_code == 404


class TestConvert:
    def test_convert_returns_audio(self, client):
        import base64

        c, mock_cloner = client

        def fake_convert(source_audio, voice_name=None, output_path="output.wav"):
            sf.write(output_path, np.zeros(24000, dtype=np.float32), 24000)

        mock_cloner.convert.side_effect = fake_convert

        src_b64 = base64.b64encode(b"fake-source").decode()

        with patch("server._is_trained_voice", return_value=True):
            resp = c.post(
                "/v1/audio/convert",
                json={
                    "source_audio": src_b64,
                    "voice_name": "my_voice",
                    "response_format": "wav",
                },
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        mock_cloner.convert.assert_called_once()

    def test_convert_requires_trained_voice(self, client):
        import base64

        c, _ = client

        src_b64 = base64.b64encode(b"fake-source").decode()

        with patch("server._is_trained_voice", return_value=False):
            resp = c.post(
                "/v1/audio/convert",
                json={
                    "source_audio": src_b64,
                    "voice_name": "unknown_voice",
                    "response_format": "wav",
                },
            )
        assert resp.status_code == 404


class TestVoiceRegistration:
    """Test the voice deletion endpoint (registration is filesystem-based)."""

    def test_delete_voice_not_found(self, client):
        c, _ = client
        resp = c.delete("/v1/audio/voices/nonexistent")
        assert resp.status_code == 404


class TestSpeechWithClonedVoice:
    """Test that /v1/audio/speech uses the cloning pipeline for trained voices."""

    def test_speech_with_trained_voice_calls_generate(self, client):
        c, mock_cloner = client

        def fake_generate(text, lang, voice_name=None, output_path="output.wav", speed=1.0):
            sf.write(output_path, np.zeros(24000, dtype=np.float32), 24000)

        mock_cloner.generate.side_effect = fake_generate

        with patch("server._is_trained_voice", return_value=True):
            resp = c.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Hello cloned voice!",
                    "voice": "clone_me",
                    "response_format": "wav",
                },
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert len(resp.content) > 0

        mock_cloner.generate.assert_called_once()
        mock_cloner.tts.assert_not_called()

    def test_speech_with_builtin_voice_calls_tts(self, client):
        c, mock_cloner = client

        resp = c.post(
            "/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": "Hello!",
                "voice": "af_bella",
                "response_format": "wav",
            },
        )
        assert resp.status_code == 200

        mock_cloner.tts.assert_called_once()
        mock_cloner.generate.assert_not_called()

    def test_speech_with_trained_voice_passes_speed(self, client):
        c, mock_cloner = client

        def fake_generate(text, lang, voice_name=None, output_path="output.wav", speed=1.0):
            sf.write(output_path, np.zeros(24000, dtype=np.float32), 24000)

        mock_cloner.generate.side_effect = fake_generate

        with patch("server._is_trained_voice", return_value=True):
            resp = c.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Fast speech",
                    "voice": "speed_test",
                    "speed": 2.0,
                    "response_format": "wav",
                },
            )
        assert resp.status_code == 200

        call_kwargs = mock_cloner.generate.call_args
        assert call_kwargs.kwargs["speed"] == 2.0

    def test_speech_with_trained_voice_lang_override(self, client):
        c, mock_cloner = client

        def fake_generate(text, lang, voice_name=None, output_path="output.wav", speed=1.0):
            sf.write(output_path, np.zeros(24000, dtype=np.float32), 24000)

        mock_cloner.generate.side_effect = fake_generate

        with patch("server._is_trained_voice", return_value=True):
            resp = c.post(
                "/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "Bonjour",
                    "voice": "french_voice",
                    "lang_code": "es",
                    "response_format": "wav",
                },
            )
        assert resp.status_code == 200

        call_kwargs = mock_cloner.generate.call_args
        assert call_kwargs.kwargs["lang"] == "es"
