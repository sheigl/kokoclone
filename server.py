"""FastAPI server with OpenAI-compatible endpoints for KokoClone.

Endpoints
---------
OpenAI-compatible (prefix /v1):
    POST /v1/audio/speech             — text → speech (built-in or trained RVC voices)
    GET  /v1/models                   — list available models
    GET  /v1/audio/voices             — list available voices (built-in + trained)

KokoClone-specific (prefix /v1):
    POST /v1/audio/clone              — text + voice name → cloned speech (RVC)
    POST /v1/audio/convert            — source audio + voice name → re-voiced speech
    POST /v1/audio/train              — queue a voice training job
    GET  /v1/audio/training/status    — get training queue status

Utility:
    GET  /health                      — health check
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from api.audio import content_type, encode_audio, file_extension
from api.schemas import (
    CloneSpeechRequest,
    ConvertSpeechRequest,
    DeleteVoiceResponse,
    ErrorDetail,
    ModelInfo,
    ModelList,
    OpenAISpeechRequest,
    PartialStateInfo,
    TrainVoiceRequest,
    TrainVoiceResponse,
    TrainingStatusResponse,
    VoiceInfo,
    VoiceList,
)
from core.cloner import KokoClone
from kokoclone_train.core.training_watcher import TrainingWatcher

_OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "am_adam",
    "ash": "af_nicole",
    "coral": "bf_emma",
    "echo": "af_bella",
    "fable": "af_sarah",
    "onyx": "bm_george",
    "nova": "bf_isabella",
    "sage": "am_michael",
    "shimmer": "af_sky",
}

_MODEL_IDS = ["tts-1", "tts-1-hd", "kokoro", "gpt-4o-mini-tts"]

_KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george",
]

_VOICE_PREFIX_TO_LANG: dict[str, str] = {
    "a": "en", "b": "en",
    "h": "hi",
    "f": "fr",
    "i": "it",
    "e": "es",
    "p": "pt",
    "j": "ja",
    "z": "zh",
}


def _lang_from_voice(voice: str) -> str:
    if not voice:
        return "en"
    return _VOICE_PREFIX_TO_LANG.get(voice[0].lower(), "en")


def _resolve_voice(voice: str) -> str:
    return _OPENAI_VOICE_MAP.get(voice, voice)


def _decode_base64_audio(data: str) -> bytes:
    try:
        return base64.b64decode(data)
    except Exception as exc:
        raise ValueError(f"Invalid base64 audio data: {exc}") from exc


_cloner: KokoClone | None = None
_cloner_lock = threading.Lock()

_training_watcher: TrainingWatcher | None = None


def get_cloner() -> KokoClone:
    global _cloner
    if _cloner is None:
        with _cloner_lock:
            if _cloner is None:
                print("Loading KokoClone models (first request)...")
                _cloner = KokoClone()
    return _cloner


def get_training_watcher() -> TrainingWatcher:
    global _training_watcher
    if _training_watcher is None:
        _training_watcher = TrainingWatcher(
            training_dir="training",
            voices_dir="voices",
        )
        _training_watcher.start()
    return _training_watcher


def _is_trained_voice(voice_name: str) -> bool:
    voice_path = Path("voices") / voice_name
    return (voice_path / "config.json").exists()


app = FastAPI(
    title="KokoClone TTS API",
    description="OpenAI-compatible TTS API with RVC-based trained voice cloning. "
    "Built on Kokoro-ONNX and RVC v2.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/v1/audio/speech")
async def create_speech(request: OpenAISpeechRequest, raw: Request):
    """Generate speech from text (OpenAI-compatible).

    If *voice* matches a trained RVC voice, applies Kokoro TTS + RVC conversion.
    Otherwise uses Kokoro's built-in voices directly.
    """
    cloner = get_cloner()
    voice = _resolve_voice(request.voice)

    if _is_trained_voice(voice):
        lang = request.lang_code or "en"
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            try:
                cloner.generate(
                    text=request.input,
                    lang=lang,
                    voice_name=voice,
                    output_path=output_path,
                    speed=request.speed,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=ErrorDetail(error="validation_error", message=str(exc), type="invalid_request_error").model_dump(),
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=ErrorDetail(error="processing_error", message=str(exc), type="server_error").model_dump(),
                )

            samples, sr = sf.read(output_path, dtype="float32")
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(output_path)

        audio_bytes = encode_audio(samples, sr, request.response_format)
        ct = content_type(request.response_format)
        ext = file_extension(request.response_format)

        return Response(
            content=audio_bytes,
            media_type=ct,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{ext}"',
                "Cache-Control": "no-cache",
            },
        )

    lang = request.lang_code or _lang_from_voice(voice)

    try:
        samples, sr = cloner.tts(
            text=request.input,
            lang=lang,
            voice=voice,
            speed=request.speed,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(error="validation_error", message=str(exc), type="invalid_request_error").model_dump(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(error="processing_error", message=str(exc), type="server_error").model_dump(),
        )

    audio_bytes = encode_audio(samples, sr, request.response_format)
    ct = content_type(request.response_format)
    ext = file_extension(request.response_format)

    return Response(
        content=audio_bytes,
        media_type=ct,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{ext}"',
            "Cache-Control": "no-cache",
        },
    )


@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelInfo(id=m) for m in _MODEL_IDS])


@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    if model_id not in _MODEL_IDS:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(error="model_not_found", message=f"Model '{model_id}' not found.", type="invalid_request_error").model_dump(),
        )
    return ModelInfo(id=model_id)


@app.get("/v1/audio/voices")
async def list_voices():
    all_voices = [VoiceInfo(id=v, name=v, trained=False) for v in _KOKORO_VOICES]

    voices_dir = Path("voices")
    if voices_dir.exists():
        for voice_dir in sorted(voices_dir.iterdir()):
            if (voice_dir / "config.json").exists():
                all_voices.append(VoiceInfo(id=voice_dir.name, name=voice_dir.name, trained=True))

    return VoiceList(voices=all_voices)


@app.post("/v1/audio/clone")
async def clone_speech(request: CloneSpeechRequest, raw: Request):
    """Generate speech from text with RVC voice conversion."""
    cloner = get_cloner()

    if not _is_trained_voice(request.voice_name):
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(error="voice_not_found", message=f"Voice '{request.voice_name}' is not trained.", type="invalid_request_error").model_dump(),
        )

    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    try:
        try:
            cloner.generate(
                text=request.input,
                lang=request.lang,
                voice_name=request.voice_name,
                output_path=output_path,
                speed=request.speed,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=ErrorDetail(error="validation_error", message=str(exc), type="invalid_request_error").model_dump(),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=ErrorDetail(error="processing_error", message=str(exc), type="server_error").model_dump(),
            )

        samples, sr = sf.read(output_path, dtype="float32")
        audio_bytes = encode_audio(samples, sr, request.response_format)

    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(output_path)

    ct = content_type(request.response_format)
    ext = file_extension(request.response_format)

    return Response(
        content=audio_bytes,
        media_type=ct,
        headers={
            "Content-Disposition": f'attachment; filename="clone.{ext}"',
            "Cache-Control": "no-cache",
        },
    )


@app.post("/v1/audio/convert")
async def convert_speech(request: ConvertSpeechRequest, raw: Request):
    """Re-voice source audio using a trained RVC voice."""
    cloner = get_cloner()

    if not _is_trained_voice(request.voice_name):
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(error="voice_not_found", message=f"Voice '{request.voice_name}' is not trained.", type="invalid_request_error").model_dump(),
        )

    source_bytes = _decode_base64_audio(request.source_audio)
    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(source_bytes)
        source_path = tmp.name

    try:
        try:
            cloner.convert(
                source_audio=source_path,
                voice_name=request.voice_name,
                output_path=output_path,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=ErrorDetail(error="processing_error", message=str(exc), type="server_error").model_dump(),
            )

        samples, sr = sf.read(output_path, dtype="float32")
        audio_bytes = encode_audio(samples, sr, request.response_format)

    finally:
        for p in (source_path, output_path):
            with contextlib.suppress(FileNotFoundError):
                os.remove(p)

    ct = content_type(request.response_format)
    ext = file_extension(request.response_format)

    return Response(
        content=audio_bytes,
        media_type=ct,
        headers={
            "Content-Disposition": f'attachment; filename="convert.{ext}"',
            "Cache-Control": "no-cache",
        },
    )


@app.post("/v1/audio/train")
async def train_voice(request: TrainVoiceRequest):
    """Queue a voice training job by uploading audio files."""
    name = request.name.strip()
    if not name:
        raise HTTPException(
            status_code=400,
            detail=ErrorDetail(error="validation_error", message="Voice name cannot be empty.", type="invalid_request_error").model_dump(),
        )

    training_dir = Path("training") / name
    if training_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=ErrorDetail(error="conflict", message=f"Voice '{name}' is already being trained or has been trained.", type="invalid_request_error").model_dump(),
        )

    training_dir.mkdir(parents=True, exist_ok=True)

    for i, audio_b64 in enumerate(request.audio_files):
        audio_bytes = _decode_base64_audio(audio_b64)
        audio_path = training_dir / f"audio_{i}.wav"
        audio_path.write_bytes(audio_bytes)

    get_training_watcher()

    return TrainVoiceResponse(
        name=name,
        status="queued",
        message=f"Voice '{name}' queued for training. Check /v1/audio/training/status for progress.",
    )


@app.get("/v1/audio/training/status")
async def training_status():
    """Get current training queue status."""
    watcher = get_training_watcher()
    watcher.scan_partial_states()
    status = watcher.get_queue_status()

    # Convert partial dict to PartialStateInfo objects
    partial_data: dict[str, PartialStateInfo] = {}
    for voice_name, state_info in status.get("partial", {}).items():
        if isinstance(state_info, dict):
            partial_data[voice_name] = PartialStateInfo(**state_info)

    return TrainingStatusResponse(
        queued=status.get("queued", []),
        training=status.get("training", []),
        trained=status.get("trained", []),
        partial=partial_data,
    )


@app.delete("/v1/audio/voices/{name}")
async def delete_voice(name: str):
    """Delete a trained voice."""
    voice_path = Path("voices") / name

    if not (voice_path / "config.json").exists():
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(error="not_found", message=f"Voice '{name}' not found.", type="invalid_request_error").model_dump(),
        )

    shutil.rmtree(voice_path)
    return DeleteVoiceResponse(name=name, message=f"Voice '{name}' deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="KokoClone OpenAI-compatible API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8880, help="Port (default: 8880)")
    args = parser.parse_args()

    print(f"Starting KokoClone API on http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")

    get_training_watcher()

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
