"""Pydantic schemas for the OpenAI-compatible API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class OpenAISpeechRequest(BaseModel):
    """Request body for POST /v1/audio/speech (OpenAI-compatible)."""

    model: str = Field(
        default="kokoro",
        description="Model name: tts-1, tts-1-hd, kokoro, or gpt-4o-mini-tts.",
    )
    input: str = Field(..., description="The text to generate audio for.")
    voice: str = Field(
        default="af_heart",
        description="Voice name (e.g. af_bella, am_adam) or trained RVC voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="Audio format. PCM returns raw 16-bit samples without headers.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed multiplier (0.25–4.0).",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="Language code override (en, hi, fr, ja, zh, it, es, pt). "
        "If not provided, inferred from the voice name prefix.",
    )


class CloneSpeechRequest(BaseModel):
    """Request body for POST /v1/audio/clone.

    Generates speech from *input* text using Kokoro TTS, then applies RVC
    voice conversion with a trained voice by name.
    """

    model: str = Field(default="kokoro", description="Model name (ignored, always kokoro).")
    input: str = Field(..., description="The text to generate audio for.")
    voice_name: str = Field(
        ...,
        description="Name of trained RVC voice in voices/ directory.",
    )
    lang: str = Field(default="en", description="Language code (en, hi, fr, ja, zh, it, es, pt).")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class ConvertSpeechRequest(BaseModel):
    """Request body for POST /v1/audio/convert.

    Re-voices *source_audio* using a trained RVC voice (audio-to-audio).
    """

    source_audio: str = Field(
        ...,
        description="Base64-encoded source audio to re-voice.",
    )
    voice_name: str = Field(
        ...,
        description="Name of trained RVC voice in voices/ directory.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="wav")


class TrainVoiceRequest(BaseModel):
    """Request body for POST /v1/audio/train.

    Queue a voice training job by uploading audio files.
    """

    name: str = Field(
        ...,
        description="Name for the trained voice (e.g. 'my_voice').",
    )
    audio_files: list[str] = Field(
        ...,
        description="List of base64-encoded audio files (WAV/MP3/FLAC) for training.",
    )


class TrainVoiceResponse(BaseModel):
    name: str
    status: str
    message: str


class PartialStateInfo(BaseModel):
    """Partial training state for a voice that was interrupted or failed."""

    stage: str = "unknown"
    can_resume: bool = False
    current_epoch: Optional[int] = None
    best_loss: Optional[float] = None
    checkpoint: Optional[str] = None
    error: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    queued: list[str] = []
    training: list[str] = []
    trained: list[str] = []
    partial: dict[str, PartialStateInfo] = {}


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 1686935002
    owned_by: str = "kokoclone"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = []


class VoiceInfo(BaseModel):
    id: str
    name: str
    trained: bool = False


class VoiceList(BaseModel):
    voices: list[VoiceInfo]


class DeleteVoiceResponse(BaseModel):
    name: str
    message: str


class ErrorDetail(BaseModel):
    error: str
    message: str
    type: str
