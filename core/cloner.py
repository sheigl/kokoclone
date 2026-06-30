"""KokoClone — Kokoro TTS + RVC voice conversion."""

import importlib.resources
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
from kokoro_onnx.config import MAX_PHONEME_LENGTH, SAMPLE_RATE
from misaki import espeak
from misaki.espeak import EspeakG2P

from core.rvc_engine import load_voice, convert_audio as rvc_convert


class KokoClone:
    def __init__(self, device_select: Optional[str] = None):
        from core.rvc_assets import detect_device
        self.device = detect_device(preferred=device_select)
        print(f"Initializing KokoClone on: {self.device.type.upper()}")

        # Kokoro cache
        self.kokoro_cache = {}

    def _get_vocab_config(self, lang):
        if lang in {"zh", "ja"}:
            zh_vocab = os.path.join("model", "config-v1.1-zh.json")
            if not os.path.exists(zh_vocab):
                print("Downloading missing file 'config-v1.1-zh.json' from hexgrad/Kokoro-82M-v1.1-zh...")
                hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M-v1.1-zh",
                    filename="config.json",
                    local_dir=".",
                )
                downloaded = os.path.join("config.json")
                if os.path.exists(downloaded):
                    os.replace(downloaded, zh_vocab)
            if os.path.exists(zh_vocab):
                return zh_vocab

        local_config = os.path.join("model", "config.json")
        if os.path.exists(local_config):
            try:
                with open(local_config, encoding="utf-8") as fp:
                    config = json.load(fp)
                if isinstance(config, dict) and "vocab" in config:
                    return local_config
                print("Warning: model/config.json is missing 'vocab'; using packaged kokoro_onnx config instead")
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: could not read model/config.json ({exc}); using packaged kokoro_onnx config instead")

        return str(importlib.resources.files("kokoro_onnx").joinpath("config.json"))

    def _patch_kokoro_compat(self, kokoro):
        input_types = {input_meta.name: input_meta.type for input_meta in kokoro.sess.get_inputs()}
        if input_types.get("speed") != "tensor(float)" or "input_ids" not in input_types:
            return kokoro

        def _create_audio_compat(instance, phonemes, voice, speed):
            if len(phonemes) > MAX_PHONEME_LENGTH:
                phonemes = phonemes[:MAX_PHONEME_LENGTH]

            start_t = time.time()
            tokens = np.array(instance.tokenizer.tokenize(phonemes), dtype=np.int64)
            assert len(tokens) <= MAX_PHONEME_LENGTH

            voice_style = voice[len(tokens)]
            inputs = {
                "input_ids": [[0, *tokens, 0]],
                "style": np.array(voice_style, dtype=np.float32),
                "speed": np.array([speed], dtype=np.float32),
            }

            audio = instance.sess.run(None, inputs)[0]
            audio_duration = len(audio) / SAMPLE_RATE
            create_duration = time.time() - start_t
            if audio_duration > 0:
                _ = create_duration / audio_duration
            return audio, SAMPLE_RATE

        import types
        kokoro._create_audio = types.MethodType(_create_audio_compat, kokoro)
        return kokoro

    def _ensure_file(self, folder, filename):
        filepath = os.path.join(folder, filename)
        repo_filepath = f"{folder}/{filename}"

        if not os.path.exists(filepath):
            print(f"Downloading missing file '{filename}' from PatnaikAshish/kokoclone...")
            hf_hub_download(
                repo_id="PatnaikAshish/kokoclone",
                filename=repo_filepath,
                local_dir=".",
            )
        return filepath

    def _create_en_callable(self):
        en_g2p = EspeakG2P(language="en-us")
        def en_callable(text):
            try:
                phonemes, _ = en_g2p(text)
                return phonemes
            except Exception:
                return text
        return en_callable

    def _get_config(self, lang):
        model_file = self._ensure_file("model", "kokoro.onnx")
        voices_file = self._ensure_file("voice", "voices-v1.0.bin")
        vocab = None
        g2p = None
        en_callable = None

        if lang == "en":
            voice = "af_bella"
        elif lang == "hi":
            g2p = EspeakG2P(language="hi")
            voice = "hf_alpha"
        elif lang == "fr":
            g2p = EspeakG2P(language="fr-fr")
            voice = "ff_siwis"
        elif lang == "it":
            g2p = EspeakG2P(language="it")
            voice = "im_nicola"
        elif lang == "es":
            g2p = EspeakG2P(language="es")
            voice = "im_nicola"
        elif lang == "pt":
            g2p = EspeakG2P(language="pt-br")
            voice = "pf_dora"
        elif lang == "ja":
            from misaki import ja
            import unidic

            if not os.path.exists(unidic.DICDIR):
                print("Downloading missing Japanese dictionary...")
                subprocess.run([sys.executable, "-m", "unidic", "download"], check=True)

            g2p = ja.JAG2P()
            voice = "jf_alpha"
            vocab = self._get_vocab_config(lang)
            en_callable = self._create_en_callable()
        elif lang == "zh":
            from misaki import zh
            import re

            base_g2p = zh.ZHG2P(version="1.1")
            en_callable = self._create_en_callable()

            def mixed_g2p(text):
                parts = re.split(r'([a-zA-Z]+)', text)
                phonemes_list = []
                for part in parts:
                    if part and part[0].isalpha() and part[0].isascii():
                        phonemes_list.append(en_callable(part))
                    else:
                        if part:
                            ph, _ = base_g2p(part)
                            phonemes_list.append(ph)
                result = "".join(phonemes_list)
                return result, text

            g2p = mixed_g2p
            voice = "zf_001"
            model_file = self._ensure_file("model", "kokoro-v1.1-zh.onnx")
            voices_file = self._ensure_file("voice", "voices-v1.1-zh.bin")
            vocab = self._get_vocab_config(lang)
        else:
            raise ValueError(f"Language '{lang}' not supported.")

        return model_file, voices_file, vocab, g2p, voice, en_callable

    def tts(self, text, lang, voice=None, speed=1.0):
        """TTS-only (no voice cloning). Returns (samples, sample_rate) numpy arrays."""
        model_file, voices_file, vocab, g2p, default_voice, en_callable = self._get_config(lang)
        voice = voice or default_voice

        if model_file not in self.kokoro_cache:
            kokoro = Kokoro(model_file, voices_file, vocab_config=vocab) if vocab else Kokoro(model_file, voices_file)
            self.kokoro_cache[model_file] = self._patch_kokoro_compat(kokoro)

        kokoro = self.kokoro_cache[model_file]

        if g2p:
            phonemes, _ = g2p(text)
            samples, sr = kokoro.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
        else:
            samples, sr = kokoro.create(text, voice=voice, speed=speed, lang="en-us")

        return samples, sr

    def generate(self, text, lang, voice_name=None, output_path="output.wav", speed=1.0):
        """Generate speech with Kokoro TTS + RVC voice conversion.

        Args:
            text: Text to synthesize.
            lang: Language code.
            voice_name: Name of trained voice in voices/ directory.
            output_path: Output WAV file path.
            speed: Speed multiplier.
        """
        model_file, voices_file, vocab, g2p, voice, en_callable = self._get_config(lang)

        # 1. Kokoro TTS Phase
        if model_file not in self.kokoro_cache:
            kokoro = Kokoro(model_file, voices_file, vocab_config=vocab) if vocab else Kokoro(model_file, voices_file)
            self.kokoro_cache[model_file] = self._patch_kokoro_compat(kokoro)

        kokoro = self.kokoro_cache[model_file]

        print(f"Synthesizing text ({lang.upper()})...")
        if g2p:
            phonemes, _ = g2p(text)
            samples, sr = kokoro.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
        else:
            samples, sr = kokoro.create(text, voice=voice, speed=speed, lang="en-us")

        # Write temp TTS output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            sf.write(temp_path, samples, sr)

        try:
            if voice_name:
                # 2. RVC Voice Conversion Phase
                print(f"Applying RVC voice conversion for '{voice_name}'...")
                voices_dir = Path("voices")
                voice_dir = voices_dir / voice_name

                if not (voice_dir / "config.json").exists():
                    raise FileNotFoundError(f"Voice '{voice_name}' not found in {voice_dir}")

                voice_info = load_voice(str(voice_dir))
                converted, target_sr = rvc_convert(samples, sr, voice_info)
                sf.write(output_path, converted, target_sr)
            else:
                # No voice conversion — just save TTS output
                import shutil
                shutil.copy2(temp_path, output_path)

            print(f"Success! Saved: {output_path}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def convert(self, source_audio, voice_name=None, output_path="output.wav"):
        """Re-voice source audio using RVC trained voice.

        Args:
            source_audio: Path to source audio file.
            voice_name: Name of trained voice in voices/ directory.
            output_path: Output WAV file path.
        """
        print(f"Applying RVC voice conversion for '{voice_name}'...")

        samples, sr = sf.read(source_audio, dtype="float32")

        voices_dir = Path("voices")
        voice_dir = voices_dir / voice_name

        if not (voice_dir / "config.json").exists():
            raise FileNotFoundError(f"Voice '{voice_name}' not found in {voice_dir}")

        voice_info = load_voice(str(voice_dir))
        converted, target_sr = rvc_convert(samples, sr, voice_info)
        sf.write(output_path, converted, target_sr)
        print(f"Success! Saved: {output_path}")
