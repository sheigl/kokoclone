"""Directory-based voice registration via directory watching.

Watch the ``voices/`` directory for trained RVC v2 model directories. A voice is
"registered" when a subdirectory contains both ``config.json`` and ``v1.pth``
(model files). When training completes and a new voice directory appears, it is
automatically detected and registered. When a voice directory is deleted, it is
automatically unregistered.

The ``registry`` dict maps voice_name → ``_WatchedVoice`` object (with ``name``,
``reference_audio_path=None``, ``lang="en"``). The ``reference_audio_path`` is
always ``None`` for trained voices since the model IS the voice.

If a voice name conflicts with a built-in Kokoro voice, ``_2``, ``_3``, … are
appended until unique.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _WatchedVoice:
    """A trained voice registered from a directory in the watched voices directory."""

    __slots__ = ("name", "reference_audio_path", "lang")

    def __init__(self, name: str, reference_audio_path: str | None, lang: str):
        self.name = name
        self.reference_audio_path = reference_audio_path
        self.lang = lang


class _VoiceDirHandler(FileSystemEventHandler):
    """Handles file-system events in the watched voices directory."""

    def __init__(
        self,
        watch_dir: str,
        registry: dict[str, Any],
        lock: threading.Lock,
        reserved_names: set[str],
    ):
        super().__init__()
        self.watch_dir = os.path.abspath(watch_dir)
        self.registry = registry
        self.lock = lock
        self.reserved_names = reserved_names

    def register_dir(self, dirpath: str) -> None:
        """Register a trained voice if config.json and v1.pth exist inside."""
        voice_dir = Path(dirpath)

        if not voice_dir.is_dir():
            return

        if not (voice_dir / "config.json").exists():
            return

        if not (voice_dir / "v1.pth").exists():
            return

        voice_name = voice_dir.name
        lang = "en"

        with self.lock:
            final_name = self._resolve_unique_name(voice_name)

            if not final_name:
                logger.warning(
                    "Skipping '%s': name conflicts with a reserved voice.", voice_name
                )
                return

            reg_voice = _WatchedVoice(
                name=final_name,
                reference_audio_path=None,
                lang=lang,
            )

            self.registry[final_name] = reg_voice

        logger.info("Registered voice '%s' from directory '%s'", final_name, voice_name)

    def unregister_dir(self, dirpath: str) -> None:
        """Unregister a voice whose directory was deleted."""
        if os.path.exists(dirpath):
            return

        voice_dir = Path(dirpath)
        voice_name = voice_dir.name

        with self.lock:
            if voice_name in self.registry:
                del self.registry[voice_name]
                logger.info("Unregistered voice '%s' (directory deleted)", voice_name)

    def _resolve_unique_name(self, base_name: str) -> str | None:
        """Return a unique voice name or ``None`` if the base is reserved."""
        if base_name in self.reserved_names:
            return None

        candidate = base_name
        counter = 2
        while candidate in self.registry:
            candidate = f"{base_name}_{counter}"
            counter += 1
        return candidate

    def on_created(self, event):
        if event.is_directory:
            time.sleep(0.2)
            self.register_dir(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            self.unregister_dir(event.src_path)


class VoiceWatcher:
    """Watches the voices directory for trained RVC v2 models and auto-registers them."""

    def __init__(
        self,
        watch_dir: str,
        registry: dict[str, Any],
        lock: threading.Lock,
        reserved_names: set[str] | None = None,
    ):
        self.watch_dir = os.path.abspath(watch_dir)
        self.registry = registry
        self.lock = lock

        if reserved_names is None:
            from server import _KOKORO_VOICES

            reserved_names = set(_KOKORO_VOICES)
        self.reserved_names = reserved_names

        os.makedirs(self.watch_dir, exist_ok=True)

        self.handler = _VoiceDirHandler(
            watch_dir=self.watch_dir,
            registry=registry,
            lock=lock,
            reserved_names=reserved_names,
        )

        self.observer = Observer()
        self.observer.schedule(self.handler, self.watch_dir, recursive=False)
        self.observer.daemon = True
        self.observer.start()

        self._initial_scan()

    def _initial_scan(self) -> None:
        """Scan existing subdirectories on startup."""
        for entry in sorted(os.listdir(self.watch_dir)):
            dirpath = os.path.join(self.watch_dir, entry)
            if os.path.isdir(dirpath):
                self.handler.register_dir(dirpath)

    def stop(self) -> None:
        """Stop watching (called on server shutdown)."""
        self.observer.stop()
        self.observer.join(timeout=2)
