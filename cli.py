import argparse
import shutil
import sys
from pathlib import Path

from core.cloner import KokoClone
from kokoclone_train.core.rvc_train import train_voice
from kokoclone_train.core.training_watcher import TrainingWatcher


def main() -> None:
    """Entry point for the KokoClone command-line tool."""
    parser = argparse.ArgumentParser(description="KokoClone: RVC-based Voice Cloning")
    parser.add_argument(
        "--mode",
        choices=["tts", "convert", "train"],
        default="tts",
        help="Operation mode: 'tts' (text → cloned speech), 'convert' (audio → re-voiced speech), or 'train' (train a new voice)",
    )
    parser.add_argument("--text", type=str, help="[tts mode] Text to synthesize")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="[tts mode] Language code (en, hi, fr, ja, zh, it, pt, es)",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="[convert mode] Path to source audio file to re-voice (.wav)",
    )
    parser.add_argument("--voice", type=str, required=True, help="Name of trained voice in voices/ directory")
    parser.add_argument("--out", type=str, default="output.wav", help="Output file path (.wav)")
    parser.add_argument(
        "--audio-dir",
        type=str,
        help="[train mode] Path to audio file or directory containing audio files (.wav/.mp3/.flac)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="[train mode] Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--pitch-adjustment",
        type=float,
        default=0.0,
        help="[train mode] Pitch shift in semitones (positive = higher, negative = lower, default: 0)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="[train mode] Start time for audio trimming (seconds or MM:SS/HH:MM:SS)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="[train mode] End time for audio trimming (seconds or MM:SS/HH:MM:SS)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="[train mode] Resume from latest checkpoint or skip completed stages",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="[train mode] Force retraining: clear all checkpoints and state",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'cuda', 'mps', 'xpu', or 'cpu'. Defaults to auto-detect. On XPU, use ONEAPI_DEVICE_SELECTOR=level_zero:N or XPU_VISIBLE_DEVICES=N to select device.",
    )

    args = parser.parse_args()

    # Validate voice exists for tts/convert modes
    if args.mode in ("tts", "convert"):
        if not (Path("voices") / args.voice / "config.json").exists():
            print(f"Error: Voice '{args.voice}' not found in voices/ directory.", file=sys.stderr)
            sys.exit(1)

    # ── TTS mode ──────────────────────────────────────────────────────────
    if args.mode == "tts":
        if not args.text:
            parser.error("--text is required when --mode is 'tts'")

        cloner = KokoClone(device_select=args.device)
        cloner.generate(
            text=args.text,
            lang=args.lang,
            voice_name=args.voice,
            output_path=args.out,
        )

    # ── Convert mode ────────────────────────────────────────────────────────
    elif args.mode == "convert":
        if not args.source:
            parser.error("--source is required when --mode is 'convert'")

        try:
            cloner = KokoClone(device_select=args.device)
            cloner.convert(
                source_audio=args.source,
                voice_name=args.voice,
                output_path=args.out,
            )
        except Exception as exc:
            print(f"Error during voice conversion: {exc}", file=sys.stderr)
            sys.exit(1)

    # ── Train mode ──────────────────────────────────────────────────────────
    elif args.mode == "train":
        if not args.audio_dir:
            parser.error("--audio-dir is required when --mode is 'train'")

        audio_path = Path(args.audio_dir)

        # Validate audio source exists
        if not audio_path.exists():
            print(f"Error: Audio path '{args.audio_dir}' does not exist.", file=sys.stderr)
            sys.exit(1)

        # Resolve the directory to pass to train_voice.
        # If it's a single file, wrap it in a temp dir (train_voice expects a directory).
        import tempfile

        audio_extensions = (".wav", ".mp3", ".flac")

        if audio_path.is_file():
            tmp_dir = Path(tempfile.mkdtemp(prefix="kokoclone_train_"))
            shutil.copy2(audio_path, tmp_dir / audio_path.name)
            print(f"[Train] Training voice '{args.voice}' from: {audio_path}")
        else:
            # Directory: verify it contains audio files
            found = (
                list(audio_path.glob("*.wav"))
                + list(audio_path.glob("*.mp3"))
                + list(audio_path.glob("*.flac"))
            )
            if not found:
                print(f"Error: No audio files (.wav/.mp3/.flac) in '{args.audio_dir}'.", file=sys.stderr)
                sys.exit(1)

            tmp_dir = None  # use the user's directory directly — no copy needed
            print(f"[Train] Training voice '{args.voice}' from {len(found)} audio file(s) in: {audio_path}")

        try:
            watcher = TrainingWatcher(training_dir="training", voices_dir="voices")

            with watcher._lock:
                watcher._queue.append(args.voice)
                watcher._training.add(args.voice)

            # Define progress callback for inline output
            def progress_callback(msg: str, percent: float) -> None:
                print(f"[Train] {msg}")

            result = train_voice(
                voice_name=args.voice,
                audio_dir=tmp_dir if tmp_dir else audio_path,
                output_dir=Path("voices"),
                epochs=args.epochs,
                pitch_adjustment=args.pitch_adjustment,
                callback=progress_callback,
                start_time=args.start,
                end_time=args.end,
                resume=args.resume,
                force=args.force,
            )

            print(f"[Train] Training complete for '{args.voice}'!")
            print(f"[Train] Model saved to: voices/{args.voice}/")

        except Exception as exc:
            print(f"[Train] Training failed for '{args.voice}': {exc}", file=sys.stderr)
            sys.exit(1)
        finally:
            # Cleanup temp dir (if we created one for a single-file source)
            if tmp_dir and tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            with watcher._lock:
                watcher._trained.add(args.voice)
                watcher._queue.discard(args.voice)
                watcher._training.discard(args.voice)


if __name__ == "__main__":
    main()
