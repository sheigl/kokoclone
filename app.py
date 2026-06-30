import gradio as gr
from pathlib import Path
from core.cloner import KokoClone
from kokoclone_train.core.training_watcher import TrainingWatcher

print("Loading KokoClone models for the Web UI...")
cloner = KokoClone()
watcher = TrainingWatcher(training_dir="training", voices_dir="voices")
watcher.start()


def clone_voice(text, lang, voice_name):
    """Gradio handler: text + trained voice → cloned speech."""
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")
    if not voice_name:
        raise gr.Error("Please select a trained voice.")

    output_file = "gradio_output.wav"

    try:
        cloner.generate(
            text=text,
            lang=lang,
            voice_name=voice_name,
            output_path=output_file,
        )
        return output_file
    except Exception as e:
        raise gr.Error(f"An error occurred during generation: {str(e)}")


def convert_voice(source_audio_path, voice_name):
    """Gradio handler: source audio + trained voice → re-voiced speech."""
    if not source_audio_path:
        raise gr.Error("Please upload or record a source audio file.")
    if not voice_name:
        raise gr.Error("Please select a trained voice.")

    output_file = "gradio_convert_output.wav"

    try:
        cloner.convert(
            source_audio=source_audio_path,
            voice_name=voice_name,
            output_path=output_file,
        )
        return output_file
    except Exception as e:
        raise gr.Error(f"An error occurred during conversion: {str(e)}")


def train_voice(name, audio_files):
    """Gradio handler: queue a training job."""
    if not name or not name.strip():
        raise gr.Error("Please enter a voice name.")
    if not audio_files:
        raise gr.Error("Please upload at least one audio file for training.")

    from pathlib import Path
    import shutil

    training_dir = Path("training") / name.strip()
    if training_dir.exists():
        raise gr.Error(f"Voice '{name}' already exists or is being trained.")

    training_dir.mkdir(parents=True, exist_ok=True)

    for i, audio_path in enumerate(audio_files):
        dest = training_dir / f"audio_{i}.wav"
        shutil.copy2(str(audio_path), str(dest))

    return f"Voice '{name}' queued for training! Check the Training Status tab."


def get_trained_voices():
    """Get list of trained voice names."""
    voices = []
    voices_dir = Path("voices")
    if voices_dir.exists():
        for vd in sorted(voices_dir.iterdir()):
            if (vd / "config.json").exists():
                voices.append(vd.name)
    return voices


def get_training_status():
    """Get current training status."""
    status = watcher.get_queue_status()
    lines = []
    if status["queued"]:
        lines.append(f"Queued: {', '.join(status['queued'])}")
    if status["training"]:
        lines.append(f"Training: {', '.join(status['training'])}")
    if status["trained"]:
        lines.append(f"Trained: {', '.join(status['trained'])}")
    return "\n".join(lines) if lines else "No active training jobs."


with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>KokoClone</h1>
            <p>Voice Cloning with RVC v2.<br>
            Train voices, then generate natural speech in any trained voice.<br>
            <i>Built on Kokoro TTS + RVC.</i></p>
        </div>
        """
    )

    with gr.Tabs():
        # ── Tab 1: Text → Cloned Speech ───────────────────────────────
        with gr.Tab("Text → Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="1. Text to Synthesize",
                        lines=4,
                        placeholder="Enter the text you want spoken...",
                    )

                    lang_input = gr.Dropdown(
                        label="2. Language",
                        choices=[
                            ("English", "en"),
                            ("Hindi", "hi"),
                            ("French", "fr"),
                            ("Japanese", "ja"),
                            ("Chinese", "zh"),
                            ("Italian", "it"),
                            ("Spanish", "es"),
                            ("Portuguese", "pt"),
                        ],
                        value="en",
                    )

                    voice_dropdown = gr.Dropdown(
                        label="3. Trained Voice",
                        choices=get_trained_voices(),
                        interactive=True,
                    )

                    submit_btn = gr.Button("Generate Clone", variant="primary")

                with gr.Column(scale=1):
                    output_audio = gr.Audio(label="Generated Cloned Audio", interactive=False)

                    refresh_btn = gr.Button("Refresh Voices")

            refresh_btn.click(fn=get_trained_voices, outputs=[voice_dropdown.choices])
            submit_btn.click(
                fn=lambda: gr.update(value="Generating...", interactive=False),
                outputs=submit_btn,
            ).then(
                fn=clone_voice,
                inputs=[text_input, lang_input, voice_dropdown],
                outputs=output_audio,
            ).then(
                fn=lambda: gr.update(value="Generate Clone", interactive=True),
                outputs=submit_btn,
            )

        # ── Tab 2: Audio → Re-voiced Speech ───────────────────────────
        with gr.Tab("Audio → Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    source_audio_input = gr.Audio(
                        label="1. Source Audio (speech to re-voice)",
                        type="filepath",
                    )

                    voice_convert_dropdown = gr.Dropdown(
                        label="2. Trained Voice",
                        choices=get_trained_voices(),
                        interactive=True,
                    )

                    convert_btn = gr.Button("Convert Voice", variant="primary")

                with gr.Column(scale=1):
                    convert_output_audio = gr.Audio(label="Converted Audio", interactive=False)

            convert_btn.click(
                fn=lambda: gr.update(value="Converting...", interactive=False),
                outputs=convert_btn,
            ).then(
                fn=convert_voice,
                inputs=[source_audio_input, voice_convert_dropdown],
                outputs=convert_output_audio,
            ).then(
                fn=lambda: gr.update(value="Convert Voice", interactive=True),
                outputs=convert_btn,
            )

        # ── Tab 3: Train New Voice ────────────────────────────────────
        with gr.Tab("Train Voice"):
            with gr.Row():
                with gr.Column(scale=1):
                    train_name = gr.Textbox(
                        label="Voice Name",
                        placeholder="e.g. my_voice",
                    )

                    train_audio = gr.Files(
                        label="Training Audio Files (WAV/MP3/FLAC)",
                        file_count="multiple",
                        type="filepath",
                    )

                    train_btn = gr.Button("Start Training", variant="primary")

                with gr.Column(scale=1):
                    train_output = gr.Textbox(label="Status")

            train_btn.click(
                fn=train_voice,
                inputs=[train_name, train_audio],
                outputs=train_output,
            )

        # ── Tab 4: Training Status ────────────────────────────────────
        with gr.Tab("Training Status"):
            status_text = gr.Textbox(label="Queue Status", interactive=False)
            auto_refresh = gr.Button("Refresh")
            auto_refresh.click(fn=get_training_status, outputs=[status_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
