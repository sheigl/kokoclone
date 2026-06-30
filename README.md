<p align="center">
  <img width="1050" height="450" alt="KokoClone Banner" src="https://github.com/user-attachments/assets/26fbb00c-220e-435a-8f54-431781449c76" />
</p>

<h1 align="center">🎙️ KokoClone</h1>

<p align="center">
  <a href="https://huggingface.co/spaces/PatnaikAshish/kokoclone">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue" alt="Hugging Face Space" />
  </a>
  <a href="https://huggingface.co/PatnaikAshish/kokoclone">
    <img src="https://img.shields.io/badge/🤗%20Models-Repository-orange" alt="Hugging Face Models" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%20to%203.13-3776AB.svg?logo=python&logoColor=white" alt="Python" />
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License" />
  </a>
</p>

**KokoClone** is a high-fidelity multilingual voice cloning system built on **Kokoro-ONNX** for text-to-speech and **RVC v2** (Retrieval-based Voice Conversion) for per-voice training. Train custom voices from audio recordings, then clone speech in real time — all through a Web UI, CLI, Python API, or OpenAI-compatible REST API.

It allows you to:
* **Train Voices:** Drop audio into a `training/` directory and KokoClone auto-trains an RVC v2 model.
* **Text → Clone:** Generate speech in any trained voice using Kokoro TTS + RVC conversion.
* **Audio → Clone:** Re-voice existing recordings to sound like a trained speaker — *no transcription needed*.

---

## Features

### Per-Voice Training (RVC v2)
Train custom voices from audio recordings using the RVC v2 architecture: SynthesizerTrn generator, HuBERT content encoder, and FAISS feature retrieval. Training produces high-quality models that capture timbre, pitch, and speaking style.

> **Note:** Unlike zero-shot cloning, each trained voice requires a training phase (~30–60 minutes on GPU for 1 hour of audio). The result is significantly higher quality than one-shot methods.

### Multilingual Speech Generation
Generate native speech in English (`en`), Hindi (`hi`), French (`fr`), Japanese (`ja`), Chinese (`zh`), Italian (`it`), Portuguese (`pt`), and Spanish (`es`).

### Audio-to-Audio Voice Conversion
Upload any existing speech recording and re-voice it to sound like a trained speaker. The pipeline skips TTS entirely and runs purely through the RVC conversion model, with automatic chunking for long audio files.

### Automatic Model Downloads
On first run, required models (HuBERT encoder, pre-trained RVC weights) are automatically downloaded from Hugging Face Hub.

### Real-Time Friendly
Built on Kokoro's efficient ONNX runtime with VRAM-aware chunking. Runs smoothly on CPUs, NVIDIA GPUs (CUDA), Intel Arc/discrete GPUs (XPU), and Apple Silicon (MPS).

---

## Live Demo
Try it instantly without installing anything:  
👉 **[KokoClone on Hugging Face Spaces](https://huggingface.co/spaces/PatnaikAshish/kokoclone)**

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ashish-Patnaik/kokoclone.git
cd kokoclone
```

### 2. Set Up the Environment & Install Dependencies

#### Option A: Using Conda (Recommended)
```bash
conda create -n kokoclone python=3.12 -y
conda activate kokoclone
```

**For CPU Users:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**For GPU Users (NVIDIA):**
```bash
pip install -r requirements.txt
pip install kokoro-onnx[gpu]
```

**For XPU Users (Intel Arc / Discrete GPUs):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
```

> **Note:** XPU support requires Intel GPU drivers and the Intel oneAPI Base Toolkit. See [PyTorch XPU setup guide](https://pytorch.org/docs/stable/notes/xpu.html).

#### Option B: Using `uv`
```bash
# For CPU Users
uv sync --extra cpu

# For GPU Users (NVIDIA)
uv sync --extra gpu

# For XPU Users (Intel Arc / Discrete GPUs)
uv sync --extra xpu

# Activate the environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

---

## Voice Training

KokoClone uses a **drop-in training** workflow: place audio files in `training/{voice_name}/` and the system automatically trains an RVC v2 model.

### Step-by-Step

1. **Prepare your audio:** Record or gather speech samples (minimum ~30 seconds, ideally 1–5 minutes). Supported formats: `.wav`, `.mp3`, `.flac`.

2. **Place audio in the training directory:**
   ```bash
   mkdir -p training/my_voice
   cp my_samples.wav training/my_voice/audio.wav
   # Or multiple files:
   cp *.wav training/my_voice/
   ```

3. **Train via CLI:**
   ```bash
   python cli.py --mode train --voice my_voice --audio-dir training/my_voice
   ```

4. **Or queue via API:**
   ```bash
   curl -X POST http://localhost:8880/v1/audio/train \
     -H "Content-Type: application/json" \
     -d '{
       "name": "my_voice",
       "audio_files": ["<base64-encoded-wav>"]
     }'
   ```

5. **Monitor progress:**
   ```bash
   curl http://localhost:8880/v1/audio/training/status
   # → {"queued":["my_voice"],"training":[],"trained":[]}
   ```

6. **Use the trained voice** once training completes (output appears in `voices/my_voice/`):
   ```bash
   python cli.py --mode tts --voice my_voice --text "Hello from my cloned voice!" --out output.wav
   ```

### Training Parameters

| Argument | Default | Description |
| --- | --- | --- |
| `--audio-dir` | — | Path to audio file or directory containing audio files *(required for train mode)* |
| `--epochs` | 100 | Number of training epochs (higher = better quality but longer) |
| `--pitch-adjustment` | 0 | Pitch shift in semitones (positive = higher, negative = lower) |

### Training Output

After training completes, the voice model is stored in:
```
voices/{voice_name}/
├── config.json      # Model configuration (sampling rate, F0 method, etc.)
├── v1.pth           # Trained model weights (~75–200 MB)
└── v1.index         # FAISS feature index for voice retrieval
```

The training audio is automatically deleted after successful training to save disk space.

---

## Usage

KokoClone can be used via Web UI, CLI, Python API, or REST API.

### 1. Web Interface (Gradio)

Launch the interactive web app:
```bash
python app.py
```

* **Tab 1 (Text → Clone):** Enter text, pick a language, select a trained voice from the dropdown, and generate.
* **Tab 2 (Audio → Clone):** Upload source audio and select a trained voice to re-voice it.
* **Tab 3 (Train Voice):** Upload audio files to queue a training job.
* **Tab 4 (Training Status):** Monitor active training jobs and completed voices.

### 2. Command Line Interface (CLI)

#### Train a Voice
```bash
# From a single audio file
python cli.py --mode train --voice my_voice --audio-dir recording.wav

# From a directory of audio files
python cli.py --mode train --voice my_voice --audio-dir ./my_samples/ \
  --epochs 200 --pitch-adjustment -3
```

#### Generate Speech (Text → Cloned)
```bash
python cli.py --mode tts --voice my_voice --text "Hello from KokoClone" --lang en --out output.wav
```

#### Re-voice Audio (Audio → Cloned)
```bash
python cli.py --mode convert --source original_speech.wav --voice my_voice --out revoiced.wav
```

| Argument | Default | Description |
| --- | --- | --- |
| `--mode` | `tts` | `tts` (text → speech), `convert` (audio → re-voiced audio), or `train` (voice training) |
| `--text` | — | Text to synthesize *(required for `tts` mode)* |
| `--lang` | `en` | Language code: `en hi fr ja zh it es pt` |
| `--source` | — | Path to source audio file *(required for `convert` mode)* |
| `--voice` | — | Name of trained voice in `voices/` directory *(always required)* |
| `--audio-dir` | — | Path to training audio (file or directory) *(required for `train` mode)* |
| `--epochs` | 100 | Number of training epochs *(for `train` mode)* |
| `--pitch-adjustment` | 0 | Pitch shift in semitones *(for `train` mode)* |
| `--out` | `output.wav` | Output file path |

### 3. Python API

**Text to Cloned Speech:**
```python
from core.cloner import KokoClone

cloner = KokoClone()
cloner.generate(
    text="This voice is cloned using KokoClone.",
    lang="en",
    voice_name="my_voice",  # Name of trained voice in voices/ directory
    output_path="output.wav"
)
```

**Audio-to-Audio Voice Conversion:**
```python
import soundfile as sf
from core.cloner import KokoClone

cloner = KokoClone()

# Convert source audio to a trained voice
samples, sr = cloner.convert(
    source_audio="original_speech.wav",
    voice_name="my_voice",  # Name of trained voice in voices/ directory
)

sf.write("revoiced_output.wav", samples, sr)
```

### 4. REST API Server (OpenAI-Compatible)

Launch the API server:
```bash
python server.py                       # default: 0.0.0.0:8880
python server.py --port 9000           # custom port
```

Interactive docs are available at `http://localhost:8880/docs`.

#### OpenAI SDK Compatibility

The `/v1/audio/speech` endpoint is fully compatible with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

# Generate speech with a Kokoro built-in voice
response = client.audio.speech.create(
    model="kokoro",
    voice="af_bella",
    input="Hello from KokoClone!",
)
response.stream_to_file("output.mp3")

# Generate speech with a trained RVC voice (same API!)
response = client.audio.speech.create(
    model="kokoro",
    voice="my_voice",  # Trained voice name — auto-detects and applies RVC conversion
    input="This is in my cloned voice!",
)
response.stream_to_file("output.mp3")
```

OpenAI voice aliases are supported: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`, `ash`, `coral`, `sage`.

#### Training Voices via API

**Queue a training job:**
```python
import base64, requests

# Upload audio files for training
with open("training_samples.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8880/v1/audio/train", json={
    "name": "my_voice",
    "audio_files": [audio_b64],
})
print(response.json())
# → {"name":"my_voice","status":"queued","message":"Voice 'my_voice' queued for training..."}

# Check training status
response = requests.get("http://localhost:8880/v1/audio/training/status")
print(response.json())
# → {"queued":[],"training":["my_voice"],"trained":[]}
```

**List available voices:**
```python
response = requests.get("http://localhost:8880/v1/audio/voices")
for voice in response.json()["voices"]:
    marker = "★" if voice["trained"] else " "
    print(f"{marker} {voice['name']} ({'trained' if voice['trained'] else 'built-in'})")
```

**Delete a trained voice:**
```python
requests.delete("http://localhost:8880/v1/audio/voices/my_voice")
```

#### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/audio/speech` | Text → speech (OpenAI-compatible; built-in or trained RVC voices) |
| `POST` | `/v1/audio/clone` | Text + voice name → cloned speech with RVC conversion |
| `POST` | `/v1/audio/convert` | Source audio + voice name → re-voiced speech (audio-to-audio) |
| `POST` | `/v1/audio/train` | Queue a voice training job by uploading audio files |
| `GET` | `/v1/audio/training/status` | Get current training queue status |
| `DELETE` | `/v1/audio/voices/{name}` | Delete a trained voice |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/audio/voices` | List available voices (built-in + trained) |
| `GET` | `/health` | Health check |

#### Supported Audio Formats

`mp3`, `wav`, `flac`, `pcm`, `opus`, `aac`. MP3/opus/aac require `ffmpeg` installed on the system. PCM returns raw 16-bit little-endian samples without headers.

---

## Memory Management for Long Audio

The RVC conversion pipeline in `core/chunked_convert.py` handles memory automatically when converting long audio recordings:

* **VRAM Budget:** On CUDA, chunks are sized so each forward pass uses at most 50% of total GPU memory (configurable via the `vram_fraction` parameter).
* **Chunk Size:** Each RVC chunk processes ~30 seconds of source audio with crossfade blending to suppress boundary artifacts.
* **Single-Pass Vocoding:** The full reassembled mel spectrogram is passed through the vocoder in one shot for clean waveform reconstruction.

---

## Project Structure

```text
app.py                → Gradio Web Interface (four-tab UI: TTS, Convert, Train, Status)
server.py             → OpenAI-compatible REST API server (FastAPI)
cli.py                → Command-line tool (tts, convert, and train modes)
api/
 ├── __init__.py
 ├── schemas.py       → Pydantic request/response models
 └── audio.py         → Audio format encoding (wav, flac, pcm, mp3)
core/
 ├── cloner.py        → Core TTS + RVC voice cloning engine
 ├── rvc_engine.py    → RVC v2 inference (HuBERT features, F0 extraction, FAISS retrieval)
 ├── rvc_train.py     → PyTorch training loop for RVC models
 ├── chunked_convert.py → VRAM-aware chunked audio conversion (~30s chunks with crossfade)
 ├── hubert_manager.py → HuBERT content feature extraction wrapper
 ├── rvc_assets.py    → Shared model downloads (HuBERT, pre-trained weights) from HF Hub
 ├── training_watcher.py → Background training queue + directory watcher
 └── rvc_models/      → RVC v2 PyTorch architecture
     ├── models.py    → SynthesizerTrn generator + discriminators
     ├── attentions.py → Transformer encoder layers
     └── modules.py   → Residual coupling layers, WaveNet blocks
training/             → Drop-in training directory (auto-trains voices)
voices/               → Trained voice models (config.json + v1.pth + v1.index)

```

---

## Star History

<a href="https://www.star-history.com/#Ashish-Patnaik/kokoclone&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Ashish-Patnaik/kokoclone&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Ashish-Patnaik/kokoclone&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Ashish-Patnaik/kokoclone&type=Date" />
 </picture>
</a>

---

## Acknowledgments

This project builds upon the incredible open-source work of:

* **[Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx)** — for fast and efficient neural speech synthesis.
* **[RVC v2 (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)** — for the high-fidelity voice conversion architecture.

---

## License

Licensed under the [Apache 2.0 License](https://www.google.com/search?q=LICENSE).
