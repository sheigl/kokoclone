from core.cloner import KokoClone

# Initialize the cloner (Auto-downloads models if missing, and auto-detects CPU/GPU)
cloner = KokoClone()

# Generate your cloned audio!
cloner.generate(
    text="Welcome to KokoClone! This is incredibly easy to use.",
    lang="en",
    reference_audio="ss.wav", # Replace with your actual reference audio file
    output_path="english_output.wav"
)