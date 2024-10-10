from pathlib import Path
from typing import Literal

Model = Literal[
    "distil-whisper/distil-large-v3",
    "distil-whisper/distil-large-v2",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
]

MODEL_DIR = Path("models/")
