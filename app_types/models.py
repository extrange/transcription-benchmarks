from pathlib import Path
from typing import Literal

PytorchModel = Literal[
    "distil-whisper/distil-large-v3",
    "distil-whisper/distil-large-v2",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
]

FasterWhisperModel = Literal[
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "Systran/faster-whisper-large-v3",
    "Systran/faster-whisper-large-v2",
]

Model = PytorchModel | FasterWhisperModel

MODEL_DIR = Path("models/")
