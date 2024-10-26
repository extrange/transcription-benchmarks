from pathlib import Path
from typing import Literal, TypeGuard, get_args

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


def is_fw_model(model: Model) -> TypeGuard[FasterWhisperModel]:
    """Test if a model is from faster-whisper."""
    return model in get_args(FasterWhisperModel)


def is_pytorch_model(model: Model) -> TypeGuard[PytorchModel]:
    """Test if a model is from pytorch."""
    return model in get_args(PytorchModel)
