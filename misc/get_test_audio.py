from pathlib import Path
from typing import Literal

_TEST_AUDIO_DIR = Path("test_audio")

AudioFilename = Literal["1min.flac", "10min.flac", "1hour.flac"]


def get_test_audio(name: AudioFilename) -> Path:
    """Return path of a test audio file."""

    return _TEST_AUDIO_DIR / name
