from pathlib import Path
from typing import Literal

import ffmpeg

_TEST_AUDIO_DIR = Path("test_audio")

AudioFilename = Literal[
    "1min.flac",
    "10min.flac",
    "1hour.flac",
    "chinese.flac",
    "noisy.flac",
    "long.flac",
    "mixed.m4a",
    "music.flac",
    "chinese_song.flac",
    "hokkien.flac",
    "mixed2.flac",
    "burmese.flac",
    "problematic.flac",
]
"""Refer to the readme.md in `test_audio` for descriptions."""


def get_test_audio(name: AudioFilename) -> Path:
    """Return path of a test audio file."""

    return _TEST_AUDIO_DIR / name


def get_duration(filename: str | Path) -> float:
    """Get duration in seconds of an audio file."""
    return float(ffmpeg.probe(filename=filename)["streams"][0]["duration"])
