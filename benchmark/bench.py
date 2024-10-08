import logging
import time
from pathlib import Path

import ffmpeg
import torch
from transformers import pipeline

from misc.get_test_audio import AudioFilename, get_test_audio
from misc.models import Model
from misc.setup_logging import setup_logging

setup_logging()
_logger = logging.getLogger(__name__)


def bench(model: Model, test_file: AudioFilename, batch_size: int | None = 24) -> float:
    """Run model on a file and return the transcribe time."""
    audio_file = get_test_audio(test_file)
    audio_duration = get_duration(audio_file)
    _logger.info("Input file duration: %ss", f"{audio_duration:.1f}")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        torch_dtype=torch.float16,
        device="cuda:0",
        model_kwargs={"attn_implementation": "sdpa"},
    )

    _logger.info("Starting benchmark...")
    now = time.time()
    res = pipe(
        str(audio_file),
        chunk_length_s=30,
        batch_size=batch_size,
        return_timestamps=True,
        generate_kwargs={"language": "english", "task": "translate"},
    )
    transcribe_time = time.time() - now
    speed = audio_duration / transcribe_time
    memory_peak = torch.cuda.max_memory_allocated()
    _logger.info(res)
    _logger.info(
        "Took %s (%sx) for model %s for file %s, peak memory %s",
        f"{transcribe_time:.1f}s",
        f"{speed:.1f}",
        model,
        test_file,
        f"{memory_peak/1_000_000:.1f}MB",
    )
    return transcribe_time


def get_duration(filename: str | Path) -> float:
    """Get duration in seconds of an audio file."""
    return float(ffmpeg.probe(filename=filename)["streams"][0]["duration"])
