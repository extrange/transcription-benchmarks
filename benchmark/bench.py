import logging
import time
from pathlib import Path

import torch
from transformers import pipeline

from benchmark.types import BenchArgs, BenchResult, Segment
from misc.get_test_audio import get_duration, get_test_audio
from misc.models import Model
from misc.setup_logging import setup_logging
from util.download import download_hf_model, get_model_dir

setup_logging()
_logger = logging.getLogger(__name__)


def bench(model: Model, args: BenchArgs | None = None) -> BenchResult:
    """Run a model on a test file and return the BenchResult."""
    if args is None:
        args = BenchArgs()

    model_dir = get_model_dir(model)
    if not model_dir.exists():
        _logger.info("Could not find model in %s, downloading", model_dir)
        download_hf_model(model)

    audio_file = get_test_audio(args.test_file)
    audio_duration = get_duration(audio_file)

    return _run_bench(model, model_dir, args, audio_file, audio_duration)


def _run_bench(
    model: Model,
    model_dir: Path,
    args: BenchArgs,
    audio_file: Path,
    audio_duration: float,
) -> BenchResult:
    _logger.info("Loading model...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(model_dir),
        torch_dtype=torch.float16,
        device="cuda:0",
        model_kwargs=args.hf_model_kwargs,
    )

    _logger.info("Starting benchmark...")
    now = time.time()
    res = pipe(
        str(audio_file),
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        return_timestamps=True,
        generate_kwargs=args.generate_kwargs,
    )
    transcribe_time = time.time() - now
    memory_peak = torch.cuda.max_memory_allocated()

    if type(res) is not dict:
        msg = f"{type(res)=} is not dict!"
        raise ValueError(msg)

    segments = [
        Segment(text=c["text"], start=c["timestamp"][0], end=c["timestamp"][1])
        for c in res["chunks"]
    ]

    return BenchResult(
        model=model,
        text=segments,
        params=args,
        time_taken_s=transcribe_time,
        gpu_mem_bytes=memory_peak,
        audio_duration_s=audio_duration,
    )
