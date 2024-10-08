import logging
import time
from pathlib import Path
from typing import get_args

import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from transformers import pipeline

from app_types.bench import (
    BenchArgs,
    BenchResult,
    FasterWhisperBatchArgs,
    Segment,
)
from app_types.models import FasterWhisperModel, Model, PytorchModel
from misc.get_test_audio import get_duration, get_test_audio
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

    if model in get_args(FasterWhisperModel):
        if args.hf_generate_kwargs or args.hf_model_kwargs:
            msg = f"You have supplied a faster-whisper {model=} but also specified PyTorch model specific arguments: {args.hf_generate_kwargs=} {args.hf_model_kwargs}"
            raise ValueError(msg)
        return _run_fw_model(model, model_dir, args, audio_file, audio_duration)

    if model in get_args(PytorchModel):
        if args.fw_args:
            msg = f"You have supplied a PyTorch {model=} but also specified faster-whisper specific arguments: {args.fw_args=}"
            raise ValueError(msg)

        return _run_transformers(model, model_dir, args, audio_file, audio_duration)

    msg = f"Invalid model name: {model=}"
    raise ValueError(msg)


def _run_fw_model(
    model: FasterWhisperModel,
    model_dir: Path,
    args: BenchArgs,
    audio_file: Path,
    audio_duration: float,
) -> BenchResult:
    fw_model = WhisperModel(str(model_dir), device="cuda", compute_type="float16")

    segments = None
    info = None

    if type(args.fw_args) is FasterWhisperBatchArgs:
        segments, info = BatchedInferencePipeline(
            fw_model, chunk_length=args.chunk_length_s or 30
        ).transcribe(
            audio=str(audio_file),
            batch_size=args.batch_size or 16,
            **args.fw_args.model_dump(),
        )
        _logger.info("Using faster-whisper BatchedInferencePipeline")
    else:
        segments, info = fw_model.transcribe(
            audio=str(audio_file), **args.fw_args.model_dump() if args.fw_args else {}
        )
        _logger.info("Using faster-whisper without batching")

    _logger.info(
        "Detected language '%s' with probability %f",
        info.language,
        info.language_probability,
    )

    start = time.time()
    output_segments = []
    for s in segments:
        output_segments.append(
            Segment(text=s.text, start=s.start, end=s.end, log_prob=s.avg_logprob)
        )
        _logger.info("[%.2fs -> %.2fs] %s", s.start, s.end, s.text)
    transcribe_time = time.time() - start

    # TODO is this accurate
    memory_peak = torch.cuda.max_memory_allocated()

    return BenchResult(
        model=model,
        text=output_segments,
        params=args,
        time_taken_s=transcribe_time,
        gpu_mem_bytes=memory_peak,
        audio_duration_s=audio_duration,
    )


def _run_transformers(
    model: PytorchModel,
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
        generate_kwargs=args.hf_generate_kwargs,
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
