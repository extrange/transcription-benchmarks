import logging
import time
from pathlib import Path

import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper_types.types import WhisperBatchOptions, WhisperOptions
from transformers import pipeline

from transcription_benchmarks.app_types.bench import (
    BenchArgs,
    BenchResult,
    Segment,
)
from transcription_benchmarks.app_types.models import (
    FasterWhisperModel,
    Model,
    PytorchModel,
    is_fw_model,
    is_pytorch_model,
)
from transcription_benchmarks.app_types.util import diff_fw_args
from transcription_benchmarks.misc.get_test_audio import get_duration, get_test_audio
from transcription_benchmarks.misc.setup_logging import setup_logging
from transcription_benchmarks.util.download import download_hf_model, get_model_dir

setup_logging()
_logger = logging.getLogger(__name__)


def bench(model: Model, args: BenchArgs | None = None) -> BenchResult:
    """Run a model on a test file and return the BenchResult."""
    if args is None:
        args = BenchArgs()

    _validate_args(model, args)

    model_dir = get_model_dir(model)
    if not model_dir.exists():
        _logger.info("Could not find model in %s, downloading", model_dir)
        download_hf_model(model)

    audio_file = get_test_audio(args.test_file)
    audio_duration = get_duration(audio_file)

    if is_fw_model(model):
        return _run_fw_model(model, model_dir, args, audio_file, audio_duration)
    if is_pytorch_model(model):
        return _run_transformers(model, model_dir, args, audio_file, audio_duration)
    msg = f"Invalid model name: {model=}"
    raise ValueError(msg)


def _validate_args(model: Model, args: BenchArgs) -> None:
    if is_fw_model(model) and (
        args.hf_generate_kwargs
        or args.hf_model_kwargs
        or args.hf_batch_size
        or args.hf_chunk_length_s
    ):
        _logger.warning(
            "You have supplied a faster-whisper model %s but also specified PyTorch model arguments which will be ignored: %s, %s, %s, %s",
            model,
            f"{args.hf_model_kwargs=}",
            f"{args.hf_generate_kwargs=}",
            f"{args.hf_batch_size=}",
            f"{args.hf_chunk_length_s=}",
        )
    # User modified the default fw_args parameter
    elif is_pytorch_model(model) and (args.fw_args != WhisperOptions()):
        _logger.warning(
            "You have supplied a pytorch model %s but also specified faster-whisper model arguments which will be ignored: %s",
            model,
            diff_fw_args(args.fw_args),
        )


def _run_fw_model(
    model: FasterWhisperModel,
    model_dir: Path,
    args: BenchArgs,
    audio_file: Path,
    audio_duration: float,
) -> BenchResult:
    fw_model = WhisperModel(
        str(model_dir),
        device=args.device,
        compute_type="float16" if args.device == "gpu" else "default",
    )

    segments = None
    info = None

    if type(args.fw_args) is WhisperBatchOptions:
        _logger.info("Using faster-whisper BatchedInferencePipeline")
        segments, info = BatchedInferencePipeline(fw_model).transcribe(
            audio=str(audio_file),
            **args.fw_args.model_dump(),
        )
    else:
        _logger.info("Using faster-whisper without batching")
        segments, info = fw_model.transcribe(
            audio=str(audio_file), **args.fw_args.model_dump()
        )

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

    memory_peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()

    return BenchResult(
        model=model,
        text=output_segments,
        params=args,
        time_taken_s=transcribe_time,
        gpu_mem_bytes=memory_peak,
        audio_duration_s=audio_duration,
        fw_transcription_info=info._asdict(),
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
        device="cuda:0" if args.device == "cuda" else -1,
        model_kwargs=args.hf_model_kwargs,  # pyright:ignore[reportArgumentType]
    )

    _logger.info("Starting benchmark...")
    now = time.time()
    res = pipe(
        str(audio_file),
        chunk_length_s=args.hf_chunk_length_s,
        batch_size=args.hf_batch_size,
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


def _cli() -> None:
    from argparse import ArgumentParser

    from pydantic_core import from_json

    parser = ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--model", default="Systran/faster-whisper-large-v2")
    args = parser.parse_args()

    with Path.open(args.config) as cfg:
        config = from_json(cfg.read())

    res = bench(args.model, BenchArgs(**config))
    _logger.info(res.get_text())
    _logger.info(res.get_stats())
    _logger.info(res.params)


if __name__ == "__main__":
    _cli()
