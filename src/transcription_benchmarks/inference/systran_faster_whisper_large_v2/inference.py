"""
Custom inference script run by Sagemaker's Multi Model Server.

For information on overriding functions, see https://huggingface.co/docs/sagemaker/en/inference#user-defined-code-and-modules.

Source of calling script:
https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/main/src/sagemaker_huggingface_inference_toolkit/handler_service.py
"""  # noqa: INP001

import base64
import logging
import subprocess
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import brotli
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.transcribe import Segment as FwSegment
from faster_whisper.transcribe import TranscriptionInfo as FwTranscriptionInfo
from faster_whisper.transcribe import TranscriptionOptions as FwTranscriptionOptions
from faster_whisper.transcribe import VadOptions as FwVadOptions
from pydantic import BaseModel, TypeAdapter, model_validator

if TYPE_CHECKING:
    from mms.context import Context


class VadOptions(BaseModel):
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


class SingleArgs(BaseModel):
    """
    Arguments as per the non-batched `WhisperModel.transcribe`.

    `audio` is removed as it is supplied by BenchArgs.
    """

    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1
    length_penalty: float = 1
    repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    temperature: float | list[float] | tuple[float, ...] = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    log_prob_low_threshold: float | None = None
    no_speech_threshold: float | None = 0.6
    condition_on_previous_text: bool = True
    prompt_reset_on_temperature: float = 0.5
    initial_prompt: str | Iterable[int] | None = None
    prefix: str | None = None
    suppress_blank: bool = True
    suppress_tokens: list[int] | None = [-1]
    without_timestamps: bool = False
    max_initial_timestamp: float = 1.0
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"  # noqa: RUF001
    multilingual: bool = False
    output_language: str | None = None
    vad_filter: bool = False
    vad_parameters: dict | VadOptions | None = None
    max_new_tokens: int | None = None
    chunk_length: int | None = None
    clip_timestamps: str | list[float] = "0"
    hallucination_silence_threshold: float | None = None
    hotwords: str | None = None
    language_detection_threshold: float | None = None
    language_detection_segments: int = 1


class BatchArgs(BaseModel):
    """
    Arguments as per BatchedInferencePipeline.transcribe.

    `audio` is removed.
    """

    batch_size: int = 16
    vad_segments: list[dict] | None = None
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    log_progress: bool = False
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1
    length_penalty: float = 1
    repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    temperature: float | list[float] | tuple[float, ...] = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    log_prob_low_threshold: float | None = None
    no_speech_threshold: float | None = 0.6
    initial_prompt: str | Iterable[int] | None = None
    prefix: str | None = None
    suppress_blank: bool = True
    suppress_tokens: list[int] | None = [-1]
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"  # noqa: RUF001
    max_new_tokens: int | None = None
    hotwords: str | None = None
    word_timestamps: bool = False
    without_timestamps: bool = True


class ModelArgs(BaseModel):
    """
    Parameters passed to CustomAttributes.

    If task=transcribe, either batch_params or single_params should be supplied.
    """

    task: Literal["transcribe", "detect_language"] = "transcribe"
    batch_params: BatchArgs | None = None
    single_params: SingleArgs | None = None
    """If truthy, will use BatchedInferencePipeline."""

    @model_validator(mode="after")
    def _check_only_either_single_or_batch_params_supplied(self) -> "ModelArgs":
        if self.task == "transcribe":
            if self.batch_params and self.single_params:
                msg = (
                    "Both batch_params and single_params were supplied: only supply one"
                )
                raise ValueError(msg)
            if not self.batch_params and not self.single_params:
                msg = "Neither batch_params or single_params were supplied"
                raise ValueError(msg)
        return self


class Word(BaseModel):  # noqa: D101
    start: float
    end: float
    word: str
    probability: float


class Segment(BaseModel):  # noqa: D101
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None
    temperature: float | None = 1.0


# Added additional parameters for multilingual videos and fixes below
class TranscriptionOptions(BaseModel):  # noqa: D101
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: float | None
    log_prob_low_threshold: float | None
    no_speech_threshold: float | None
    compression_ratio_threshold: float | None
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: list[float]
    initial_prompt: str | Iterable[int] | None
    prefix: str | None
    suppress_blank: bool
    suppress_tokens: list[int] | None
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    multilingual: bool
    output_language: str | None
    max_new_tokens: int | None
    clip_timestamps: str | list[float]
    hallucination_silence_threshold: float | None
    hotwords: str | None


class TranscriptionInfo(BaseModel):  # noqa: D101
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: list[tuple[str, float]] | None
    transcription_options: TranscriptionOptions
    vad_options: VadOptions | None


def model_fn(model_dir: str, _context: Any) -> WhisperModel:
    """
    Override the default method for loading a model. The return value model will be used in predict for predictions.

    `model_dir`:  the path to your unzipped model.tar.gz.
    """
    logging.info(
        "nvidia-smi:\n%s",
        subprocess.run(  # noqa: S603
            "/usr/bin/nvidia-smi", capture_output=True, check=True
        ).stdout.decode("utf-8"),
    )
    logging.info("Pytorch version: %s", torch.__version__)
    return WhisperModel(model_dir, "cuda", compute_type="float16")


def predict_fn(
    data: dict, model: WhisperModel, context: "Context"
) -> tuple[list[Segment], TranscriptionInfo]:
    """
    Override the default method for prediction.

    `data`: dict with key 'inputs' containing raw bytes
    `model`: The output of `model_fn`
    `context`: mms.Context object containing custom_attributes passed in with the request. Refer to ModelArgs for passable parameters.

    If the task is "detect_language", list[Segment] will be an empty list.
    """
    torch.cuda.empty_cache()
    logging.info("VRAM memory cache cleared.")
    _log_memory()
    audio: bytes = data["inputs"]

    args = _get_params(context)

    if args.task == "detect_language":
        return _detect_language(audio, model)

    if args.single_params:
        return _single(audio, args.single_params, model)
    if args.batch_params:
        return _batch(audio, args.batch_params, model)

    msg = f"Invalid arguments: {args=}"
    raise ValueError(msg)


def output_fn(
    output: tuple[list[Segment], TranscriptionInfo], _accept: str, _context: Any = None
) -> str:
    """Override the output serialization method."""
    ta = TypeAdapter(tuple[list[Segment], TranscriptionInfo])
    return ta.dump_json(output).decode("utf-8")


def _log_memory() -> None:
    free, total = torch.cuda.mem_get_info()
    logging.info("VRAM Usage: %sGiB", f"{(total - free) / (1024*1024*1024):.1f}")


def _get_params(context: "Context") -> ModelArgs:
    custom_attrs: str | None = context.get_all_request_header(0).get(
        "X-Amzn-SageMaker-Custom-Attributes"
    )
    if not custom_attrs:
        return ModelArgs(task="transcribe", single_params=SingleArgs())

    params = brotli.decompress(base64.b64decode(custom_attrs))
    return ModelArgs.model_validate_json(params)


def _detect_language(
    audio: bytes, model: WhisperModel
) -> tuple[list[Segment], TranscriptionInfo]:
    info = model.transcribe(audio=audio)[1]  # pyright: ignore[reportArgumentType]
    torch.cuda.empty_cache()
    return [], _transcription_info_to_pydantic(info)


def _single(
    audio: bytes, params: SingleArgs, model: WhisperModel
) -> tuple[list[Segment], TranscriptionInfo]:
    output = model.transcribe(audio=audio, **params.model_dump())  # pyright: ignore[reportArgumentType]
    return _fw_transcribe_output_to_pydantic(output)


def _batch(
    audio: bytes, params: BatchArgs, model: WhisperModel
) -> tuple[list[Segment], TranscriptionInfo]:
    batched_model = BatchedInferencePipeline(model)
    output = batched_model.transcribe(
        audio,  # pyright: ignore[reportArgumentType]
        **params.model_dump(),
    )

    return _fw_transcribe_output_to_pydantic(output)


def _fw_transcribe_output_to_pydantic(
    o: tuple[Iterable[FwSegment], FwTranscriptionInfo],
) -> tuple[list[Segment], TranscriptionInfo]:
    ta = TypeAdapter(tuple[list[Segment], TranscriptionInfo])
    return ta.validate_python(
        ([_segment_to_pydantic(s) for s in o[0]], _transcription_info_to_pydantic(o[1]))
    )


def _transcription_info_to_pydantic(info: FwTranscriptionInfo) -> TranscriptionInfo:
    info_dict = info._asdict()
    if isinstance(info_dict["transcription_options"], FwTranscriptionOptions):
        info_dict["transcription_options"] = info_dict[
            "transcription_options"
        ]._asdict()
    if isinstance(info_dict["vad_options"], FwVadOptions):
        info_dict["vad_options"] = info_dict["vad_options"]._asdict()
    return TranscriptionInfo(**info_dict)


def _segment_to_pydantic(s: FwSegment) -> Segment:
    s_dict = s._asdict()
    if isinstance(s_dict["words"], list):
        s_dict["words"] = [Word(**w._asdict()) for w in s_dict["words"]]
    return Segment(**s_dict)
