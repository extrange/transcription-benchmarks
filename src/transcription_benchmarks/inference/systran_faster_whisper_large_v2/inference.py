"""
Custom inference script run by Sagemaker's Multi Model Server.

For information on overriding functions, see https://huggingface.co/docs/sagemaker/en/inference#user-defined-code-and-modules.

Source of calling script:
https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/main/src/sagemaker_huggingface_inference_toolkit/handler_service.py
"""  # noqa: INP001

import base64
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import brotli
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.transcribe import TranscriptionInfo
from faster_whisper.vad import VadOptions
from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from faster_whisper.transcribe import Segment
    from mms.context import Context


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
    batch_size: int | None = None
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


def model_fn(model_dir: str, *_args: Any) -> "WhisperModel":
    """
    Override the default method for loading a model. The return value model will be used in predict for predictions.

    `model_dir`:  the path to your unzipped model.tar.gz.
    """
    return WhisperModel(model_dir, "cuda", compute_type="float16")


def predict_fn(
    data: dict, model: WhisperModel, context: "Context"
) -> "tuple[list[Segment], TranscriptionInfo] | TranscriptionInfo":
    """
    Override the default method for prediction.

    `data`: dict with key 'inputs' containing raw bytes
    `model`: The output of `model_fn`
    `context`: mms.Context object containing custom_attributes passed in with the request. Refer to ModelArgs for passable parameters.
    """
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


def _log_memory() -> None:
    free, total = torch.cuda.mem_get_info()
    logging.info("VRAM Usage: %sGiB", f"{(total - free) / (1024*1024*1024):.1f}")


def _get_params(context: "Context") -> ModelArgs:
    custom_attrs: str | None = context.get_all_request_header(0).get(
        "X-Amzn-SageMaker-Custom-Attributes"
    )
    if not custom_attrs:
        return ModelArgs(task="transcribe", batch_size=None, single_params=SingleArgs())

    params = brotli.decompress(base64.b64decode(custom_attrs))
    return ModelArgs.model_validate_json(params)


def _detect_language(audio: bytes, model: WhisperModel) -> TranscriptionInfo:
    info = model.transcribe(audio=audio)[1]  # pyright: ignore[reportArgumentType]
    torch.cuda.empty_cache()
    return info


def _single(
    audio: bytes, params: SingleArgs, model: WhisperModel
) -> "tuple[list[Segment], TranscriptionInfo]":
    segments, info = model.transcribe(audio=audio, **params.model_dump())  # pyright: ignore[reportArgumentType]
    return list(segments), info


def _batch(
    audio: bytes, params: BatchArgs, model: WhisperModel
) -> "tuple[list[Segment], TranscriptionInfo]":
    batched_model = BatchedInferencePipeline(model)
    segments, info = batched_model.transcribe(audio, **params.model_dump())  # pyright: ignore[reportArgumentType]

    return list(segments), info
