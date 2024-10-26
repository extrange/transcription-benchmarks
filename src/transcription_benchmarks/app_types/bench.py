from collections.abc import Iterable
from typing import Any, Literal, TypedDict

from faster_whisper.vad import VadOptions
from pydantic import BaseModel
from python_utils.dict_diff import dict_diff
from python_utils.format_hhmmss import format_hhmmss

from transcription_benchmarks.misc.get_test_audio import AudioFilename


class _DictDiffBase(BaseModel):
    def dict_diff(self, other: BaseModel) -> dict[str, Any]:
        return dict_diff(self.model_dump(), other.model_dump())


class FasterWhisperArgs(_DictDiffBase):
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


class FasterWhisperBatchArgs(_DictDiffBase):
    """
    Arguments as per BatchedInferencePipeline.transcribe.

    `audio` and `batch_size` are removed.
    """

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


class Segment(BaseModel):
    """Transcription segment."""

    text: str
    start: float
    end: float | None
    """Can be None, if audio was cutoff during a word."""
    log_prob: float | None = None


class _HfModelKwargs(TypedDict):
    """Selected parameters only. All are optional."""

    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"]
    """By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual "eager" implementation."""


class _HfGenerateKwargs(TypedDict):
    """Selected parameters only. All are optional."""

    return_timestamps: bool
    task: Literal["translate", "transcribe"]
    language: str
    is_multilingual: bool
    prompt_condition_type: str
    condition_on_prev_tokens: bool
    temperature: float | list[float]
    compression_ratio_threshold: float
    logprob_threshold: float
    no_speech_threshold: float
    return_segments: bool
    return_dict_in_generate: bool

    # Passed to generation_config
    num_beams: int
    do_sample: bool
    penalty_alpha: float
    top_k: int


class BenchArgs(BaseModel):
    """Parameters for benchmarks."""

    test_file: AudioFilename = "1min.flac"
    batch_size: int | None = 24

    """Note: for faster-whisper, you should specify a batch size of 16 which is the default."""
    chunk_length_s: int | None = 30
    hf_model_kwargs: _HfModelKwargs | dict[str, Any] = {}
    """
    Dict of parameters passed to [PreTrainedModel.from_pretrained](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained).
    """
    hf_generate_kwargs: _HfGenerateKwargs | dict[str, Any] = {}
    """
    Dict of parameters passed to [WhisperForConditionalGeneration.generate](https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate). You can also pass parameters for [generation_config](https://huggingface.co/docs/transformers/v4.45.2/en/generation_strategies#decoding-strategies) (e.g. `num_beams`)
    """
    fw_args: FasterWhisperArgs | FasterWhisperBatchArgs = FasterWhisperArgs()
    """Arguments for faster-whisper. Ignored if model is not a faster-whisper one."""


class DetectedLanguage(BaseModel):
    """Detected language for a model."""

    name: str
    probability: float | None = None


class BenchResult(BaseModel):
    """Model benchmark result."""

    model: str
    text: list[Segment]
    """Should be sorted."""
    params: BenchArgs
    """Parameters for the benchmark."""
    time_taken_s: float
    """Transcription time"""
    audio_duration_s: float
    gpu_mem_bytes: float
    fw_transcription_info: dict | None = None
    """Transcription info from faster-whisper"""

    def get_text(self, *, with_timestamps: bool = True) -> str:
        """Return transcribed text, optionally with timestamps."""
        if with_timestamps:
            return "\n".join(
                f"[{format_hhmmss(s.start)} -> {format_hhmmss(s.end or 0)}] {s.text}"
                for s in self.text
            )
        return "".join(s.text for s in self.text)

    def get_stats(self) -> str:
        """Return formatted statistics."""
        speed = self.audio_duration_s / self.time_taken_s
        return f"Took {self.time_taken_s:.1f}s ({speed:.1f}x) for model={self.model}, file={self.params.test_file}, peak memory={self.gpu_mem_bytes/1_000_000:.1f}MB"
