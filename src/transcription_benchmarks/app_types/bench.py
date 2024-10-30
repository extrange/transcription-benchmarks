from typing import Any, Literal, TypedDict

from faster_whisper_types.types import WhisperBatchOptions, WhisperOptions
from pydantic import BaseModel
from python_utils.format_hhmmss import format_hhmmss

from transcription_benchmarks.misc.get_test_audio import AudioFilename


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

    device: Literal["cpu", "cuda"] = "cuda"
    test_file: AudioFilename = "1min.flac"
    hf_batch_size: int | None = None
    hf_chunk_length_s: int | None = None
    hf_model_kwargs: _HfModelKwargs | dict[str, Any] = {}
    """
    Dict of parameters passed to [PreTrainedModel.from_pretrained](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained).
    """
    hf_generate_kwargs: _HfGenerateKwargs | dict[str, Any] = {}
    """
    Dict of parameters passed to [WhisperForConditionalGeneration.generate](https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate). You can also pass parameters for [generation_config](https://huggingface.co/docs/transformers/v4.45.2/en/generation_strategies#decoding-strategies) (e.g. `num_beams`)
    """
    fw_args: WhisperOptions | WhisperBatchOptions = WhisperOptions()
    """Arguments for faster-whisper. Ignored if model is not a faster-whisper one."""


class DetectedLanguage(BaseModel):
    """Detected language for a model."""

    name: str
    probability: float | None = None


class Segment(BaseModel):
    """Transcription segment."""

    text: str
    start: float
    end: float | None
    """Can be None, if audio was cutoff during a word."""
    log_prob: float | None = None


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
