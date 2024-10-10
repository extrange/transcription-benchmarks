from collections.abc import Iterable

from faster_whisper.vad import VadOptions
from pydantic import BaseModel

from misc.get_test_audio import AudioFilename


class FasterWhisperArgs(BaseModel):
    """Arguments as per the non-batched `WhisperModel.transcribe`."""

    audio: str  # Modified to just allow str
    language: str | None = None
    task: str = "transcribe"
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


class Segment(BaseModel):
    """Transcription segment."""

    text: str
    start: float
    end: float
    log_prob: float | None = None


class BenchArgs(BaseModel):
    """Parameters for benchmarks."""

    test_file: AudioFilename = "1min.flac"
    batch_size: int | None = 24
    chunk_length_s: int | None = 30
    hf_model_kwargs: dict = {}
    """
    Dict of parameters passed to the pipeline's [model_kwargs](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.model_kwargs).
    """
    generate_kwargs: dict = {}
    """
    Dict of parameters passed to [WhisperForConditionalGeneration.generate](https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate)
    """
    fw_args: FasterWhisperArgs | None = None


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

    def get_text(self, *, with_timestamps: bool = True) -> str:
        """Return transcribed text, optionally with timestamps."""
        if with_timestamps:
            return "\n".join(
                f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text}" for s in self.text
            )
        return "".join(s.text for s in self.text)

    def get_stats(self) -> str:
        """Return formatted statistics."""
        speed = self.audio_duration_s / self.time_taken_s
        return f"Took {self.time_taken_s:.1f}s ({speed:.1f}x) for model={self.model}, file={self.params.test_file}, peak memory={self.gpu_mem_bytes/1_000_000:.1f}MB"
