from pydantic import BaseModel

from misc.get_test_audio import AudioFilename


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
