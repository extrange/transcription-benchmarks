from typing import Any

from pydantic import BaseModel


class Segment(BaseModel):
    """Transcription segment."""

    text: str
    start: float
    end: float
    log_prob: float | None = None


class BenchResult(BaseModel):
    """Model benchmark result."""

    name: str
    text: list
    params: dict[str, Any]
