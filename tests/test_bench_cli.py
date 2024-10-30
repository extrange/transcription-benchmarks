import sys

import pytest

from transcription_benchmarks import _bench
from transcription_benchmarks.app_types.bench import BenchArgs, BenchResult
from transcription_benchmarks.app_types.models import Model


def _fake_bench(model: Model, args: BenchArgs) -> BenchResult:
    return BenchResult(
        model=model,
        text=[],
        params=args,
        time_taken_s=1.0,
        audio_duration_s=1.0,
        gpu_mem_bytes=1.0,
    )


def test_bench_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            sys.argv[0],
            "--config",
            "tests/assets/english.json",
            "--model",
            "Systran/faster-whisper-large-v2",
        ],
    )
    monkeypatch.setattr(_bench, "bench", _fake_bench)
    _bench._cli()
