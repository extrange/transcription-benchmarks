import sys

import pytest

from benchmark import bench


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
    bench._cli()
