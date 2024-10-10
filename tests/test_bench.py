from benchmark.bench import bench
from benchmark.types import BenchArgs


def test_openai_whisper_large_v2() -> None:
    res = bench("openai/whisper-large-v2", BenchArgs())
    assert res is not None
