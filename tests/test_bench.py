from benchmark.bench import bench
from benchmark.types import BenchArgs


def test_openai_whisper_large_v2() -> None:
    res = bench("openai/whisper-large-v2", BenchArgs())
    assert res is not None
    assert type(res.get_text(with_timestamps=True)) is str
    assert type(res.get_text(with_timestamps=False)) is str
    assert type(res.get_stats()) is str
