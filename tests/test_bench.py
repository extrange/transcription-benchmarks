from faster_whisper_types.types import WhisperBatchOptions

from transcription_benchmarks.app_types.bench import BenchArgs
from transcription_benchmarks.benchmark.bench import bench


def test_transformers() -> None:
    res = bench(
        "openai/whisper-tiny.en", BenchArgs(test_file="short.flac", device="cpu")
    )
    assert res is not None
    assert (
        res.get_text(with_timestamps=True)
        == "[0:00:00 -> 0:00:04]  Ladies and gentlemen, thank you for being here and for your written representations.\n[0:00:05 -> 0:00:09]  You know what the purpose of this Select Committee is."
    )
    assert type(res.get_text(with_timestamps=False)) is str
    assert type(res.get_stats()) is str


def test_transformers_batch() -> None:
    res = bench(
        "openai/whisper-tiny.en",
        BenchArgs(test_file="short.flac", device="cpu", hf_batch_size=2),
    )
    assert (
        res.get_text()
        == "[0:00:00 -> 0:00:04]  Ladies and gentlemen, thank you for being here and for your written representations.\n[0:00:05 -> 0:00:09]  You know what the purpose of this Select Committee is."
    )


def test_fw_whisper() -> None:
    res = bench(
        "Systran/faster-whisper-tiny.en",
        BenchArgs(test_file="short.flac", device="cpu"),
    )
    assert (
        res.get_text()
        == "[0:00:00 -> 0:00:04]  Ladies and gentlemen, thank you for being here and for your written representations.\n[0:00:05 -> 0:00:10]  You know what the purpose of this select committee is."
    )


def test_fw_whisper_large_v2_batch() -> None:
    res = bench(
        "Systran/faster-whisper-tiny.en",
        BenchArgs(
            test_file="short.flac",
            device="cpu",
            fw_args=WhisperBatchOptions(chunk_length=30, batch_size=2),
        ),
    )
    assert (
        res.get_text()
        == "[0:00:00 -> 0:00:09]  Ladies and gentlemen, thank you for being here and for your written representations. You know what the purpose of this Select Committee is."
    )
