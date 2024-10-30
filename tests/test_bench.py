"""
Tests to ensure that model output is fixed, given parameters.
"""

import pytest
from faster_whisper_types.types import WhisperBatchOptions

from transcription_benchmarks.app_types.bench import BenchArgs
from transcription_benchmarks.benchmark.bench import bench
from transcription_benchmarks.util.gpu import gpu_available


@pytest.mark.skipif(not gpu_available(), reason="Skip if not using GPU")
class TestBench:
    def test_openai_whisper_large_v2(self) -> None:
        res = bench("openai/whisper-large-v2", BenchArgs())
        assert res is not None
        assert (
            res.get_text(with_timestamps=True)
            == "[0:00:00 -> 0:00:06]  Ladies and gentlemen, thank you for being here and for your written representations.\n[0:00:06 -> 0:00:10]  You know what the purpose of this select committee is.\n[0:00:10 -> 0:00:19]  We are exploring the risks, the ways in which deliberate online falsehoods are spread,\n[0:00:19 -> 0:00:25]  and we are primarily restricting ourselves to deliberate online falsehoods.\n[0:00:25 -> 0:00:26]  We call them DOFs.\n[0:00:26 -> 0:00:30]  And what we should do about the situation.\n[0:00:30 -> 0:00:33]  So I think it's useful to be clear about the approach\n[0:00:33 -> 0:00:36]  we are going to take in these discussions.\n[0:00:36 -> 0:00:41]  We see you, the entire panel, as people\n[0:00:41 -> 0:00:43]  who enable communications, and technology\n[0:00:43 -> 0:00:45]  enables communications, it has\n[0:00:45 -> 0:00:48]  brought immense benefits.\n[0:00:48 -> 0:00:50]  It has revolutionized societies.\n[0:00:50 -> 0:00:53]  It has given more freedom to people.\n[0:00:53 -> 0:00:56]  And at the same time, there are some issues.\n[0:00:56 -> 0:00:59]  And we see you as partners in trying to deal with those issues."
        )
        assert type(res.get_text(with_timestamps=False)) is str
        assert type(res.get_stats()) is str

    def test_openai_whisper_large_v3_turbo(self) -> None:
        res = bench("openai/whisper-large-v3-turbo", BenchArgs())
        assert (
            res.get_text()
            == "[0:00:00 -> 0:00:04]  Ladies and gentlemen, thank you for being here and for your written representations.\n[0:00:05 -> 0:00:09]  You know what the purpose of this select committee is.\n[0:00:10 -> 0:00:18]  We are exploring the risks, the ways in which deliberate online falsehoods are spread,\n[0:00:18 -> 0:00:24]  and we are primarily restricting ourselves to deliberate online falsehoods.\n[0:00:24 -> 0:00:26]  We call them DOFs.\n[0:00:26 -> 0:00:29]  And what we should do about this situation.\n[0:00:30 -> 0:00:35]  So I think it's useful to be clear about the approach we are going to take in these discussions.\n[0:00:36 -> 0:00:42]  We see you, the entire panel, as people who enable communications,\n[0:00:42 -> 0:00:55]  and technology enables communications. it has brought immense benefits, it has revolutionized societies, it has given more freedom to people, and at the same time there are some issues.\n[0:00:55 -> 0:01:00]  And we see you as partners in trying to deal with those issues."
        )

    def test_fw_whisper_large_v2(self) -> None:
        res = bench("Systran/faster-whisper-large-v2")
        assert (
            res.get_text()
            == "[0:00:00 -> 0:00:06]  Ladies and gentlemen, thank you for being here and for your written representations.\n[0:00:06 -> 0:00:10]  You know what the purpose of this select committee is.\n[0:00:10 -> 0:00:19]  We are exploring the risks, the ways in which deliberate online falsehoods are spread,\n[0:00:19 -> 0:00:24]  and we are primarily restricting ourselves to deliberate online falsehoods.\n[0:00:24 -> 0:00:30]  We call them DOFs, and what we should do about the situation.\n[0:00:30 -> 0:00:36]  So I think it's useful to be clear about the approach we are going to take in these discussions.\n[0:00:36 -> 0:00:45]  We see you, the entire panel, as people who enable communications and technology enables communications.\n[0:00:45 -> 0:00:50]  It has brought immense benefits. It has revolutionized societies.\n[0:00:50 -> 0:00:55]  It has given more freedom to people, and at the same time there are some issues.\n[0:00:55 -> 0:00:59]  And we see you as partners in trying to deal with those issues."
        )

    def test_fw_whisper_large_v2_batch(self) -> None:
        res = bench(
            "Systran/faster-whisper-large-v2",
            BenchArgs(fw_args=WhisperBatchOptions(chunk_length=30)),
        )
        assert (
            res.get_text()
            == "[0:00:00 -> 0:00:29]  Ladies and gentlemen, thank you for being here and for your written representations. You know what the purpose of this select committee is. We are exploring the risks, the ways in which deliberate online falsehoods are spread, and we are primarily restricting ourselves to deliberate online falsehoods. We call them DOFs. And what we should do about the situation.\n[0:00:30 -> 0:00:59]  So I think it's useful to be clear about the approach we are going to take in these discussions. We see you, the entire panel, as people who enable communications and technology enables communications. It has brought immense benefits. It has revolutionized societies. It has given more freedom to people. And at the same time, there are some issues. And we see you as partners in trying to deal with those issues."
        )
