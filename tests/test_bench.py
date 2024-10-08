import pytest

from app_types.bench import BenchArgs, FasterWhisperArgs, FasterWhisperBatchArgs
from benchmark.bench import bench


def test_openai_whisper_large_v2() -> None:
    res = bench("openai/whisper-large-v2", BenchArgs())
    assert res is not None
    assert (
        res.get_text(with_timestamps=True)
        == "[0.00s -> 6.00s]  Ladies and gentlemen, thank you for being here and for your written representations.\n[6.00s -> 10.70s]  You know what the purpose of this select committee is.\n[10.70s -> 19.20s]  We are exploring the risks, the ways in which deliberate online falsehoods are spread,\n[19.20s -> 25.00s]  and we are primarily restricting ourselves to deliberate online falsehoods.\n[25.00s -> 26.84s]  We call them DOFs.\n[26.84s -> 30.40s]  And what we should do about the situation.\n[30.40s -> 33.16s]  So I think it's useful to be clear about the approach\n[33.16s -> 36.96s]  we are going to take in these discussions.\n[36.96s -> 41.04s]  We see you, the entire panel, as people\n[41.04s -> 43.52s]  who enable communications, and technology\n[43.52s -> 45.72s]  enables communications, it has\n[45.72s -> 48.04s]  brought immense benefits.\n[48.04s -> 50.56s]  It has revolutionized societies.\n[50.56s -> 53.40s]  It has given more freedom to people.\n[53.40s -> 56.08s]  And at the same time, there are some issues.\n[56.08s -> 59.12s]  And we see you as partners in trying to deal with those issues."
    )
    assert type(res.get_text(with_timestamps=False)) is str
    assert type(res.get_stats()) is str


def test_openai_whisper_large_v3_turbo() -> None:
    res = bench("openai/whisper-large-v3-turbo", BenchArgs())
    assert (
        res.get_text()
        == "[0.00s -> 4.62s]  Ladies and gentlemen, thank you for being here and for your written representations.\n[5.54s -> 9.10s]  You know what the purpose of this select committee is.\n[10.22s -> 18.08s]  We are exploring the risks, the ways in which deliberate online falsehoods are spread,\n[18.76s -> 24.88s]  and we are primarily restricting ourselves to deliberate online falsehoods.\n[24.88s -> 26.70s]  We call them DOFs.\n[26.70s -> 29.10s]  And what we should do about this situation.\n[30.20s -> 35.80s]  So I think it's useful to be clear about the approach we are going to take in these discussions.\n[36.70s -> 42.70s]  We see you, the entire panel, as people who enable communications,\n[42.70s -> 55.00s]  and technology enables communications. it has brought immense benefits, it has revolutionized societies, it has given more freedom to people, and at the same time there are some issues.\n[55.00s -> 60.00s]  And we see you as partners in trying to deal with those issues."
    )


def test_fw_whisper_large_v2() -> None:
    res = bench("Systran/faster-whisper-large-v2")
    assert (
        res.get_text()
        == "[0.00s -> 6.00s]  Ladies and gentlemen, thank you for being here and for your written representations.\n[6.00s -> 10.70s]  You know what the purpose of this select committee is.\n[10.70s -> 19.20s]  We are exploring the risks, the ways in which deliberate online falsehoods are spread,\n[19.20s -> 24.90s]  and we are primarily restricting ourselves to deliberate online falsehoods.\n[24.90s -> 30.30s]  We call them DOFs, and what we should do about the situation.\n[30.30s -> 36.90s]  So I think it's useful to be clear about the approach we are going to take in these discussions.\n[36.90s -> 45.50s]  We see you, the entire panel, as people who enable communications and technology enables communications.\n[45.50s -> 50.50s]  It has brought immense benefits. It has revolutionized societies.\n[50.50s -> 55.90s]  It has given more freedom to people, and at the same time there are some issues.\n[55.90s -> 59.90s]  And we see you as partners in trying to deal with those issues."
    )


def test_fw_whisper_large_v2_batch() -> None:
    res = bench(
        "Systran/faster-whisper-large-v2", BenchArgs(fw_args=FasterWhisperBatchArgs())
    )
    assert (
        res.get_text()
        == "[0.00s -> 28.95s]  Ladies and gentlemen, thank you for being here and for your written representations. You know what the purpose of this select committee is. We are exploring the risks, the ways in which deliberate online falsehoods are spread, and we are primarily restricting ourselves to deliberate online falsehoods. We call them DOFs. And what we should do about the situation.\n[30.23s -> 59.34s]  So I think it's useful to be clear about the approach we are going to take in these discussions. We see you, the entire panel, as people who enable communications, and technology enables communications. It has brought immense benefits. It has revolutionized societies. It has given more freedom to people. And at the same time, there are some issues. And we see you as partners in trying to deal with those issues."
    )


def test_raise_on_fw_model_with_pytorch_args() -> None:
    with pytest.raises(
        ValueError,
        match="You have supplied a faster-whisper model='deepdml/faster-whisper-large-v3-turbo-ct2' but also specified PyTorch model specific arguments: args.hf_generate_kwargs={} {'attn_implementation': 'sdpa'}",
    ):
        bench(
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            BenchArgs(hf_model_kwargs={"attn_implementation": "sdpa"}),
        )


def test_raise_on_pytorch_model_with_fw_args() -> None:
    with pytest.raises(
        ValueError,
        match="You have supplied a PyTorch model='openai/whisper-large-v2' but also specified faster-whisper specific arguments:",
    ):
        bench("openai/whisper-large-v2", BenchArgs(fw_args=FasterWhisperArgs()))


def test_raise_on_pytorch_model_with_fw_batch_args() -> None:
    with pytest.raises(
        ValueError,
        match="You have supplied a PyTorch model='openai/whisper-large-v2' but also specified faster-whisper specific arguments:",
    ):
        bench("openai/whisper-large-v2", BenchArgs(fw_args=FasterWhisperBatchArgs()))
