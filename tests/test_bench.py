from benchmark.bench import bench
from benchmark.types import BenchArgs


def test_openai_whisper_large_v2() -> None:
    res = bench("openai/whisper-large-v2", BenchArgs())
    assert res is not None
    assert (
        res.get_text(with_timestamps=True)
        == "[0.00s -> 6.00s]  Ladies and gentlemen, thank you for being here and for your written representations.\n[6.00s -> 10.70s]  You know what the purpose of this select committee is.\n[10.70s -> 19.20s]  We are exploring the risks, the ways in which deliberate online falsehoods are spread,\n[19.20s -> 25.00s]  and we are primarily restricting ourselves to deliberate online falsehoods.\n[25.00s -> 26.84s]  We call them DOFs.\n[26.84s -> 30.40s]  And what we should do about the situation.\n[30.40s -> 33.16s]  So I think it's useful to be clear about the approach\n[33.16s -> 36.96s]  we are going to take in these discussions.\n[36.96s -> 41.04s]  We see you, the entire panel, as people\n[41.04s -> 43.52s]  who enable communications, and technology\n[43.52s -> 45.72s]  enables communications, it has\n[45.72s -> 48.04s]  brought immense benefits.\n[48.04s -> 50.56s]  It has revolutionized societies.\n[50.56s -> 53.40s]  It has given more freedom to people.\n[53.40s -> 56.08s]  And at the same time, there are some issues.\n[56.08s -> 59.12s]  And we see you as partners in trying to deal with those issues."
    )
    assert type(res.get_text(with_timestamps=False)) is str
    assert type(res.get_stats()) is str
