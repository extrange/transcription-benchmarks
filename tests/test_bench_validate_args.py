"""
Tests to ensure arguments for bench are validated correctly.
"""

import logging

from transcription_benchmarks.app_types.bench import (
    BenchArgs,
    FasterWhisperArgs,
    FasterWhisperBatchArgs,
)
from transcription_benchmarks.benchmark.bench import _validate_args


def test_warn_on_fw_model_with_pytorch_args(caplog) -> None:
    """If we give pytorch arguments for a fw model, we get a warning containing the pytorch arguments passed."""
    caplog.set_level(logging.WARNING)
    _validate_args(
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        BenchArgs(hf_model_kwargs={"attn_implementation": "sdpa"}),
    )
    assert (
        caplog.get_records("call")[0].message
        == "You have supplied a faster-whisper model deepdml/faster-whisper-large-v3-turbo-ct2 but also specified PyTorch model arguments which will be ignored: {'attn_implementation': 'sdpa'}, {}"
    )


def test_no_warn_on_pytorch_model_with_fw_default_args(caplog) -> None:
    """If we specify default arguments into the FasterWhisperArgs constructor, there should be no warning."""
    caplog.set_level(logging.WARNING)
    _validate_args(
        "openai/whisper-large-v2",
        BenchArgs(fw_args=FasterWhisperArgs(language=None, beam_size=5)),
    )
    assert len(caplog.get_records("call")) == 0


def test_no_warn_on_pytorch_model_with_fw_batch_default_args(caplog) -> None:
    """Same as above but for FW batch args. In that case, a warning should also be given, since the default is FasterWhisperArgs and not FasterWhisperBatchArgs."""
    caplog.set_level(logging.WARNING)
    _validate_args(
        "openai/whisper-large-v2",
        BenchArgs(fw_args=FasterWhisperBatchArgs()),
    )
    assert (
        caplog.get_records("call")[0].message
        == "You have supplied a pytorch model openai/whisper-large-v2 but also specified faster-whisper model arguments which will be ignored: {}"
    )


def test_warn_on_pytorch_model_with_fw_args(caplog) -> None:
    """If we give FW args to a pytorch model, we get a warning, containing the FW arguments passed."""
    caplog.set_level(logging.WARNING)
    _validate_args(
        "openai/whisper-large-v2", BenchArgs(fw_args=FasterWhisperArgs(language="zh"))
    )
    assert (
        caplog.get_records("call")[0].message
        == "You have supplied a pytorch model openai/whisper-large-v2 but also specified faster-whisper model arguments which will be ignored: {'language': 'zh'}"
    )


def test_warn_on_pytorch_model_with_fw_batch_args(caplog) -> None:
    """Same as above, but for FW batch args."""
    caplog.set_level(logging.WARNING)
    _validate_args(
        "openai/whisper-large-v2",
        BenchArgs(fw_args=FasterWhisperBatchArgs(language="zh")),
    )
    assert (
        caplog.get_records("call")[0].message
        == "You have supplied a pytorch model openai/whisper-large-v2 but also specified faster-whisper model arguments which will be ignored: {'language': 'zh'}"
    )
