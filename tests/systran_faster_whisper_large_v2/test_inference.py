"""
Test the inference.py script of the faster-whisper large v2 model.
"""

import pytest
from faster_whisper import WhisperModel
from mms.context import Context

from transcription_benchmarks.inference.systran_faster_whisper_large_v2.inference import (
    Segment,
    TranscriptionInfo,
    model_fn,
    predict_fn,
)
from transcription_benchmarks.misc.get_test_audio import get_test_audio
from transcription_benchmarks.util.download import get_model_dir
from transcription_benchmarks.util.gpu import gpu_available


@pytest.fixture(scope="module")
def audio_bytes() -> bytes:
    return get_test_audio("1min.flac").read_bytes()


@pytest.fixture(scope="module")
def model() -> WhisperModel:
    return model_fn(str(get_model_dir("Systran/faster-whisper-large-v2")), None)


@pytest.fixture
def context(monkeypatch) -> Context:
    context = Context(
        model_name="model_name",
        model_dir="model_dir",
        manifest=None,
        batch_size=None,
        gpu=None,
        mms_version=None,
    )
    # Use an empty request object for now
    monkeypatch.setattr(context, "get_all_request_header", lambda _: {})
    return context


@pytest.mark.skipif(not gpu_available(), reason="Skip if not using GPU")
class TestInference:
    def test_inference(self, audio_bytes, model, context):
        segments, info = predict_fn({"inputs": audio_bytes}, model, context)
        assert isinstance(segments, list)
        assert isinstance(segments[0], Segment)
        assert isinstance(info, TranscriptionInfo)
