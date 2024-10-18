"""
Tests Pydantic conversion of various faster-whisper types.
"""

import pytest
from faster_whisper.transcribe import Segment as FwSegment
from faster_whisper.transcribe import TranscriptionInfo as FwTranscriptionInfo
from faster_whisper.transcribe import TranscriptionOptions as FwTranscriptionOptions
from faster_whisper.transcribe import VadOptions as FwVadOptions
from faster_whisper.transcribe import Word as FwWord

from transcription_benchmarks.inference.systran_faster_whisper_large_v2.inference import (
    Segment,
    TranscriptionInfo,
    _fw_transcribe_output_to_pydantic,
    _segment_to_pydantic,
    _transcription_info_to_pydantic,
)


@pytest.fixture(
    scope="module",
    params=[
        [
            FwWord(start=10.48, end=10.88, word=" We", probability=0.91064453125),
            FwWord(start=10.88, end=11.06, word=" are", probability=0.80322265625),
        ],
        None,
    ],
)
def segment(request) -> FwSegment:
    return FwSegment(
        id=3,
        seek=2490,
        start=10.700000000000001,
        end=18.08,
        text=" We are exploring the risks, the ways in which deliberate online falsehoods are spread,",
        tokens=[
            50899,
            492,
        ],
        avg_logprob=-0.26447609329924865,
        compression_ratio=1.597883597883598,
        no_speech_prob=0.0135040283203125,
        words=request.param,
        temperature=0.0,
    )


@pytest.fixture(
    scope="module",
    params=[
        FwVadOptions(
            threshold=10, min_silence_duration_ms=100, max_speech_duration_s=100
        ),
        None,
    ],
)
def transcription_info(request) -> FwTranscriptionInfo:
    return FwTranscriptionInfo(
        language="en",
        language_probability=0.98291015625,
        duration=60.0,
        duration_after_vad=60.0,
        all_language_probs=[
            ("en", 0.98291015625),
            ("ms", 0.0034351348876953125),
        ],
        transcription_options=FwTranscriptionOptions(
            beam_size=5,
            best_of=5,
            patience=1,
            length_penalty=1,
            repetition_penalty=1,
            no_repeat_ngram_size=0,
            log_prob_threshold=-1.0,
            log_prob_low_threshold=None,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=True,
            prompt_reset_on_temperature=0.5,
            temperatures=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            initial_prompt=None,
            prefix=None,
            suppress_blank=True,
            suppress_tokens=(
                1,
                2,
            ),  # pyright: ignore[reportArgumentType] This is the actual output from faster-whisper
            without_timestamps=False,
            max_initial_timestamp=1.0,
            word_timestamps=True,
            prepend_punctuations="\"'“¿([{-",
            append_punctuations="\"'.。,，!！?？:：”)]}、",  # noqa: RUF001
            multilingual=False,
            output_language=None,
            max_new_tokens=None,
            clip_timestamps="0",
            hallucination_silence_threshold=None,
            hotwords=None,
        ),
        vad_options=request.param,
    )


def test_segment(segment):
    output = _segment_to_pydantic(segment)
    assert isinstance(output, Segment)


def test_transcription_info(transcription_info):
    output = _transcription_info_to_pydantic(transcription_info)
    assert isinstance(output, TranscriptionInfo)


def test_type_adaptor(transcription_info, segment):
    output = _fw_transcribe_output_to_pydantic(([segment], transcription_info))
    assert isinstance(output[0], list)
    assert isinstance(output[0][0], Segment)
    assert isinstance(output[1], TranscriptionInfo)
