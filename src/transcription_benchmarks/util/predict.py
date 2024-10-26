import base64
import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import boto3
import brotli
from pydantic import TypeAdapter
from sagemaker import Predictor
from sagemaker.async_inference import WaiterConfig
from sagemaker.huggingface.model import HuggingFacePredictor
from sagemaker.local import LocalSession
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.serializers import DataSerializer

from transcription_benchmarks.inference.systran_faster_whisper_large_v2.inference import (
    ModelArgs,
    Segment,
    TranscriptionInfo,
)
from transcription_benchmarks.misc.get_test_audio import AudioFilename, get_test_audio

_logger = logging.getLogger(__name__)


def predict_local(
    test_file: AudioFilename = "10min.flac",
    params: ModelArgs | None = None,
) -> tuple[list[Segment], TranscriptionInfo]:
    """Test a local Sagemaker endpoint."""
    sagemaker_session = LocalSession(boto3.Session(region_name="ap-southeast-1"))
    sagemaker_session.config = {"local": {"local_code": True}}

    predictor = Predictor(
        endpoint_name="local",
        sagemaker_session=sagemaker_session,
        serializer=DataSerializer(content_type="audio/x-audio"),
    )

    with (
        _log_stats(test_file, "local"),
        get_test_audio(test_file).open("rb") as f,
    ):
        if params:
            response = predictor.predict(
                data=f.read(),
                initial_args=_compress_args(params),
            )
            _logger.info(json.loads(response))
        else:
            response = predictor.predict(data=f.read())
            _logger.info(json.loads(response))
        # For local endpoints, the result is in bytes
        return _parse_response(json.loads(response))


def predict_aws(
    endpoint_name: str,
    test_file: AudioFilename = "10min.flac",
    params: ModelArgs | None = None,
) -> tuple[list[Segment], TranscriptionInfo]:
    """Test a deployed AsyncEndpoint."""
    predictor = AsyncPredictor(
        HuggingFacePredictor(
            endpoint_name=endpoint_name,
            serializer=DataSerializer(content_type="audio/x-audio"),
        ),
        name=endpoint_name,
    )
    if not predictor:
        raise RuntimeError
    waiter_config = WaiterConfig(max_attempts=500, delay=2)
    with (
        _log_stats(test_file, endpoint_name),
        get_test_audio(test_file).open("rb") as f,
    ):
        if params:
            response = predictor.predict(
                data=f.read(),
                initial_args=_compress_args(params),
                waiter_config=waiter_config,
            )
        else:
            response = predictor.predict(data=f.read(), waiter_config=waiter_config)
        return _parse_response(response)


def _compress_args(params: ModelArgs) -> dict[str, str]:
    return {
        "CustomAttributes": base64.b64encode(
            brotli.compress(params.model_dump_json().encode("utf-8"))
        ).decode("utf-8")
    }


def _parse_response(
    response: Any,
) -> tuple[list[Segment], TranscriptionInfo]:
    ta = TypeAdapter(tuple[list[Segment], TranscriptionInfo])
    return ta.validate_python(response)


@contextmanager
def _log_stats(test_file: str, endpoint_name: str) -> Generator[None, None, None]:
    now = time.time()
    _logger.info("Starting prediction for file %s", test_file)
    try:
        yield
    finally:
        _logger.info(
            "Took %ss, %s, %s",
            f"{time.time() - now:.1f}",
            f"{test_file=}",
            f"{endpoint_name=}",
        )
