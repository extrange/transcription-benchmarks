import base64
import json
import logging
import time

import boto3
import brotli
from faster_whisper.transcribe import Segment, TranscriptionInfo
from sagemaker import Predictor
from sagemaker.huggingface.model import HuggingFacePredictor
from sagemaker.local import LocalSession
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.serializers import DataSerializer

from transcription_benchmarks.inference.systran_faster_whisper_large_v2.inference import (
    ModelArgs,
)
from transcription_benchmarks.misc.get_test_audio import AudioFilename, get_test_audio

_logger = logging.getLogger(__name__)


def predict_local(
    endpoint_name: str,
    test_file: AudioFilename = "10min.flac",
    params: ModelArgs | None = None,
) -> tuple[list[Segment], TranscriptionInfo] | TranscriptionInfo:
    """Test a local Sagemaker endpoint."""
    sagemaker_session = LocalSession(boto3.Session(region_name="ap-southeast-1"))
    sagemaker_session.config = {"local": {"local_code": True}}

    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=DataSerializer(content_type="audio/x-audio"),
    )

    return _predict(predictor, test_file, endpoint_name, params)


def predict_aws(
    endpoint_name: str,
    test_file: AudioFilename = "10min.flac",
    params: ModelArgs | None = None,
) -> tuple[list[Segment], TranscriptionInfo] | TranscriptionInfo:
    """Test a deployed AsyncEndpoint."""
    model = AsyncPredictor(
        HuggingFacePredictor(
            endpoint_name=endpoint_name,
            serializer=DataSerializer(content_type="audio/x-audio"),
        ),
        name=endpoint_name,
    )
    if not model:
        raise RuntimeError

    return _predict(model, test_file, endpoint_name, params)


def _predict(
    predictor: Predictor | AsyncPredictor,
    test_file: AudioFilename,
    endpoint_name: str,
    params: ModelArgs | None = None,
) -> tuple[list[Segment], TranscriptionInfo] | TranscriptionInfo:
    now = time.time()
    _logger.info("Starting prediction for file %s", test_file)
    with get_test_audio(test_file).open("rb") as f:
        if params:
            compressed = base64.b64encode(
                brotli.compress(params.model_dump_json().encode("utf-8"))
            )
            response = json.loads(
                predictor.predict(
                    data=f.read(),
                    initial_args={"CustomAttributes": compressed},
                )
            )
        else:
            response = json.loads(predictor.predict(data=f.read()))
    _logger.info(response)
    _logger.info(
        "Took %ss, %s, %s",
        f"{time.time() - now:.1f}",
        f"{test_file=}",
        f"{endpoint_name=}",
    )
    if params and params.task == "detect_language":
        return TranscriptionInfo(*response)
    return [Segment(*s) for s in response[0]], TranscriptionInfo(*response[1])
