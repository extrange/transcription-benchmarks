"""
Deploy SageMaker models locally or on AWS.

Note: AWS credentials should be configured on the local machine prior to running, otherwise images cannot be pulled from ECR.
"""

import logging
import os
from typing import cast

import boto3
from sagemaker import Session
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.huggingface.model import HuggingFaceModel, Predictor
from sagemaker.local import LocalSession
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.serializers import DataSerializer
from sagemaker.utils import name_from_base

from misc.setup_logging import setup_logging

setup_logging()
_logger = logging.getLogger(__name__)


def deploy_locally(
    model_artifacts: str, code_dir: str, inference_filename: str = "inference.py"
) -> Predictor:
    """
    Deploy a model with custom inference code locally (on GPU), using Sagemaker [Local Mode](https://sagemaker.readthedocs.io/en/stable/overview.html#local-mode). Requires Docker to be installed.

    Run this function in a Jupyter Notebook.

    `model_artifacts`: Path to the model files, packaged as a .tar.gz file(downloaded for the specific version of Transformers). Will be extracted and then bind mounted to the docker container at `/opt/ml/model:rw`.
    `code_dir`: Parent directory of inference code, bind mounted to to /opt/ml/code:rw.
    `inference_filename`: Name of inference file to run, within `code_dir`.

    Returns a `Predictor` you can use, such as:

    ```python
    predictor = deploy("models/distil-whisper-large-v3", "")
    with open("your file", "rb) as f:
        predictor.predict(f.read())
    ```

    TODO: Speedup by removing the repacking step
    """

    # Fix issues with .pyc files in code_dir being owned by root
    _logger.info("chowning %s", code_dir)
    os.system(f"sudo chown -R ubuntu {code_dir}")  # noqa: S605

    sagemaker_session = LocalSession(boto3.Session(region_name="ap-southeast-1"))
    sagemaker_session.config = {"local": {"local_code": True}}

    model_name = name_from_base("local")

    # Dummy role in the real account for local training
    role = "arn:aws:iam::189606967604:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

    # HuggingFaceModel: https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html#hugging-face-model
    # Sagemaker Model: https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model
    huggingface_model = HuggingFaceModel(
        name=model_name,
        model_data=f"file://{model_artifacts}",
        # Enable GPU support
        image_uri="763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04",
        sagemaker_session=sagemaker_session,
        role=role,
        source_dir=code_dir,
        entry_point=inference_filename,
        # See README.md for why this is necessary
        env={
            "MMS_MAX_REQUEST_SIZE": "2000000000",
            "MMS_MAX_RESPONSE_SIZE": "2000000000",
            "MMS_DEFAULT_RESPONSE_TIMEOUT": "900",
        },
    )

    _logger.info("Deploying locally...")
    # https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="local_gpu",
        endpoint_name=model_name,
    )

    if not predictor:
        err = "Predictor is None!"
        raise RuntimeError(err)

    _logger.info("Deployment successful.")
    _logger.info("Endpoint name: %s", predictor.endpoint_name)

    predictor.serializer = DataSerializer(content_type="audio/x-audio")

    # In local mode, we don't get an AsyncPredictor
    return cast(Predictor, predictor)


def deploy_on_aws(  # noqa: PLR0913
    model_and_endpoint_name: str,
    model_data: str,
    source_dir: str,
    role: str = "arn:aws:iam::189606967604:role/service-role/AmazonSageMaker-ExecutionRole-20240503T161356",
    entry_point: str = "inference.py",
    bucket_name: str | None = None,
) -> AsyncPredictor:
    """
    Deploy a model with custom inference code to AWS.

    `model_and_endpoint_name`: Will be appended with the current timestamp.

    `model_data`: Path to the HuggingFace model files, in a tar.gz file, e.g. `models/distil-whisper-large-v3/model_artifact.tar.gz`. Note: this should not contain custom inference code.

    `source_dir`: Path to the directory containing your custom inference code, e.g. `models/distil-whisper-large-v3/code`. This does not need to be in `model_data`.

    `role`: IAM Role for Sagemaker to use.

    `entry_point`: Filename of the custom inference script, within `source_dir`.

    `bucket_name`: Name of S3 bucket where output/failure responses from the model will be saved.
    """
    session = Session()
    sagemaker_default_bucket = session.default_bucket()

    _safe_name = name_from_base(model_and_endpoint_name.replace("/", "-"))

    async_config = AsyncInferenceConfig(
        output_path=f"s3://{bucket_name or sagemaker_default_bucket}/output/",
        failure_path=f"s3://{bucket_name or sagemaker_default_bucket}/failure/",
    )

    # create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        model_data=model_data,
        transformers_version="4.37.0",
        pytorch_version="2.1.0",
        py_version="py310",
        source_dir=source_dir,  # Seems like this directory is only checked (and used) when model_data is supplied
        entry_point=entry_point,
        role=role,
        name=_safe_name,
        env={  # See readme on why this is necessary.
            "MMS_MAX_REQUEST_SIZE": "2000000000",
            "MMS_MAX_RESPONSE_SIZE": "2000000000",
            "MMS_DEFAULT_RESPONSE_TIMEOUT": "900",
        },
    )

    predictor = huggingface_model.deploy(
        endpoint_name=_safe_name,
        initial_instance_count=1,
        instance_type="ml.g4dn.xlarge",
        async_inference_config=async_config,
    )

    if predictor is None:
        err = "predictor is None!"
        raise RuntimeError(err)

    return predictor
