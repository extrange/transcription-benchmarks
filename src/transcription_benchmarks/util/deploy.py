"""
Deploy SageMaker models locally or on AWS.

Note: AWS credentials should be configured on the local machine prior to running, otherwise images cannot be pulled from ECR.
"""

import logging

import boto3
from sagemaker import Session
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.utils import name_from_base

from transcription_benchmarks.misc.setup_logging import setup_logging

_logger = logging.getLogger(__name__)


def deploy_on_aws(  # noqa: PLR0913
    model_and_endpoint_name: str,
    model_data: str,
    source_dir: str,
    role: str = "arn:aws:iam::189606967604:role/service-role/AmazonSageMaker-ExecutionRole-20240503T161356",
    entry_point: str = "inference.py",
    bucket_name: str | None = None,
) -> str:
    """
    Deploy a model with custom inference code to AWS.

    `model_and_endpoint_name`: Will be appended with the current timestamp.

    `model_data`: Path to the HuggingFace model files, in a tar.gz file, e.g. `models/distil-whisper-large-v3/model_artifact.tar.gz`. Note: this should not contain custom inference code.

    `source_dir`: Path to the directory containing your custom inference code, e.g. `models/distil-whisper-large-v3/code`. This does not need to be in `model_data`.

    `role`: IAM Role for Sagemaker to use.

    `entry_point`: Filename of the custom inference script, within `source_dir`.

    `bucket_name`: Name of S3 bucket where output/failure responses from the model will be saved.

    Returns the name of the endpoint.
    """
    setup_logging()
    session = Session()
    sagemaker_default_bucket = session.default_bucket()

    _model_name_safe = name_from_base(model_and_endpoint_name.replace("/", "-"))

    # create Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        model_data=model_data,
        transformers_version="4.37.0",
        pytorch_version="2.1.0",
        py_version="py310",
        source_dir=source_dir,  # Seems like this directory is only checked (and used) when model_data is supplied
        entry_point=entry_point,
        role=role,
        name=_model_name_safe,
        env={  # See readme on why this is necessary.
            "MMS_MAX_REQUEST_SIZE": "2000000000",
            "MMS_MAX_RESPONSE_SIZE": "2000000000",
            "MMS_DEFAULT_RESPONSE_TIMEOUT": "900",
        },
    )

    # Create the model, but don't deploy it
    _logger.info("Creating model...")
    huggingface_model.create(instance_type="ml.g4dn.xlarge")
    _logger.info("Model created.")

    # Upgrade the AMI image so we have CUDA 12.4
    client = boto3.client("sagemaker")
    client.create_endpoint_config(
        EndpointConfigName=_model_name_safe,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": _model_name_safe,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.g4dn.xlarge",
                "InitialVariantWeight": 1.0,
                "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-2",
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://{bucket_name or sagemaker_default_bucket}/output/",
                "S3FailurePath": f"s3://{bucket_name or sagemaker_default_bucket}/failure/",
            }
        },
        EnableNetworkIsolation=False,
    )
    _logger.info("Created endpoint config '%s'. Creating endpoint...", _model_name_safe)
    client.create_endpoint(
        EndpointName=_model_name_safe, EndpointConfigName=_model_name_safe
    )
    waiter = client.get_waiter("endpoint_in_service")
    _logger.info("Waiting for endpoint '%s' to be InService...", _model_name_safe)
    waiter.wait(EndpointName=_model_name_safe)
    return _model_name_safe
