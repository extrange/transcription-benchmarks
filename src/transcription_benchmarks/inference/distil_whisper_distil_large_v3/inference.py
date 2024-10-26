"""
Custom inference script run by Sagemaker's Multi Model Server.

For information on overriding functions, see https://huggingface.co/docs/sagemaker/en/inference#user-defined-code-and-modules.

Source of calling script:
https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/main/src/sagemaker_huggingface_inference_toolkit/handler_service.py
"""  # noqa: INP001

from typing import TYPE_CHECKING, Any

import torch
from transformers import pipeline

if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline


def model_fn(model_dir: str, *_args: Any) -> "Pipeline":
    """
    Override the default method for loading a model. The return value model will be used in predict for predictions.

    `model_dir`:  the path to your unzipped model.tar.gz.
    """
    return pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        torch_dtype=torch.float16,
        device="cuda:0",
    )


def predict_fn(data: dict, model: "Pipeline"):  # noqa: ANN201
    """
    Override the default method for prediction.

    `data`: dict with key 'inputs' containing raw bytes
    `model`: The output of `model_fn`
    """
    return model(
        data["inputs"],
        # Enable chunking - faster, but slightly worse performance
        # https://huggingface.co/openai/whisper-large-v3#chunked-long-form
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
        generate_kwargs={"language": "english"},
    )
