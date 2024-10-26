"""
Custom inference script run by Sagemaker's Multi Model Server.

For information on overriding functions, see https://huggingface.co/docs/sagemaker/en/inference#user-defined-code-and-modules.

Source of calling script:
https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/main/src/sagemaker_huggingface_inference_toolkit/handler_service.py
"""  # noqa: INP001

from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline


def model_fn(model_dir: str, *_args: Any) -> "Pipeline":
    """
    Override the default method for loading a model. The return value model will be used in predict for predictions.

    `model_dir`:  the path to your unzipped model.tar.gz.
    """

    # ylacombe/whisper-large-v3-turbo will not work if these are not specified explicitly
    device = "cuda:0"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_dir)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
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
