import logging
from pathlib import Path

import torch
from transformers import pipeline

from app_types.models import MODEL_DIR, Model
from misc.setup_logging import setup_logging

setup_logging()
_logger = logging.getLogger(__name__)


def download_hf_model(model: Model, save_dir: Path = MODEL_DIR) -> Path:
    """
    Download a HuggingFace model to a directory.

    Returns path to the model directory.
    """
    target_dir = get_model_dir(model, save_dir)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        torch_dtype=torch.float16,
        device="cuda:0",
    )
    pipe.save_pretrained(target_dir)
    _logger.info("Downloaded model '%s' to %s", model, target_dir)
    return target_dir


def get_model_dir(model: Model, base_dir: Path = MODEL_DIR) -> Path:
    """Return path to a model."""
    return base_dir / model.replace("/", "-")
