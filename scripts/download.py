import logging

import torch
from transformers import pipeline

from misc.models import MODEL_DIR, Model
from misc.setup_logging import setup_logging

setup_logging()
_logger = logging.getLogger(__name__)


def download_model(model: Model) -> None:
    """Download a HuggingFace model to a the `models/` directory."""

    target_dir = MODEL_DIR / model.replace("/", "-")
    _logger.info("Downloading model to %s", target_dir)
    if target_dir.exists():
        _logger.error("%s already exists, bailing", target_dir)
        return

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        torch_dtype=torch.float16,
        device="cuda:0",
    )

    pipe.save_pretrained(target_dir)
