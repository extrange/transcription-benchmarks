import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from transcription_benchmarks.app_types.models import MODEL_DIR, Model
from transcription_benchmarks.misc.setup_logging import setup_logging

setup_logging()
_logger = logging.getLogger(__name__)


def download_hf_model(model: Model, save_dir: Path = MODEL_DIR) -> Path:
    """
    Download a HuggingFace model to a directory.

    Returns path to the model directory.
    """
    target_dir = get_model_dir(model, save_dir)
    snapshot_download(
        model,
        local_dir=target_dir,
        repo_type="model",
        ignore_patterns=["*.msgpack", "*.h5"],  # Ignore FLAX and TF
    )
    _logger.info("Downloaded model '%s' to %s", model, target_dir)
    return target_dir


def get_model_dir(model: Model, base_dir: Path = MODEL_DIR) -> Path:
    """Return path to a model."""
    return base_dir / model.replace("/", "-")
