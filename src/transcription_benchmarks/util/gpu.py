import torch


def gpu_available() -> bool:
    """Whether GPU is available on this system."""
    return torch.cuda.is_available()
