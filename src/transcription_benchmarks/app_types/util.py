from typing import Any

from faster_whisper_types.types import WhisperBatchOptions, WhisperOptions


def diff_fw_args(fw_args: WhisperOptions | WhisperBatchOptions) -> dict[str, Any]:
    """Return the arguments that were overridden by the user from the defaults."""
    if type(fw_args) is WhisperOptions:
        return WhisperOptions().dict_diff(fw_args)
    return WhisperBatchOptions().dict_diff(fw_args)
