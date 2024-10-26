from typing import Any

from transcription_benchmarks.app_types.bench import (
    FasterWhisperArgs,
    FasterWhisperBatchArgs,
)


def diff_fw_args(fw_args: FasterWhisperArgs | FasterWhisperBatchArgs) -> dict[str, Any]:
    """Return the arguments that were overridden by the user from the defaults."""
    if type(fw_args) is FasterWhisperArgs:
        return FasterWhisperArgs().dict_diff(fw_args)
    return FasterWhisperBatchArgs().dict_diff(fw_args)
