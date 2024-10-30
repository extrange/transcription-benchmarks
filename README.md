# Whisper in Sagemaker

Repo for experiments on speech transcription.

Refer [here][test-files] for test file descriptions.

All tests done on a `g4dn.xlarge` EC2 instance (16GB RAM, T4 GPU with 16GM VRAM).

| Model                                     | Speed | Peak VRAM | Config                             | Test File  |
| ----------------------------------------- | ----- | --------- | ---------------------------------- | ---------- |
| openai/whisper-large-v3                   | 31.5x | 9792MB    | fp16, batch=24, SDPA               | 1hour.flac |
| distil-whisper/distil-large-v3            | 87.1x | 2667MB    | fp16, batch=24, SDPA               | 1hour.flac |
| distil-whisper/distil-large-v3            | 62.0x |           | fp16, batch=24                     | 1hour.flac |
| openai/whisper-large-v3-turbo             | 81.8x | 2772MB    | fp16, batch=24, SDPA               | 1hour.flac |
| openai/whisper-large-v3-turbo             | 31.4x | 2772MB    | fp16, batch=None, SDPA             | 1hour.flac |
| nvidia/canary-1b                          |       | OOM       | vanilla                            | 1hour.flac |
| Systran/faster-whisper-large-v3           | 10.5x |           | batch=None, beam_size=1            | 1hour.flac |
| Systran/faster-whisper-large-v3           | 51.8x |           | batch=24, beam_size=1              | 1hour.flac |
| Systran/faster-whisper-large-v2           | 50.9x | 10GB      | batch=16, beam_size=1              | 1hour.flac |
| Systran/faster-whisper-large-v2           | 34x   | 11GB      | batch=16, beam_size=5              | 1hour.flac |
| Systran/faster-whisper-large-v2           |       | OOM       | batch=16, beam_size=5              | long.flac  |
| Systran/faster-whisper-large-v2           | 53.5x | 14GB      | batch=12, beam_size=5              | long.flac  |
| Systran/faster-whisper-large-v2           | 50.9x | 13GB      | batch=8, beam_size=5               | long.flac  |
| Systran/faster-whisper-large-v2           | 49x   | 12GB      | batch=6, beam_size=5               | long.flac  |
| Systran/faster-whisper-large-v2           | 49x   | ~11GB     | batch=6, beam_size=5, int8_float16 | long.flac  |
| Systran/faster-whisper-large-v2           | 11.9x | 6.8GB     | batch=None, beam_size=5            | 1hour.flac |
| Systran/faster-whisper-large-v2           | 18.3x | 6.8GB     | batch=None, beam_size=1            | 1hour.flac |
| Systran/faster-distil-whisper-large-v3    | 89.4x |           | batch=24, beam_size=1              | 1hour.flac |
| Systran/faster-distil-whisper-large-v3    | 83.7x |           | batch=24, beam_size=5              | 1hour.flac |
| deepdml/faster-whisper-large-v3-turbo-ct2 | 21.8x | ~4700MB   | batch=None, beam_size=5            | 1hour.flac |
| deepdml/faster-whisper-large-v3-turbo-ct2 | 76.2x | ~10237MB  | batch=24, beam_size=5              | 1hour.flac |

## Setup

You will need a GPU (e.g. an EC2 instance with a T4 GPU like `ml.g4dn.xlarge`, $0.70/hour), with the drivers installed. Installing the CUDA toolkit is unnecessary as PyTorch bundles its own CUDA runtime.

**Note**: Presently, `ctranslate2>=4.5.0` is [not able][ctranslate-issue] to locate and use the CuDNN v9 libraries included with Pytorch `2.*.*+cu124`. You have to point `LD_LIBRARY_PATH` to the Pytorch CuDNN libraries at `.venv/lib/python3.12/site-packages/nvidia/cudnn/lib`, or to your system CuDNN installation.

Create and sync the virtual environment with `uv sync`, then activate it with `source .venv/bin/activate`.

<details>
<summary>Installing a different version of Pytorch</summary>

The steps above will install the latest version of Pytorch, which is generally compatible with the latest CUDA drivers. If for some reason, you need to run an older version of Pytorch (e.g., if you are unable to upgrade your GPU drivers to support a later CUDA runtime version), follow these instructions.

Note: Pytorch ships with its own CUDA runtime API versions, which will work as long as they are [supported][CUDA compatibility] by the version of the driver.

To install Pytorch with a specific CUDA runtime version, edit your `pyproject.toml` file. For example, to install Pytorch compiled against CUDA runtime version 12.1:

```toml
[[tool.uv.index]]
url = "https://download.pytorch.org/whl/cu121"
name = "pytorch-cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }
```

Then run `uv add torch==2.5.1+cu121 torchvision torchaudio`. The correct versions of `torchvision` and `torchaudio` will be installed automatically.

Similarly, for the CUDA 12.4 runtime, modify the index urls in `pyproject.toml` accordingly and then run `uv add torch==2.5.1+cu124 torchvision torchaudio`.

</details>

## Benchmark

```python
from transcription_benchmarks.benchmark.bench import bench
res = bench("openai/whisper-large-v2")
print(res.get_text())
```

## Testing

`coverage run --branch -m pytest && coverage html`

## Notes

### nvidia-smi and CUDA runtime versions

`nvidia-smi` [does not show][nvidia-smi] does not show the CUDA runtime API version (what is commonly and confusingly referred to as 'CUDA version'). Instead, it shows the CUDA driver API of the installed driver (e.g. `535.182.01` ships with CUDA driver API `12.2` - full table [here][support matrix]).

### Quality Comparison

- `Systran/faster-whisper-large-v3` translates medical terms better than `Systran/faster-whisper-large-v2`, however repetition is much more frequent.
- `distil` models are all bad at multilingual speech.
- faster-whisper performs better than the native Transformers library.
- The best performing model for noisy/multilingual environments is `Systran/faster-whisper-large/v2`, presumably because of many enhancements by faster-whisper made to filter out low probability segments (I cannot replicate the exact parameters, e.g. the output of both are different even with batch_size=None and num_beams=5).
- The output of faster-whisper is only deterministic if `temperature=0`. Otherwise, the default parameters use repeatedly higher temperatures when repetition is detected.

## FAQ

### Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so} Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor

This is caused by `ctranslate>=4.5.0` not being able to locate the CuDNN v9 libraries. There is a [known incompatibility](https://github.com/SYSTRAN/faster-whisper/issues/1086) between Pytorch and CTranslate2 versions currently.

- If you are running torch `2.*.*+cu121` pin `ctranslate2==4.4.0`.
- If you are running torch `2.*.*+cu124` or later, follow the instructions in the [Setup section.](#setup) to point `LD_LIBRARY_PATH` to Pytorch's bundled CuDNN library.

### Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory

Similar to the above, however this time it is caused by `ctranslate <=4.4.0` not being able to locate CuDNN v8. You can use those bundled with Pytorch `2.*.*+cu121` by adding `path/to/your/venv/lib/python3.xx/site-packages/nvidia/cudnn/lib` to `LD_LIBRARY_PATH`.

### ImportError: /home/ubuntu/transcription-benchmarks/.venv/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: \_\_nvJitLinkComplete_12_4, version libnvJitLink.so.12

Pytorch is not using its bundled version of the CUDA Runtime libraries, and is instead using those on your system, which are incompatible.

Either [uninstall CuDNN and CUDA], or unset the environment variable `LD_LIBRARY_PATH` and ensure that scripts setting it in e.g. `~/.bashrc` or `/etc/profile.d` are removed.

### RuntimeError: cuDNN error: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH

PyTorch is using your system's CuDNN library and not its bundled one. Confirm this with `ldconfig -p | grep libcudnn`.

Either point `LD_LIBRARY_PATH` to PyTorch's bundled version, or [uninstall CuDNN and CUDA] from your system.

[test-files]: test_audio/readme.md
[support matrix]: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id5
[CUDA compatibility]: https://docs.nvidia.com/deploy/cuda-compatibility/index.html
[nvidia-smi]: https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi#comment93719643_53422407
[ctranslate-issue]: https://github.com/OpenNMT/CTranslate2/issues/1806
[uninstall CuDNN and CUDA]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit
