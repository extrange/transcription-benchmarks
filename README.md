# Whisper in Sagemaker

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

You will need a GPU (e.g. an EC2 instance with a T4 GPU like `ml.g4dn.xlarge`, $0.70/hour), with the drivers installed. Follow the instructions [here][install-cuda], as well as setting the [environment variables] especially `LD_LIBRARY_PATH`.

Create and sync the virtual environment with `uv sync`, then activate it with `source .vern/bin/activate`.

<details>
<summary>Installing a different version of Pytorch</summary>

The steps above will install the latest version of Pytorch, which is generally compatible with the latest CUDA drivers. If for some reason, you need to run an older version of Pytorch (e.g., if you are unable to upgrade your GPU drivers to support a later CUDA runtime version), follow these instructions.

If you are using an EC2 instance: For the AWS AMI image [`Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)`][ami-image], the default CUDA runtime version selected is 12.1 (set via `LD_LIBRARY_PATH`, via `/etc/profile.d/dlami.sh`). CUDA Toolkit runtime versions 12.2 and 12.3 are also pre-installed.

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

Similarly, for the CUDA 12.4 runtime. Modify the index urls in `pyproject.toml` accordingly and then run `uv add torch==2.5.1+cu124 torchvision torchaudio`.

</details>

## Benchmark

```python
from transcription_benchmarks.benchmark.bench import bench
res = bench("openai/whisper-large-v2")
print(res.get_text())
```

## Deployment

### Deploy Locally

Edit `compose.yaml` with the appropriate volume mounts to your code directory, then run `predict_local`:

```python
from transcription_benchmarks.util.predict import predict_local

predict_local()
```

### Deploy to AWS

```python
from transcription_benchmarks.util.deploy import deploy_on_aws
from transcription_benchmarks.util.predict import predict_aws

endpoint = deploy_on_aws("your-endpoint-name","/home/ubuntu/transcription-benchmarks/models/Systran-faster-whisper-large-v2/model_artifact.tar.gz", "/home/ubuntu/transcription-benchmarks/src/transcription_benchmarks/inference/systran_faster_whisper_large_v2")

predict_aws(endpoint)
```

<details>
<summary>Note on CUDA runtime versions in Sagemaker</summary>

The SageMaker model container uses the NVIDIA GPU Driver version 470.256.02 by default, which will only support CUDA runtime versions up to 11.x. CUDA runtime 12.x requires GPU driver version >= 525.60.13 ([support matrix], see also [CUDA compatibility]).

To use a later GPU driver version, override the Sagemaker Session's `create_endpoint_config` method with `InferenceAmiVersion: al2-ami-sagemaker-inference-gpu-2`. This provides the more recent [GPU driver version 535.183.01][sagemaker-image] which supports CUDA runtime versions 12.x.

</details>

## Testing

`coverage run --branch -m pytest && coverage html`

To include tests for local endpoints, set the env var `LOCAL=1`.

To include tests for AWS endpoints, set the env var `AWS_ENDPOINT=aws-endpoint-name`.

## Notes

### nvidia-smi and CUDA runtime versions

`nvidia-smi` [does not show][nvidia-smi] does not show the CUDA runtime API version (what is commonly and confusingly referred to as 'CUDA version'). Instead, it shows the CUDA driver API of the installed driver (e.g. `535.182.01` ships with CUDA driver API `12.2` - full table [here][support matrix]).

Pytorch ships with its own CUDA runtime API versions, which will work as long as they are [supported][CUDA compatibility] by the version of the driver.

### Quality Comparison

- `Systran/faster-whisper-large-v3` translates medical terms better than `Systran/faster-whisper-large-v2`, however repetition is much more frequent.
- `distil` models are all bad at multilingual speech.
- faster-whisper performs better than the native Transformers library.
- The best performing model for noisy/multilingual environments is `Systran/faster-whisper-large/v2`, presumably because of many enhancements by faster-whisper made to filter out low probability segments (I cannot replicate the exact parameters, e.g. the output of both are different even with batch_size=None and num_beams=5).
- The output of faster-whisper is only deterministic if `temperature=0`. Otherwise, the default parameters use repeatedly higher temperatures when repetition is detected.

### Sagemaker Instance Comparisons

[Link][sagemaker-pricing]

- p2: Nvidia K80, 12GB VRAM/GPU
- p3: Nvidia V100, 16GB VRAM/GPU
- p4: Nvidia A100, 40GB VRAM/GPU
- g4dn: Nvidia T4, 16GB VRAM/GPU
- g5/g5n: Nvidia A10G, 24GB VRAM/GPU
- g6: Nvidia L4, 16GB VRAM/GPU

[Speed comparison]\: A100 > V100 > A10G > L4 > T4 > K80

## FAQ

### Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}

You are running torch `2.*.*+cu121` and have `ctranslate>=4.5.0` installed, which is a [known incompatibility](https://github.com/SYSTRAN/faster-whisper/issues/1086). Either pin `ctranslate2==4.4.0` or use torch `2.*.*+cu124` or later.

### Sagemaker 'Worked Died'

The underlying EC2 instance ran out of GPU memory.

### ImportError: /home/ubuntu/transcription-benchmarks/.venv/lib/python3.12/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: \_\_nvJitLinkComplete_12_4, version libnvJitLink.so.12

You are using a Pytorch CUDA runtime version which is different from the default CUDA runtime version installed on your system. Either use the bundled Pytorch CUDA runtime by setting `LD_LIBRARY_PATH` to point to `path/to/your/venv/lib/python3.xx/site-packages/nvidia/nvjitlink`, or upgrade your system's CUDA Toolkit a version equal to or later than that of Pytorch's.

[mms-env-vars]: https://github.com/awslabs/multi-model-server/blob/master/docs/configuration.md
[configmanager.java]: https://github.com/awslabs/multi-model-server/blob/master/frontend/server/src/main/java/com/amazonaws/ml/mms/util/ConfigManager.java
[test-files]: test_audio/readme.md
[support matrix]: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id5
[CUDA compatibility]: https://docs.nvidia.com/deploy/cuda-compatibility/index.html
[sagemaker-image]: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html#sagemaker-Type-ProductionVariant-InferenceAmiVersion
[nvidia-smi]: https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi#comment93719643_53422407
[ami-image]: https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-22-04/
[sagemaker-pricing]: https://aws.amazon.com/sagemaker/pricing/
[Speed comparison]: https://www.domainelibre.com/comparing-the-performance-and-cost-of-a100-v100-t4-gpus-and-tpu-in-google-colab/
[install-cuda]: https://developer.nvidia.com/cuda-downloads
[environment variables]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup
