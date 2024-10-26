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

Notes:

- SDPA is only available in Pytorch 2.1.1 and above.
- The SageMaker model container uses the NVIDIA GPU Driver version 470.256.02, which will only support CUDA 11.8 (regardless of the image used). CUDA 12.x requires version >= 525.60.13 ([support matrix]). Verified on both the gd4n.xlarge and g4dn.2xlarge instances.
  - This is because the default AMI uses that driver. To get around this, use the Sagemaker Session's `create_endpoint_config` method with `InferenceAmiVersion: al2-ami-sagemaker-inference-gpu-2`. This will give you the GPU driver version 535.183.01 with CUDA Version: 12.2.
- For faster-whisper, [CTranslate2 4.0.0 does not work with CUDA 11.8][ctranslate2]. The last version supporting CUDA 11.8 is ctranslate2==3.24.0.
- [List of available Sagemaker Deep Learning Container images][sagemaker-dlc-images]. Make sure you use the `ap-southeast-1` region or you will get auth errors.

## Setup

Run in an EC2 instance with a T4 GPU (`ml.g4dn.xlarge`, $0.70/hour).

This project uses `uv` for dependency management.

Create a virtual environment with `uv venv`. Subsequently, configure VSCode to use that environment.

After activating that environment, you can then run modules with `python -m your.module`.

## Usage

Benchmark:

```python
from transcription_benchmarks.benchmark.bench import bench
res = bench("openai/whisper-large-v2")
print(res.get_text())
```

Local deployment: Edit `compose.yaml` with the appropriate volume mounts to your code directory, then run `predict_local`.

```python
from transcription_benchmarks.util.predict import predict_local

predict_local()
```

Deploy to AWS:

```python
from transcription_benchmarks.util.deploy import deploy_on_aws
from transcription_benchmarks.util.predict import predict_aws

endpoint = deploy_on_aws("your-endpoint-name","/home/ubuntu/transcription-benchmarks/models/Systran-faster-whisper-large-v2/model_artifact.tar.gz", "/home/ubuntu/transcription-benchmarks/src/transcription_benchmarks/inference/systran_faster_whisper_large_v2")

predict_aws(endpoint)
```

## Quality Notes

- `Systran/faster-whisper-large-v3` translates medical terms better than `Systran/faster-whisper-large-v2`, however repetition is much more frequent.
- `distil` models are all bad at multilingual speech.
- faster-whisper performs better than the native Transformers library.
- The best performing model for noisy/multilingual environments is `Systran/faster-whisper-large/v2`, presumably because of many enhancements by faster-whisper made to filter out low probability segments (I cannot replicate the exact parameters, e.g. the output of both are different even with batch_size=None and num_beams=5).
- The output of faster-whisper is only deterministic if `temperature=0`. Otherwise, the default parameters use repeatedly higher temperatures when repetition is detected.

## Testing

`coverage run --branch -m pytest && coverage html`

To include tests for local endpoints, set the env var `LOCAL=1`.

To include tests for AWS endpoints, set the env var `AWS_ENDPOINT=aws-endpoint-name`.

## Sagemake Pricing and Instances

https://aws.amazon.com/sagemaker/pricing/

- p2: Nvidia K80, 12GB VRAM/GPU
- p3: Nvidia V100, 16GB VRAM/GPU
- p4: Nvidia A100, 40GB VRAM/GPU
- g4dn: Nvidia T4, 16GB VRAM/GPU
- g5/g5n: Nvidia A10G, 24GB VRAM/GPU
- g6: Nvidia L4, 16GB VRAM/GPU

Speed: A100 > V100 > A10G > L4 > T4 > K80

https://www.domainelibre.com/comparing-the-performance-and-cost-of-a100-v100-t4-gpus-and-tpu-in-google-colab/

## MMS Config

The MMS (Multi Model Server) stores its configuration in `/etc/sagemaker-mms.properties`. By default, `enable_envvars_config=true` is set, so you can configure the MMS via setting [environment variables][mms-env-vars] on the container.

The following important environment variables are configured in [`ConfigManager.java`][configmanager.java] (they are not documented):

```
MMS_MAX_REQUEST_SIZE=6553500 (6.5MB)
MMS_MAX_RESPONSE_SIZE=6553500
MMS_DEFAULT_RESPONSE_TIMEOUT=120 (in minutes)
```

They must be increased (via passing as a dictionary to the argument `env` in `HuggingFaceModel`), otherwise you will get 413 errors.

## Local Mode

Sagemaker local mode resources: https://github.com/aws-samples/amazon-sagemaker-local-mode

Includes information on using your own containers.

## Pyannote

https://aws.amazon.com/blogs/machine-learning/deploy-a-hugging-face-pyannote-speaker-diarization-model-on-amazon-sagemaker-as-an-asynchronous-endpoint/

diariaztion: can pass use_auth_token https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/94b1d8408684581c4e7b056eb366e0051197c418/src/sagemaker_huggingface_inference_toolkit/mms_model_server.py#L80

realtime diarization: https://github.com/juanmc2005/diart
reverb diarization: https://huggingface.co/Revai/reverb-diarization-v2

## FAQ

### Sagemaker 'Worked Died'

The underlying EC2 instance ran out of GPU memory.

[mms-env-vars]: https://github.com/awslabs/multi-model-server/blob/master/docs/configuration.md
[configmanager.java]: https://github.com/awslabs/multi-model-server/blob/master/frontend/server/src/main/java/com/amazonaws/ml/mms/util/ConfigManager.java
[test-files]: test_audio/readme.md
[support matrix]: https://docs.nvidia.com/deploy/cuda-compatibility/
[ctranslate2]: https://github.com/SYSTRAN/faster-whisper/issues/734
[sagemaker-dlc-images]: https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sagemaker-framework-containers-sm-support-only
