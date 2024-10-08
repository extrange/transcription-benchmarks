# Whisper in Sagemaker

Using [audio file][1hour.flac] of length 1:13:10 (4390s) for all tests.

| Model                           | Speed | Peak VRAM | Config                                |
|---------------------------------|-------|-----------|---------------------------------------|
| openai/whisper-large-v3         | 31.5x | 9792MB    | fp16, batch=24, SDPA                  |
| distil-whisper/distil-large-v3  | 87.1x | 2667MB    | fp16, batch=24, SDPA                  |
| distil-whisper/distil-large-v3  | 62.0x |           | fp16, batch=24                        |
| ylacombe/whisper-large-v3-turbo | 81.8x | 2772MB    | fp16, batch=24, SDPA                  |
| ylacombe/whisper-large-v3-turbo | 31.4x | 2772MB    | fp16, batch=None, SDPA                |
| nvidia/canary-1b                |       | OOM       | vanilla                               |
| openai/whisper-large-v3         | 10.5x |           | faster-whisper, beam_size=1           |
| openai/whisper-large-v3         | 51.8x |           | faster-whisper, batch=24, beam_size=1 |
| distil-whisper/distil-large-v3  | 89.4x |           | faster-whisper, batch=24, beam_size=1 |
| distil-whisper/distil-large-v3  | 83.7x |           | faster-whisper, batch=24, beam_size=5 |
| ylacombe/whisper-large-v3-turbo | 21.8x | ~4700MB   | faster-whisper, batch=None            |
| ylacombe/whisper-large-v3-turbo | 76.2x | ~10237MB  | faster-whisper, batch=24              |

Notes:

- The best performing model for noisy/multilingual environments is `openai/whisper-large-v2`.
- SDPA is only available in Pytorch 2.1.1 and above. In order to use it in the SageMaker images, the PyTorch version is overridden via `requirements.txt`. Tested in Sagemaker Local Mode.
- faster-whisper doesn't really provide significant speedups. Consider investigating whether total memory usage is lower, however.

## Setup

Run in an EC2 instance with a T4 GPU (`ml.g4dn.xlarge`, $0.70/hour).

This project uses `uv` for dependency management.

Create a virtual environment with `uv venv`. Subsequently, configure VSCode to use that environment.

After activating that environment, you can then run modules with `python -m your.module`.

Usage:

```python
# Deploy
from scripts.deploy import deploy_locally

predictor = deploy_locally("models/ylacombe-whisper-large-v3-turbo/model_artifact.tar.gz", "/home/ubuntu/whisper/models/code")

# Predict
from scripts.predict import predict_local

predict_local(predictor.endpoint_name, "10min.flac")
```

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

## FAQ

### Sagemaker 'Worked Died'

The underlying EC2 instance ran out of GPU memory.

[mms-env-vars]: https://github.com/awslabs/multi-model-server/blob/master/docs/configuration.md
[configmanager.java]: https://github.com/awslabs/multi-model-server/blob/master/frontend/server/src/main/java/com/amazonaws/ml/mms/util/ConfigManager.java
[1hour.flac]: test_audio/1hour.flac
