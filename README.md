# Whisper in Sagemaker

Refer [here][test-files] for test file descriptions.

All tests done on a `g4dn.xlarge` EC2 instance (16GB RAM, T4 GPU with 16GM VRAM).

| Model                                     | Speed | Peak VRAM | Config                  | Test File  |
| ----------------------------------------- | ----- | --------- | ----------------------- | ---------- |
| openai/whisper-large-v3                   | 31.5x | 9792MB    | fp16, batch=24, SDPA    | 1hour.flac |
| distil-whisper/distil-large-v3            | 87.1x | 2667MB    | fp16, batch=24, SDPA    | 1hour.flac |
| distil-whisper/distil-large-v3            | 62.0x |           | fp16, batch=24          | 1hour.flac |
| openai/whisper-large-v3-turbo             | 81.8x | 2772MB    | fp16, batch=24, SDPA    | 1hour.flac |
| openai/whisper-large-v3-turbo             | 31.4x | 2772MB    | fp16, batch=None, SDPA  | 1hour.flac |
| nvidia/canary-1b                          |       | OOM       | vanilla                 | 1hour.flac |
| Systran/faster-whisper-large-v3           | 10.5x |           | batch=None, beam_size=1 | 1hour.flac |
| Systran/faster-whisper-large-v3           | 51.8x |           | batch=24, beam_size=1   | 1hour.flac |
| Systran/faster-whisper-large-v2           | 50.9x | 10GB      | batch=16, beam_size=1   | 1hour.flac |
| Systran/faster-whisper-large-v2           | 34x   | 11GB      | batch=16, beam_size=5   | 1hour.flac |
| Systran/faster-whisper-large-v2           |       | OOM       | batch=16, beam_size=5   | long.flac  |
| Systran/faster-whisper-large-v2           | 53.5x | 14GB      | batch=12, beam_size=5   | long.flac  |
| Systran/faster-whisper-large-v2           | 50.9x | 13GB      | batch=8, beam_size=5    | long.flac  |
| Systran/faster-whisper-large-v2           | 49x   | 12GB      | batch=6, beam_size=5    | long.flac  |
| Systran/faster-whisper-large-v2           | 11.9x | 6.8GB     | batch=None, beam_size=5 | 1hour.flac |
| Systran/faster-whisper-large-v2           | 18.3x | 6.8GB     | batch=None, beam_size=1 | 1hour.flac |
| Systran/faster-distil-whisper-large-v3    | 89.4x |           | batch=24, beam_size=1   | 1hour.flac |
| Systran/faster-distil-whisper-large-v3    | 83.7x |           | batch=24, beam_size=5   | 1hour.flac |
| deepdml/faster-whisper-large-v3-turbo-ct2 | 21.8x | ~4700MB   | batch=None, beam_size=5 | 1hour.flac |
| deepdml/faster-whisper-large-v3-turbo-ct2 | 76.2x | ~10237MB  | batch=24, beam_size=5   | 1hour.flac |

Notes:

- `distil` models are all bad at multilingual speech.
- The best performing model for noisy/multilingual environments is `openai/whisper-large-v2`, with the faster-whisper CT2 implementation performing better than the Transformers one, presumably because of many enhancements made to filter out low probability segments (I cannot replicate the exact parameters, e.g. the output of both are different even with batch_size=None and num_beams=5).
- SDPA is only available in Pytorch 2.1.1 and above. In order to use it in the SageMaker images, the PyTorch version is overridden via `requirements.txt`. Tested in Sagemaker Local Mode.
- faster-whisper doesn't really provide significant speedups. Consider investigating whether total memory usage is lower, however.

## Setup

Run in an EC2 instance with a T4 GPU (`ml.g4dn.xlarge`, $0.70/hour).

This project uses `uv` for dependency management.

Create a virtual environment with `uv venv`. Subsequently, configure VSCode to use that environment.

After activating that environment, you can then run modules with `python -m your.module`.

## Usage

```python
# Benchmark
from benchmark.bench import bench
res = bench("openai/whisper-large-v2")
print(res.get_text())

# Deploy
from scripts.deploy import deploy_locally

predictor = deploy_locally("models/ylacombe-whisper-large-v3-turbo/model_artifact.tar.gz", "/home/ubuntu/whisper/models/code")

# Predict
from scripts.predict import predict_local

predict_local(predictor.endpoint_name, "10min.flac")
```

## Testing

`coverage run --branch -m pytest && coverage html`

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
