# Nemotron ASR word boosting guide

A small, container-first guide to dynamic word boosting with NVIDIA Nemotron
ASR Streaming NIM. It shows the baseline transcript, applies request-time boost
phrases, and saves an A/B comparison as JSON.

No virtual environment or host Python installation is required. The ASR server
and the example client both run in containers.

> This repository uses `nvcr.io/nim/nvidia/nemotron-asr-streaming:1.4.0`
> with the English `batch_size=32` profile. That profile uses about 4 GB of GPU
> memory, plus normal runtime overhead.

## Directory

| I want to... | Go here |
| --- | --- |
| Run the example | [Quick start](#quick-start) |
| Understand word boosting | [How it works](#how-it-works) |
| Change the words or scores | [`config/boosting.json`](config/boosting.json) and [What to change](#what-to-change) |
| Read the smallest code example | [`examples/minimal.py`](examples/minimal.py) |
| Read the full A/B client | [`src/compare.py`](src/compare.py) |
| Reuse the Riva/NIM helpers | [`src/word_boosting.py`](src/word_boosting.py) |
| See every command | [Command reference](#command-reference) |
| Use my own WAV file | [Use your own audio](#use-your-own-audio) |
| Tune safely | [Tuning guide](#tuning-guide) |
| Fix a problem | [Troubleshooting](#troubleshooting) |

### Repository map

| Path | Purpose |
| --- | --- |
| [`compose.yaml`](compose.yaml) | Runs the GPU-backed NIM and CPU-only Python client |
| [`Dockerfile`](Dockerfile) | Builds the pinned client image |
| [`config/boosting.json`](config/boosting.json) | The phrases and scores you normally edit |
| [`examples/minimal.py`](examples/minimal.py) | Short, single-transcription example with obvious edit points |
| [`src/compare.py`](src/compare.py) | Baseline-versus-boosted command used by `make compare` |
| [`src/word_boosting.py`](src/word_boosting.py) | Configuration, Riva request, and streaming audio code |
| [`src/reporting.py`](src/reporting.py) | Match analysis and JSON report creation |
| [`tests/`](tests) | Fast tests, run inside the client container |
| [`Makefile`](Makefile) | Friendly wrappers around Docker Compose |

## How it works

Word boosting changes decoding, not training. The client adds one or more
`SpeechContext` entries to a request. Each context contains phrases and a score:

- A positive score makes a phrase more likely.
- A negative score makes a phrase less likely.
- A score of zero has no useful effect.
- Nemotron ASR Streaming accepts scores from `-200` through `200`.

The flow in this repository is:

```text
WAV file -> client container -> Riva gRPC -> Nemotron ASR NIM (GPU)
              |                                      |
              +-- baseline: no speech context        |
              +-- boosted: phrases + scores ---------+
```

The comparison matters: a boosted transcript alone cannot tell you whether the
model already recognized the term. The A/B client sends the same audio twice
and reports which target phrases appear in each transcript.

## Requirements

- Linux with an NVIDIA GPU and current NVIDIA driver.
- Docker Engine, Docker Compose v2, and NVIDIA Container Toolkit.
- An NGC API key with access to the Nemotron ASR Streaming NIM.
- Enough disk space for the NIM image and cached model artifacts.

The selected `batch_size=32` English profile is the smallest documented
Nemotron ASR Streaming profile. Larger profiles are intended for more
concurrent streams, not a better transcript for this tutorial.

## Quick start

Set your NGC API key in the shell. It is passed to the NIM container and is not
written to the repository:

```bash
export NGC_API_KEY='your-ngc-api-key'
```

Then run:

```bash
make doctor
make ngc-login
make build
make start
make wait
make samples
make compare
```

The first start can take a while because the NIM downloads and prepares model
artifacts. `make wait` stays attached until the HTTP readiness endpoint reports
that the service is ready.

Expected comparison for NVIDIA's word-boosting sample:

```text
BASELINE: aunt bertha and ab loper ...
BOOSTED:  AntiBERTa and ABlooper ...
NEW MATCHES: AntiBERTa, ABlooper
```

Exact punctuation and surrounding words may vary. The structured result is
written to `results/comparison.json`.

Stop the NIM when finished:

```bash
make stop
```

## What to change

For most experiments, edit only [`config/boosting.json`](config/boosting.json):

```json
{
  "language_code": "en-US",
  "boost_groups": [
    {
      "score": 20,
      "phrases": ["AntiBERTa", "ABlooper"]
    }
  ]
}
```

The important fields are:

| Field | Change it when... |
| --- | --- |
| `language_code` | You change to a compatible deployed language/profile |
| `boost_groups[].phrases` | You want different names, products, or domain terms |
| `boost_groups[].score` | You want a weaker or stronger bias for that group |

Different phrases can use different scores by creating more groups:

```json
{
  "language_code": "en-US",
  "boost_groups": [
    {"score": 20, "phrases": ["AntiBERTa", "ABlooper"]},
    {"score": 8, "phrases": ["graph neural networks"]},
    {"score": -5, "phrases": ["a phrase to discourage"]}
  ]
}
```

The client adds one Riva speech context per group. This is how Nemotron ASR can
apply different request-time scores to different phrases.

For the shortest possible code path, open
[`examples/minimal.py`](examples/minimal.py). The values marked `EDIT HERE` are
the server, audio path, phrase list, and score. Run it in the client container:

```bash
make minimal
```

## Use your own audio

Put a mono PCM WAV file in `data/`. Audio is ignored by Git, so it will not be
published accidentally. Then pass the path through the Makefile:

```bash
make compare \
  AUDIO=data/my-recording.wav \
  CONFIG=config/boosting.json \
  RESULT=results/my-recording.json
```

To keep several phrase sets, copy the example JSON and select another file:

```bash
cp config/boosting.json config/product-names.json
# Edit config/product-names.json, then:
make compare CONFIG=config/product-names.json
```

## Tuning guide

Start with a modest positive score such as `10` or `20`, then test on both:

1. Positive audio that really contains the target phrases.
2. Negative-control audio that does not contain them.

Increase the score only when target recall is still poor. A very large score
can insert a boosted term into unrelated audio. Case can matter, so include the
casing forms you expect, such as `NVIDIA` and `Nvidia`, when necessary.

Use the ordinary sample as a negative control:

```bash
make negative
```

The report flags any boosted phrase that appears in the negative-control
transcript. That warning is a tuning signal; it is not hidden or rewritten.

## Command reference

| Command | What it does |
| --- | --- |
| `make help` | Lists the common commands |
| `make doctor` | Checks Docker, Compose, NVIDIA runtime, and the key environment variable |
| `make ngc-login` | Authenticates Docker to `nvcr.io` without printing the key |
| `make build` | Builds the pinned Python client container |
| `make start` | Starts Nemotron ASR Streaming NIM on GPU 0 |
| `make wait` | Waits for the NIM readiness endpoint |
| `make logs` | Follows NIM logs |
| `make samples` | Copies NVIDIA's word-boosting and ordinary WAV samples from the NIM |
| `make baseline` | Transcribes once with no speech context |
| `make boosted` | Transcribes once with `config/boosting.json` |
| `make minimal` | Runs the small code example |
| `make compare` | Runs baseline and boosted transcription and writes JSON |
| `make negative` | Runs the A/B client on ordinary audio as a false-insertion check |
| `make test` | Runs unit tests inside the client container |
| `make stop` | Stops the NIM container |
| `make clean` | Removes generated WAV and JSON files, but keeps the model cache |

You can override the image, GPU, config, audio, and result path without editing
the Makefile. For example:

```bash
make start GPU_DEVICE=1
make compare CONFIG=config/product-names.json AUDIO=data/demo.wav
```

## Troubleshooting

### `make doctor` cannot find the NVIDIA runtime

Install NVIDIA Container Toolkit, configure Docker to use it, restart Docker,
and rerun `make doctor`.

### The NIM is still starting

Run `make logs`. Wait for `Riva gRPC Server is READY`, or leave `make wait`
running until the readiness endpoint returns success. First startup is slower
because model artifacts are downloaded and prepared.

### The client cannot reach `nim:50051`

Run `make start`, then `make wait`. The client container and NIM must be on the
Docker Compose network; `nim` is the service name on that network.

### Boosting inserts terms that were not spoken

Lower the score, remove overly broad phrases, and keep using negative-control
audio. A stronger boost is not automatically a better boost.

### The target is already correct in the baseline

That is a valid result. The JSON report records that the baseline already had
the term instead of claiming an improvement.

## References and scope

- [NVIDIA Nemotron ASR Streaming deployment](https://docs.nvidia.com/nim/speech/26.10.0/asr/deploy-asr-models/nemotron-asr-streaming.html)
- [NVIDIA Speech NIM word boosting](https://docs.nvidia.com/nim/speech/26.10.0/asr/customization/customization.html#word-boosting)
- [NVIDIA Speech NIM support matrix](https://docs.nvidia.com/nim/speech/26.10.0/reference/support-matrix/asr.html)
- [NVIDIA Riva Python client](https://pypi.org/project/nvidia-riva-client/)

This guide covers dynamic, per-request word boosting over the Riva gRPC API.
Static/global boosting, custom pronunciation mappings, multilingual deployment,
Helm, and production load testing are intentionally out of scope.

Code in this repository is licensed under Apache-2.0. NVIDIA containers,
models, samples, and packages remain governed by their respective licenses.
