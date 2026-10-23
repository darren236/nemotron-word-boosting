# Study guide: code word boosting with Nemotron ASR

This guide teaches the code path in this repository. By the end, you should be
able to add phrases, assign scores, send them with a streaming ASR request, and
measure whether they helped. The example uses the English Nemotron ASR
Streaming NIM and the Riva Python client. Both run through containers.

## Learning path

1. [Build a mental model](#1-build-a-mental-model)
2. [Run the existing example](#2-run-the-existing-example)
3. [Read the smallest code example](#3-read-the-smallest-code-example)
4. [Trace one phrase through the full client](#4-trace-one-phrase-through-the-full-client)
5. [Code your own boost list](#5-code-your-own-boost-list)
6. [Evaluate and tune it](#6-evaluate-and-tune-it)
7. [Practice tasks and answers](#7-practice-tasks-and-answers)

Keep the [README](README.md) open for the full directory and command reference.

## 1. Build a mental model

An ASR model receives audio and chooses a sequence of words. A rare name may
sound like common words, so the decoder may choose the common spelling. A
**boost phrase** nudges the decoder toward a specific word or phrase when the
audio supports it. It does not add a word to the model's training data or
guarantee that the word will appear.

In this repo, the boost list is **dynamic**: the client sends it with each
streaming request. You can change the list for the next request without
rebuilding the NIM container. NVIDIA documents scores from `-200` to `200`
for Nemotron ASR Streaming. Positive values encourage a phrase; negative
values discourage it. Start around `10` or `20` and measure the result.

```text
config/boosting.json
       ↓ parse and validate
BoostGroup(score=20, phrases=("AntiBERTa", "ABlooper"))
       ↓ add_word_boosting_to_config(...)
RecognitionConfig with speech context
       ↓ wrap in StreamingRecognitionConfig
audio chunks + config → Riva gRPC → Nemotron NIM → final transcript
```

The central distinction is between the **baseline** request, which has no
speech context, and the **boosted** request, which has one or more contexts.
Both use the same WAV file.

## 2. Run the existing example

You need a Linux GPU host with Docker Compose, NVIDIA Container Toolkit, and
an NGC API key. From the repository directory:

```bash
export NGC_API_KEY='your-ngc-api-key'
make doctor
make ngc-login
make build
make start
make wait
make samples
make baseline
make boosted
make compare
make negative
make stop
```

The first NIM startup may take time to download and prepare the model.
`make samples` copies NVIDIA's WAV examples to `data/`. `make compare` saves
`results/comparison.json`; `make negative` checks an ordinary speech sample
for false insertions. Generated audio and reports are ignored by Git.

If you only want to study the code, begin with the next section. Running the
model is not required to follow the request construction.

## 3. Read the smallest code example

Open [`examples/minimal.py`](examples/minimal.py). These are the edit points:

```python
SERVER = "nim:50051"                     # Compose service and gRPC port
AUDIO = Path("data/en-US_wordboosting_sample.wav")
PHRASES = ("AntiBERTa", "ABlooper")      # exact target spellings
BOOST_SCORE = 20.0                        # one score for this group
```

The core call is:

```python
config = build_streaming_config(
    riva.client,
    language_code="en-US",
    boost_groups=(BoostGroup(score=BOOST_SCORE, phrases=PHRASES),),
)
```

`BoostGroup` is our small data structure, not an NVIDIA API. The actual Riva
call happens inside [`src/word_boosting.py`](src/word_boosting.py):

```python
recognition_config = riva_client.RecognitionConfig(language_code="en-US")
riva_client.add_word_boosting_to_config(
    recognition_config,
    ["AntiBERTa", "ABlooper"],
    20.0,
)
streaming_config = riva_client.StreamingRecognitionConfig(
    config=recognition_config,
    interim_results=False,
)
```

This shortened excerpt leaves out the other recognition options. The key
detail is where the phrase list goes: into the inner `RecognitionConfig`
before that config is wrapped for streaming.

Run the minimal example after the service and samples are ready:

```bash
make minimal
```

## 4. Trace one phrase through the full client

Follow `AntiBERTa` from start to finish:

| Step | File | What happens |
| --- | --- | --- |
| 1 | [`config/boosting.json`](config/boosting.json) | You set `score: 20` and include `AntiBERTa` in `phrases`. |
| 2 | [`src/word_boosting.py`](src/word_boosting.py) | `load_boosting_config()` validates the JSON and creates `BoostGroup`. |
| 3 | [`src/compare.py`](src/compare.py) | The baseline config is built with no boost groups. |
| 4 | [`src/compare.py`](src/compare.py) | The boosted config receives the groups. |
| 5 | [`src/word_boosting.py`](src/word_boosting.py) | Each group becomes one `add_word_boosting_to_config()` call. |
| 6 | [`src/word_boosting.py`](src/word_boosting.py) | `transcribe()` streams WAV chunks and collects only final transcript segments. |
| 7 | [`src/reporting.py`](src/reporting.py) | The two transcripts are compared and saved as JSON. |

Why two configs? If you added boost groups to the baseline too, you would lose
the unboosted reference. In `src/compare.py`, look for one
`build_streaming_config()` call without `boost_groups` and another with
`boost_groups=boosting.boost_groups`.

Why `is_final`? Streaming ASR can send intermediate hypotheses that change as
more audio arrives. The client records the final segments to avoid reporting
provisional words as the transcript.

Why group phrases? Riva applies one score to all phrases in a speech context.
Two scores require two groups and therefore two calls to
`add_word_boosting_to_config()`.

## 5. Code your own boost list

For ordinary use, edit [`config/boosting.json`](config/boosting.json):

```json
{
  "language_code": "en-US",
  "boost_groups": [
    {"score": 20, "phrases": ["AntiBERTa", "ABlooper"]},
    {"score": 8, "phrases": ["graph neural networks"]}
  ]
}
```

One group is enough if all phrases share a score. Add a group when a phrase
needs a different score. The loader rejects empty phrases, duplicate exact
phrases, and scores outside the supported range. It allows case variants;
NVIDIA recommends trying relevant casing forms when recognition is sensitive
to case.

To use your own audio, place a mono PCM WAV in `data/` and run:

```bash
make compare AUDIO=data/my-speech.wav RESULT=results/my-speech.json
```

To keep a separate phrase set:

```bash
cp config/boosting.json config/my-terms.json
# Edit config/my-terms.json.
make compare AUDIO=data/my-speech.wav CONFIG=config/my-terms.json
```

If you are writing a new Python application instead of using this repo's CLI,
the minimum sequence is: create `Auth`, create `ASRService`, create a
`RecognitionConfig`, add word boosting to it, wrap it in a
`StreamingRecognitionConfig`, then stream the audio. The working reference is
[`examples/minimal.py`](examples/minimal.py) plus the helper implementation in
[`src/word_boosting.py`](src/word_boosting.py).

## 6. Evaluate and tune it

Open `results/comparison.json` after `make compare`. The most useful fields are:

```json
{
  "baseline": {"transcript": "...", "matched_phrases": []},
  "boosted": {"transcript": "...", "matched_phrases": ["AntiBERTa"]},
  "new_matches_after_boosting": ["AntiBERTa"]
}
```

The file contains more fields; this excerpt shows the ones to inspect first.
The match check is case insensitive for reporting, but the request sends your
phrase strings unchanged.

Use a small evaluation set with known spoken terms:

1. At least a few recordings containing each target phrase.
2. Similar sounding recordings that do **not** contain it.
3. Ordinary speech unrelated to your phrase list.

For each score, record correct recoveries and false insertions. If the baseline
already recognizes a phrase, boosting did not create that success. If a
negative-control transcript gains a boosted term, lower the score or narrow the
list. A score is an experimental setting, not a universal best value.

You can test the configuration and reporting logic without a GPU:

```bash
make build
make test
```

The client container runs the tests. Full transcription requires the NIM
container and a supported NVIDIA GPU.

## 7. Practice tasks and answers

Try each task before reading its answer.

**Task A:** Add `NVIDIA` at score `10`, while keeping the two example terms at
score `20`. Where do you edit?

<details>
<summary>Answer</summary>

Add a second object to `boost_groups` in `config/boosting.json`:
`{"score": 10, "phrases": ["NVIDIA"]}`. The client will create a second speech
context.

</details>

**Task B:** Which line keeps the baseline free of boosting?

<details>
<summary>Answer</summary>

In `src/compare.py`, the baseline call to `build_streaming_config()` omits
`boost_groups`. The function defaults to an empty tuple, so its loop makes no
`add_word_boosting_to_config()` call.

</details>

**Task C:** Why does `transcribe()` inspect `result.is_final`?

<details>
<summary>Answer</summary>

Streaming results may include interim text that changes. The report should
contain only final transcript segments.

</details>

**Task D:** Your boosted transcript recognizes a term, but the baseline also
does. Did boosting help on this recording?

<details>
<summary>Answer</summary>

There is no observed gain for that term on this recording. Look for it under
`baseline_already_contains_all_targets` and check more recordings before
drawing a conclusion.

</details>

## References

- [NVIDIA Speech NIM: dynamic word boosting](https://docs.nvidia.com/nim/speech/26.10.0/asr/customization/customization.html#word-boosting)
- [NVIDIA Nemotron ASR Streaming deployment](https://docs.nvidia.com/nim/speech/26.10.0/asr/deploy-asr-models/nemotron-asr-streaming.html)

This guide describes the NIM request path used by this repository. NeMo
GPU phrase boosting has different configuration and score semantics.
