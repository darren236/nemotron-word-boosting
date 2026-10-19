"""Nemotron ASR Streaming NIM word-boosting helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoostGroup:
    """Phrases that share one Nemotron dynamic boost score."""

    score: float
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class BoostingConfig:
    """Validated request-time boosting configuration."""

    language_code: str
    boost_groups: tuple[BoostGroup, ...]

    @property
    def phrases(self) -> list[str]:
        return [phrase for group in self.boost_groups for phrase in group.phrases]


def validate_score(value: object) -> float:
    """Return a valid Nemotron score in the documented -200..200 range."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("boost score must be a number")
    score = float(value)
    if not math.isfinite(score) or not -200.0 <= score <= 200.0:
        raise ValueError("boost score must be finite and between -200 and 200")
    return score


def _load_group(value: object, index: int, seen: set[str]) -> BoostGroup:
    if not isinstance(value, dict):
        raise ValueError(f"boost_groups[{index}] must be an object")

    score = validate_score(value.get("score"))
    raw_phrases = value.get("phrases")
    if not isinstance(raw_phrases, list) or not raw_phrases:
        raise ValueError(f"boost_groups[{index}].phrases must be a non-empty list")

    phrases: list[str] = []
    for phrase_index, raw_phrase in enumerate(raw_phrases):
        if not isinstance(raw_phrase, str) or not raw_phrase.strip():
            raise ValueError(
                f"boost_groups[{index}].phrases[{phrase_index}] must be non-empty text"
            )
        phrase = raw_phrase.strip()
        if phrase in seen:
            raise ValueError(f"duplicate boosted phrase: {phrase!r}")
        phrases.append(phrase)
        seen.add(phrase)

    return BoostGroup(score=score, phrases=tuple(phrases))


def load_boosting_config(path: Path) -> BoostingConfig:
    """Load and validate the user-editable JSON configuration."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ValueError("boosting config must contain a JSON object")

    language_code = raw.get("language_code")
    if not isinstance(language_code, str) or not language_code.strip():
        raise ValueError("language_code must be non-empty text")

    raw_groups = raw.get("boost_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("boost_groups must be a non-empty list")

    seen: set[str] = set()
    groups = tuple(
        _load_group(raw_group, index, seen)
        for index, raw_group in enumerate(raw_groups)
    )
    return BoostingConfig(
        language_code=language_code.strip(),
        boost_groups=groups,
    )


def build_streaming_config(
    riva_client: Any,
    *,
    language_code: str,
    boost_groups: tuple[BoostGroup, ...] = (),
) -> Any:
    """Build a streaming request; an empty group list is a true baseline."""

    recognition_config = riva_client.RecognitionConfig(
        language_code=language_code,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        profanity_filter=False,
        verbatim_transcripts=True,
    )

    for group in boost_groups:
        riva_client.add_word_boosting_to_config(
            recognition_config,
            list(group.phrases),
            group.score,
        )

    return riva_client.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=False,
    )


def transcribe(
    riva_client: Any,
    *,
    service: Any,
    audio: Path,
    streaming_config: Any,
    chunk_frames: int = 1600,
) -> str:
    """Stream a WAV file to NIM and concatenate its final segments."""

    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be greater than zero")

    final_segments: list[str] = []
    with riva_client.AudioChunkFileIterator(str(audio), chunk_frames) as chunks:
        responses = service.streaming_response_generator(
            audio_chunks=chunks,
            streaming_config=streaming_config,
        )
        for response in responses:
            for result in response.results:
                if result.is_final and result.alternatives:
                    text = result.alternatives[0].transcript.strip()
                    if text:
                        final_segments.append(text)

    if not final_segments:
        raise RuntimeError("NIM returned no final transcript")
    return " ".join(final_segments)
