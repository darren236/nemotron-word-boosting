"""Build readable and machine-readable baseline/boosted comparisons."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def _normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def phrase_is_present(transcript: str, phrase: str) -> bool:
    """Match a phrase case-insensitively on word-like boundaries."""

    pattern = rf"(?<!\w){re.escape(_normalise(phrase))}(?!\w)"
    return re.search(pattern, _normalise(transcript)) is not None


def analyse(transcript: str, phrases: list[str]) -> dict[str, Any]:
    matches = {phrase: phrase_is_present(transcript, phrase) for phrase in phrases}
    return {
        "transcript": transcript,
        "matches": matches,
        "matched_phrases": [phrase for phrase, found in matches.items() if found],
        "missing_phrases": [phrase for phrase, found in matches.items() if not found],
    }


def build_report(
    *,
    audio: Path,
    phrases: list[str],
    groups: list[dict[str, Any]],
    baseline_transcript: str,
    boosted_transcript: str,
    negative_control: bool,
) -> dict[str, Any]:
    baseline = analyse(baseline_transcript, phrases)
    boosted = analyse(boosted_transcript, phrases)
    new_matches = [
        phrase
        for phrase in phrases
        if not baseline["matches"][phrase] and boosted["matches"][phrase]
    ]
    report: dict[str, Any] = {
        "engine": "nemotron-asr-streaming-nim",
        "audio": str(audio),
        "boost_groups": groups,
        "baseline": baseline,
        "boosted": boosted,
        "new_matches_after_boosting": new_matches,
        "baseline_already_contains_all_targets": not baseline["missing_phrases"],
        "boosted_contains_all_targets": not boosted["missing_phrases"],
        "negative_control": negative_control,
    }
    if negative_control:
        report["unexpected_boosted_phrases"] = boosted["matched_phrases"]
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_report(report: dict[str, Any], output: Path) -> None:
    baseline = report["baseline"]
    boosted = report["boosted"]
    print(f"BASELINE: {baseline['transcript']}")
    print(f"BOOSTED:  {boosted['transcript']}")
    new_matches = report["new_matches_after_boosting"]
    print("NEW MATCHES:", ", ".join(new_matches) if new_matches else "none")
    if report["baseline_already_contains_all_targets"]:
        print("NOTE: the baseline already contained every target phrase")
    if report.get("unexpected_boosted_phrases"):
        print(
            "WARNING: boosted phrases appeared in negative-control audio:",
            ", ".join(report["unexpected_boosted_phrases"]),
        )
    print(f"JSON: {output}")
