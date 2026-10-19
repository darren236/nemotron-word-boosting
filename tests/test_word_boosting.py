from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.word_boosting import (
    BoostGroup,
    build_streaming_config,
    load_boosting_config,
    validate_score,
)


class FakeRecognitionConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeStreamingRecognitionConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeRivaClient:
    RecognitionConfig = FakeRecognitionConfig
    StreamingRecognitionConfig = FakeStreamingRecognitionConfig

    def __init__(self):
        self.boost_calls = []

    def add_word_boosting_to_config(self, config, phrases, score):
        self.boost_calls.append((config, phrases, score))


class ConfigTests(unittest.TestCase):
    def write_config(self, value: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "boosting.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_loads_multiple_score_groups(self) -> None:
        path = self.write_config(
            {
                "language_code": "en-US",
                "boost_groups": [
                    {"score": 20, "phrases": ["AntiBERTa", "ABlooper"]},
                    {"score": -5, "phrases": ["discourage me"]},
                ],
            }
        )
        config = load_boosting_config(path)
        self.assertEqual(config.language_code, "en-US")
        self.assertEqual(config.phrases, ["AntiBERTa", "ABlooper", "discourage me"])
        self.assertEqual(config.boost_groups[1].score, -5.0)

    def test_rejects_empty_or_duplicate_phrases(self) -> None:
        empty = self.write_config(
            {"language_code": "en-US", "boost_groups": [{"score": 20, "phrases": []}]}
        )
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            load_boosting_config(empty)

        duplicate = self.write_config(
            {
                "language_code": "en-US",
                "boost_groups": [
                    {"score": 20, "phrases": ["NVIDIA"]},
                    {"score": 10, "phrases": ["NVIDIA"]},
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate boosted phrase"):
            load_boosting_config(duplicate)

    def test_validates_documented_score_range(self) -> None:
        self.assertEqual(validate_score(-200), -200.0)
        self.assertEqual(validate_score(200), 200.0)
        for value in (-200.1, 200.1, math.inf, math.nan, True, "20"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_score(value)


class RivaConfigTests(unittest.TestCase):
    def test_baseline_adds_no_word_boosting_context(self) -> None:
        client = FakeRivaClient()
        streaming = build_streaming_config(client, language_code="en-US")
        self.assertEqual(client.boost_calls, [])
        self.assertEqual(streaming.kwargs["config"].kwargs["language_code"], "en-US")

    def test_each_group_becomes_one_context(self) -> None:
        client = FakeRivaClient()
        groups = (
            BoostGroup(score=20.0, phrases=("AntiBERTa", "ABlooper")),
            BoostGroup(score=-5.0, phrases=("discourage me",)),
        )
        streaming = build_streaming_config(
            client,
            language_code="en-US",
            boost_groups=groups,
        )
        recognition = streaming.kwargs["config"]
        self.assertEqual(
            client.boost_calls,
            [
                (recognition, ["AntiBERTa", "ABlooper"], 20.0),
                (recognition, ["discourage me"], -5.0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
