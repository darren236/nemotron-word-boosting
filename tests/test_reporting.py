from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.reporting import build_report, phrase_is_present, write_report


class ReportingTests(unittest.TestCase):
    def test_phrase_matching_is_case_insensitive_and_bounded(self) -> None:
        self.assertTrue(phrase_is_present("AntiBERTa is a model", "antiberta"))
        self.assertTrue(phrase_is_present("Use Riva   Speech", "Riva Speech"))
        self.assertFalse(phrase_is_present("notAntiBERTax", "AntiBERTa"))

    def test_comparison_reports_new_matches_and_writes_json(self) -> None:
        report = build_report(
            audio=Path("sample.wav"),
            phrases=["AntiBERTa", "ABlooper"],
            groups=[{"score": 20.0, "phrases": ["AntiBERTa", "ABlooper"]}],
            baseline_transcript="aunt bertha and ab loper",
            boosted_transcript="AntiBERTa and ABlooper",
            negative_control=False,
        )
        self.assertEqual(report["new_matches_after_boosting"], ["AntiBERTa", "ABlooper"])
        self.assertTrue(report["boosted_contains_all_targets"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_report(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["boosted"]["matched_phrases"], ["AntiBERTa", "ABlooper"])

    def test_negative_control_flags_insertions(self) -> None:
        report = build_report(
            audio=Path("ordinary.wav"),
            phrases=["AntiBERTa"],
            groups=[{"score": 20.0, "phrases": ["AntiBERTa"]}],
            baseline_transcript="ordinary sentence",
            boosted_transcript="ordinary AntiBERTa sentence",
            negative_control=True,
        )
        self.assertEqual(report["unexpected_boosted_phrases"], ["AntiBERTa"])


if __name__ == "__main__":
    unittest.main()
