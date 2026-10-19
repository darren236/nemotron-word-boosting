"""Run a baseline-versus-boosted Nemotron ASR comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.reporting import build_report, print_report, write_report
from src.word_boosting import (
    build_streaming_config,
    load_boosting_config,
    transcribe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Nemotron ASR output without and with word boosting."
    )
    parser.add_argument("--server", default="nim:50051")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=1600)
    parser.add_argument("--negative-control", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(f"audio file not found: {args.audio}")

    boosting = load_boosting_config(args.config)

    try:
        import riva.client
    except ImportError as exc:
        raise RuntimeError("run this command inside the client container") from exc

    service = riva.client.ASRService(riva.client.Auth(uri=args.server))
    baseline_config = build_streaming_config(
        riva.client,
        language_code=boosting.language_code,
    )
    boosted_config = build_streaming_config(
        riva.client,
        language_code=boosting.language_code,
        boost_groups=boosting.boost_groups,
    )

    baseline = transcribe(
        riva.client,
        service=service,
        audio=args.audio,
        streaming_config=baseline_config,
        chunk_frames=args.chunk_frames,
    )
    boosted = transcribe(
        riva.client,
        service=service,
        audio=args.audio,
        streaming_config=boosted_config,
        chunk_frames=args.chunk_frames,
    )

    groups = [
        {"score": group.score, "phrases": list(group.phrases)}
        for group in boosting.boost_groups
    ]
    report = build_report(
        audio=args.audio,
        phrases=boosting.phrases,
        groups=groups,
        baseline_transcript=baseline,
        boosted_transcript=boosted,
        negative_control=args.negative_control,
    )
    write_report(args.output, report)
    print_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
