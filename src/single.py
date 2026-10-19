"""Transcribe once in baseline or boosted mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.word_boosting import build_streaming_config, load_boosting_config, transcribe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="nim:50051")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "boosted"), required=True)
    args = parser.parse_args()

    if not args.audio.is_file():
        raise FileNotFoundError(f"audio file not found: {args.audio}")
    boosting = load_boosting_config(args.config)

    try:
        import riva.client
    except ImportError as exc:
        raise RuntimeError("run this command inside the client container") from exc

    groups = boosting.boost_groups if args.mode == "boosted" else ()
    streaming_config = build_streaming_config(
        riva.client,
        language_code=boosting.language_code,
        boost_groups=groups,
    )
    service = riva.client.ASRService(riva.client.Auth(uri=args.server))
    transcript = transcribe(
        riva.client,
        service=service,
        audio=args.audio,
        streaming_config=streaming_config,
    )
    print(f"{args.mode.upper()}: {transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
