"""The smallest useful Nemotron ASR word-boosting example."""

from pathlib import Path

import riva.client

from src.word_boosting import BoostGroup, build_streaming_config, transcribe


# EDIT HERE: these four values are the usual customization points.
SERVER = "nim:50051"
AUDIO = Path("data/en-US_wordboosting_sample.wav")
PHRASES = ("AntiBERTa", "ABlooper")
BOOST_SCORE = 20.0


def main() -> None:
    service = riva.client.ASRService(riva.client.Auth(uri=SERVER))
    config = build_streaming_config(
        riva.client,
        language_code="en-US",
        boost_groups=(BoostGroup(score=BOOST_SCORE, phrases=PHRASES),),
    )
    print(
        transcribe(
            riva.client,
            service=service,
            audio=AUDIO,
            streaming_config=config,
        )
    )


if __name__ == "__main__":
    main()
