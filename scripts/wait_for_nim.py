"""Wait until the NIM HTTP readiness endpoint returns a ready response."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if response.status != 200:
                return False
            body = json.loads(response.read().decode("utf-8"))
            return body.get("status") == "ready"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://nim:9000/v1/health/ready")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    print(f"Waiting for {args.url}", flush=True)
    while time.monotonic() < deadline:
        if is_ready(args.url):
            print("Nemotron ASR NIM is ready.")
            return 0
        time.sleep(5)
    print(f"Timed out after {args.timeout} seconds.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
