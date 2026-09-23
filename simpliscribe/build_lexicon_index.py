from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .lexicon_index import build_index, default_index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the required SimpliScribe medicine lexicon index.")
    parser.add_argument("--output", type=Path, default=default_index_path())
    args = parser.parse_args()
    started = time.perf_counter()
    result = build_index(args.output)
    result["generation_ms"] = round((time.perf_counter() - started) * 1000)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
