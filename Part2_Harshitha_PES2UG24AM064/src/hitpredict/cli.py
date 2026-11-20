"""Command-line entry point for Part 2: ML Modeling, Calibration, Evaluation & Streamlit Demo.

Owned by Member 2: Harshitha (PES2UG24AM064)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hitpredict.config import ensure_output_directories, load_config
from hitpredict.training import train_and_evaluate


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hitpredict",
        description="HitPredict Part 2: ML Modeling, Calibration & App (Member 2: Harshitha)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "all", "acquire", "prepare", "audit"]:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="configs/default.yaml")
        if name in {"train", "all"}:
            subparser.add_argument("--profile", choices=["quick", "full"], default="quick")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_config(arguments.config)
    ensure_output_directories(config)
    try:
        if arguments.command in {"acquire", "prepare", "audit"}:
            print(
                f"[INFO] Command '{arguments.command}' is part of Part 1 "
                "(Owned by Member 1: Chiranth D Nandi - PES2UG24AM048).\n"
                "Part 2 utilizes preprocessed data from Part 1 to run model training and evaluation."
            )
            return 0
        if arguments.command in {"train", "all"}:
            _print(train_and_evaluate(config, profile=arguments.profile))
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
