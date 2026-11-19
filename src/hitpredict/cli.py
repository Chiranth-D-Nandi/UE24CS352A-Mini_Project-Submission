"""Command-line entry point for Part 1: Data Engineering, Acquisition, Linkage & Audit Pipeline.

Owned by Member 1: Chiranth D Nandi (PES2UG24AM048)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hitpredict.config import ensure_output_directories, load_config
from hitpredict.data.acquire import acquire_sources, spotify_missing_message
from hitpredict.data.audit import run_data_audit
from hitpredict.data.build_dataset import prepare_dataset


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hitpredict",
        description="HitPredict Part 1: Data Engineering, Acquisition, Linkage & Audit (Member 1: Chiranth D Nandi)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["acquire", "prepare", "audit", "train", "all"]:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="configs/default.yaml")
        if name in {"train", "all"}:
            subparser.add_argument("--profile", choices=["quick", "full"], default="quick")
        if name == "acquire":
            subparser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_config(arguments.config)
    ensure_output_directories(config)
    try:
        if arguments.command == "acquire":
            result = acquire_sources(config, force=arguments.force)
            _print(result)
            if not Path(config["data"]["spotify_csv"]).exists():
                print("\n" + spotify_missing_message(config), file=sys.stderr)
            return 0
        if arguments.command == "prepare":
            _print(prepare_dataset(config))
            return 0
        if arguments.command == "audit":
            _print(run_data_audit(config))
            return 0
        if arguments.command in {"train", "all"}:
            if arguments.command == "all":
                acquire_sources(config, force=False)
                _print({"prepare": prepare_dataset(config)})
                _print({"audit": run_data_audit(config)})
            print(
                "\n[INFO] Model training, calibration, evaluation, and Streamlit demo application\n"
                "belong to Part 2 of this project (Owned by Member 2: Harshitha - PES2UG24AM064).\n"
                "Please run `hitpredict acquire`, `hitpredict prepare`, or `hitpredict audit` for Part 1."
            )
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
