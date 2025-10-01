"""Command-line entry point for running topic classification experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from topic_modeling import ExperimentConfig, run_experiments

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run topic classification experiments.")
    parser.add_argument(
        "--training-sizes",
        nargs="*",
        type=str,
        default=["10","20","100","200","500", "1000", "2000", "full"],
        help=(
            "Training set sizes to evaluate. Use 'full' to indicate the complete dataset. "
            "Values greater than the dataset size will be clipped."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory to store experiment outputs. Defaults to the 'reports' folder alongside "
            "run_experiments.py."
        ),

    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _parse_training_sizes(raw_sizes: List[str]) -> List[int | None]:
    parsed: List[int | None] = []
    for entry in raw_sizes:
        value = entry.strip().lower()
        if value in {"full", "max", "all"}:
            parsed.append(None)
        else:
            try:
                parsed.append(int(value))
            except ValueError as exc:  # pragma: no cover - CLI validation
                raise ValueError(f"Invalid training size '{entry}'. Use integers or 'full'.") from exc
    if not parsed:
        parsed = [None]
    return parsed


def main() -> None:
    args = parse_args()
    sizes = _parse_training_sizes(args.training_sizes)
    config = ExperimentConfig(
        training_sizes=sizes,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )
    run_experiments(config)


if __name__ == "__main__":
    main()
