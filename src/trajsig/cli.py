from __future__ import annotations

import argparse
from pathlib import Path

from .audit import audit_experiment
from .config import load_config
from .evaluation import evaluate_all
from .packaging import package_training_artifacts
from .report import generate_figures, generate_report
from .train import train_all


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trajectory-signature Phase-0 experiment")
    parser.add_argument("command", choices=["train", "audit", "evaluate", "report", "pipeline", "package"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true", help="Retrain even complete runs")
    parser.add_argument("--destination", help="Package destination")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    if args.command in {"train", "pipeline"}:
        train_all(config, output, force=args.force)
    if args.command in {"audit", "pipeline"}:
        result = audit_experiment(config, output)
        print(f"Audit: {result['status']}")
    if args.command in {"evaluate", "pipeline"}:
        paths = evaluate_all(config, output)
        print(f"Evaluation: {paths['summary']}")
    if args.command in {"report", "pipeline"}:
        figures = generate_figures(output)
        report = generate_report(config, output)
        print(f"Report: {report}; figures: {len(figures)}")
    if args.command == "package":
        archive = package_training_artifacts(output, args.destination)
        print(f"Archive: {archive}")


if __name__ == "__main__":
    main()

