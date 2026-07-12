#!/usr/bin/env python3
"""Orchestrate the complete global residual-state discovery evidence pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str]) -> None:
    command = [sys.executable, "-u", *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--skip-dossiers", action="store_true")
    parser.add_argument("--skip-model-ablation", action="store_true")
    parser.add_argument("--run-final-selection-hpo", action="store_true")
    parser.add_argument("--force-state-build", action="store_true")
    args = parser.parse_args()

    if not args.skip_events:
        _run(["scripts/run_global_residual_state_discovery.py", "--stage", "all"])
    latent = ["scripts/run_global_residual_latent_state.py", "--stage", "all"]
    if args.force_state_build:
        latent.append("--force")
    if args.mode == "smoke":
        latent.extend(
            [
                "--latent-dims",
                "6,8",
                "--aux-lambdas",
                "0,0.05",
                "--gmm-components",
                "4,6",
                "--gmm-covariance",
                "diag",
                "--gmm-reg-covars",
                "0.001",
                "--gmm-n-init",
                "1",
                "--ae-epochs",
                "30",
                "--search-finalists",
                "2",
            ]
        )
    _run(latent)
    if not args.skip_dossiers:
        _run(["scripts/report_global_residual_event_dossiers.py"])
    _run(["scripts/report_global_residual_state_validation.py"])
    if not args.skip_model_ablation:
        champion = ["scripts/run_global_residual_champion_enhancement.py"]
        if args.mode == "smoke":
            champion.extend(
                [
                    "--encoder-epochs",
                    "12",
                    "--latent-dim",
                    "6",
                    "--gmm-components",
                    "4,6",
                    "--smoke",
                    "--max-greedy-rounds",
                    "1",
                ]
            )
        if args.run_final_selection_hpo:
            champion.append("--run-final-selection-hpo")
        _run(champion)


if __name__ == "__main__":
    main()
