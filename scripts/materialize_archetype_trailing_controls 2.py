"""Recover stored archetype-specific trailing controls into a replay contract.

The historical policy summary intentionally stored a compact final geometry.
Its complete fit summary retains the train-selected trailing trial, including
``trailing_power`` and ``giveback_beta``.  This utility carries only those two
pre-fit fields into the compact geometry; it does not fit or select on replay.
"""

from __future__ import annotations

import argparse
import ast
import re

import pandas as pd


def parse_mapping(value: object) -> dict[str, object]:
    # Artifact strings are Python mappings with occasional bare NaN values.
    # Normalize those numerical sentinels before literal parsing, never eval.
    cleaned = re.sub(
        r"(?<![A-Za-z0-9_\"'])(?:nan|inf|-inf)(?![A-Za-z0-9_\"'])",
        "None",
        str(value),
    )
    parsed = ast.literal_eval(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected mapping")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    diagnostics: list[str] = []
    for idx, row in frame.iterrows():
        geometry = parse_mapping(row["shrinkage_final_geometry"])
        fit = parse_mapping(row["full_fit_summary"])
        trial = (
            fit.get("trailing_stage", {})
            .get("selection", {})
            .get("selected_trial", {})
            .get("params", {})
        )
        if not isinstance(trial, dict):
            raise ValueError(f"Missing selected trailing trial for row {idx}")
        for field in ("trailing_power", "giveback_beta"):
            value = trial.get(field)
            if value is None:
                raise ValueError(f"Missing {field} for row {idx}")
            geometry[field] = float(value)
        frame.at[idx, "shrinkage_final_geometry"] = repr(geometry)
        diagnostics.append(
            f"{row['strategy_id']}|{row['policy_archetype']}: "
            f"power={geometry['trailing_power']}, giveback={geometry['giveback_beta']}"
        )
    frame.to_csv(args.output, index=False)
    print("\n".join(diagnostics))


if __name__ == "__main__":
    main()
