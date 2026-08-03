#!/usr/bin/env python3
"""Extend the 29-field current health panel with resolved frozen-forward OOS.

The extension may train only a later detector.  Original OOF and frozen
forward provenance remain explicit and mutually exclusive; no forward row may
be used to evaluate a detector fitted on this combined history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_current_model_health import (  # noqa: E402
    CURRENT_MODEL_HEALTH_COLUMNS,
    build_hourly_current_model_health,
)
from materialize_historical_exact_model_health import (  # noqa: E402
    _safe,
    _sha256,
    _write_json,
)


DEFAULT_HISTORY = ROOT / (
    "data_perp/artifacts/failure_first_current_strict_model_oos_history_"
    "20260726_v1/strict_model_oos_history.parquet"
)
DEFAULT_CURRENT_HANDOFF = ROOT / (
    "data_perp/artifacts/execution_ev_repaired_heads_representation_handoff_"
    "20260726_v7/joined.parquet"
)
DEFAULT_FORWARD_HANDOFF = ROOT / (
    "data_perp/artifacts/execution_ev_context_head_clean_forward_july19_"
    "20260726_v2/strict_forward_winner_inputs_and_raw_scores.parquet"
)

RICH_COLUMNS = (
    "candidate_id",
    "__ts__",
    "execution_decision_utc",
    "base_oof_score",
    "base_margin_to_cutoff_z",
    "catboost_entropy",
    "alpha_prediction_uncertainty",
)


def assemble_extended_rich_handoff(
    history: pd.DataFrame,
    current_handoff: pd.DataFrame,
    forward_handoff: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "candidate_id",
        "failure_first_history_role",
        "failure_first_score_is_strict_model_oos",
        "causal_recent_side_isotonic_ev__is_oof",
        "causal_recent_side_isotonic_ev__is_forward_oos",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"strict history lacks {missing}")
    if history["candidate_id"].duplicated().any():
        raise ValueError("strict history candidate IDs must be unique")
    if not history["failure_first_score_is_strict_model_oos"].fillna(
        False
    ).astype(bool).all():
        raise ValueError("history contains a non-strict model-OOS row")
    oof = history["causal_recent_side_isotonic_ev__is_oof"].fillna(
        False
    ).astype(bool)
    forward = history[
        "causal_recent_side_isotonic_ev__is_forward_oos"
    ].fillna(False).astype(bool)
    if not (oof ^ forward).all():
        raise ValueError("OOF and forward provenance must be exclusive")
    expected_role = pd.Series(
        "outer_oof", index=history.index, dtype=object
    )
    expected_role.loc[forward] = "retired_resolved_forward_oos"
    if not history["failure_first_history_role"].astype(str).eq(
        expected_role
    ).all():
        raise ValueError("history role disagrees with score provenance")

    for name, frame in (
        ("current handoff", current_handoff),
        ("forward handoff", forward_handoff),
    ):
        missing = sorted(set(RICH_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} lacks {missing}")
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} candidate IDs must be unique")
    current_ids = set(history.loc[oof, "candidate_id"].astype(str))
    forward_ids = set(history.loc[forward, "candidate_id"].astype(str))
    current = current_handoff.loc[
        current_handoff["candidate_id"].astype(str).isin(current_ids),
        list(RICH_COLUMNS),
    ].copy()
    retired = forward_handoff.loc[
        forward_handoff["candidate_id"].astype(str).isin(forward_ids),
        list(RICH_COLUMNS),
    ].copy()
    if set(current["candidate_id"].astype(str)) != current_ids:
        raise ValueError("current OOF history lacks exact rich handoff identity")
    if set(retired["candidate_id"].astype(str)) != forward_ids:
        raise ValueError("retired forward history lacks exact rich handoff identity")
    rich = pd.concat([current, retired], ignore_index=True)
    if rich["candidate_id"].duplicated().any() or len(rich) != len(history):
        raise ValueError("extended rich handoff is not one-to-one with history")
    return rich


def run(args: argparse.Namespace) -> dict[str, Any]:
    history_path = Path(args.history)
    current_path = Path(args.current_handoff)
    forward_path = Path(args.forward_handoff)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    history = pd.read_parquet(history_path)
    rich = assemble_extended_rich_handoff(
        history,
        pd.read_parquet(current_path),
        pd.read_parquet(forward_path),
    )
    health, report = build_hourly_current_model_health(history, rich)
    output.mkdir(parents=True, exist_ok=False)
    health_path = output / "hourly_model_health.parquet"
    catalog_path = output / "field_catalog.csv"
    health.to_parquet(health_path, index=False, compression="zstd")
    pd.DataFrame({"feature": CURRENT_MODEL_HEALTH_COLUMNS}).to_csv(
        catalog_path, index=False
    )
    role_counts = (
        history["failure_first_history_role"].value_counts().to_dict()
    )
    report.update(
        {
            "schema": "current_lineage_extended_model_health_v1",
            "status": "STRICT_MODEL_OOS_HEALTH_EXTENSION_COMPLETE",
            "role_counts": role_counts,
            "evaluation_policy": (
                "resolved forward rows may train only a later detector and "
                "must be excluded from evaluation of any detector fitted on them"
            ),
            "sources": {
                "history": {
                    "path": str(history_path),
                    "sha256": _sha256(history_path),
                },
                "current_handoff": {
                    "path": str(current_path),
                    "sha256": _sha256(current_path),
                },
                "forward_handoff": {
                    "path": str(forward_path),
                    "sha256": _sha256(forward_path),
                },
            },
            "outputs": {
                "health": {
                    "path": str(health_path),
                    "sha256": _sha256(health_path),
                },
                "catalog": {
                    "path": str(catalog_path),
                    "sha256": _sha256(catalog_path),
                },
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        }
    )
    _write_json(output / "manifest.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--current-handoff", type=Path, default=DEFAULT_CURRENT_HANDOFF
    )
    parser.add_argument(
        "--forward-handoff", type=Path, default=DEFAULT_FORWARD_HANDOFF
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
