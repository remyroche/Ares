#!/usr/bin/env python3
"""Freeze the target-free H4 age-expectation reference for one live release.

The H4 model is scored only from completed position state.  Its age-relative
features require prior state observations, so this tool projects the exact
prior-resolved four-month fit population into the minimal target-free columns
used by ``add_causal_age_expectations``.  No policy outcome or advantage label
is written to the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = (
    "state_decision_ts", "candidate_id", "state_bar_15m", "time_in_trade",
    "entry_atr_fraction", "current_MFE_ATR", "current_MAE_ATR",
    "giveback_from_MFE_ATR", "return_1h_atr", "RV_15m_vs_1h",
)
TARGET_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_activation50_advantage_20260830_v1/activation50_advantage_states.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default="2026-08-29T00:00:00Z")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("H4 reference output must be immutable")
    cutoff = _utc(args.cutoff)
    start = cutoff - pd.DateOffset(months=int(args.train_months))
    # Do not import the broad feature-study runner here: its research imports
    # initialise unrelated HPO dependencies.  This reference needs only the
    # explicit target-free state columns below from the same immutable parent
    # panel used to fit H4.
    panel = pd.read_parquet(
        TARGET_PANEL,
        columns=[
            *FEATURES, "entry_decision_ts", "policy_label_available_ts",
            "MC1_expected_bps",
        ],
    )
    panel["state_decision_ts"] = pd.to_datetime(panel["state_decision_ts"], utc=True, errors="raise")
    panel["entry_decision_ts"] = pd.to_datetime(panel["entry_decision_ts"], utc=True, errors="raise")
    panel["policy_label_available_ts"] = pd.to_datetime(panel["policy_label_available_ts"], utc=True, errors="raise")
    selected = panel.loc[
        pd.to_numeric(panel["MC1_expected_bps"], errors="coerce").ge(30.0)
        & panel["entry_decision_ts"].ge(start)
        & panel["entry_decision_ts"].lt(cutoff)
        & panel["state_decision_ts"].lt(cutoff)
        & panel["policy_label_available_ts"].lt(cutoff),
        list(FEATURES),
    ].copy()
    if selected.empty or selected.duplicated(["state_decision_ts", "candidate_id", "state_bar_15m"]).any():
        raise RuntimeError("H4 reference has no unique causally eligible state rows")
    selected = selected.sort_values(["state_decision_ts", "candidate_id", "state_bar_15m"], kind="stable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=False)
    reference = output / "h4_expectation_reference_target_free.parquet"
    selected.to_parquet(reference, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_p8u_e2_h4_expectation_reference_v1",
        "status": "target_free_frozen_prior_state_reference",
        "order_submission": False,
        "cutoff": cutoff.isoformat(),
        "training_window": {"start": start.isoformat(), "end_exclusive": cutoff.isoformat(), "months": int(args.train_months)},
        "source": {
            "target_panel": str(TARGET_PANEL.relative_to(ROOT)),
            "target_panel_sha256": _sha256(TARGET_PANEL),
        },
        "selection": "MC1_expected_bps>=30; entry/state/label availability strictly before cutoff",
        "rows": int(len(selected)),
        "columns": list(FEATURES),
        "forbidden_output_columns": ["policy_net_bps", "policy_gross_bps", "activation50_advantage_bps"],
        "reference_file": {"name": reference.name, "sha256": _sha256(reference)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
