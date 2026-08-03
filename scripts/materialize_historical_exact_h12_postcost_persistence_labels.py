#!/usr/bin/env python3
"""Create conditional persistence/giveback labels from exact post-cost events.

These labels are future outcomes and are explicitly not execution features.
They are the training targets for a future strict-OOF persistence head:
conditional on clearing fixed post-cost value before the adverse barrier, does
the candidate retain positive exact-H12 net value or give it back?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_historical_exact_h12_postcost_events import TARGET_ID
from scripts.materialize_historical_exact_h12_alignment_sidecar import COST_MODEL_ID, EXECUTION_POLICY_ID


EVENTS = ROOT / "data_perp/artifacts/historical_exact_h12_postcost_events_20260731_v1/postcost_events.parquet"
ALIGNMENT = ROOT / "data_perp/artifacts/historical_exact_h12_alignment_sidecar_research_only_20260731_v1/alignment_sidecar.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/historical_exact_h12_postcost_persistence_labels_20260731_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_labels(events: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    required_events = {"candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "postcost_target_id", "execution_policy_id", "cost_model_id", "postcost_h0_event", "postcost_h25_event"}
    required_alignment = {"candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "exact_h12_net_bps", "execution_policy_id", "cost_model_id"}
    if required_events.difference(events) or required_alignment.difference(alignment):
        raise ValueError("post-cost events or alignment schema incomplete")
    if events.candidate_id.duplicated().any() or alignment.candidate_id.duplicated().any():
        raise ValueError("candidate identities must be unique")
    if events.postcost_target_id.nunique() != 1 or events.postcost_target_id.iloc[0] != TARGET_ID:
        raise ValueError("incorrect exact post-cost target contract")
    output = events.merge(
        alignment.loc[:, ["candidate_id", "exact_h12_net_bps"]],
        on="candidate_id", how="inner", validate="one_to_one",
    )
    if len(output) != len(events):
        raise ValueError("alignment join loses post-cost event rows")
    for hurdle, net_floor in (("h0", 0.0), ("h25", 25.0)):
        event = output[f"postcost_{hurdle}_event"]
        clear = event.eq("clear_cost_first")
        retained = clear & output.exact_h12_net_bps.gt(net_floor)
        giveback = clear & ~retained
        state = pd.Series("not_reached", index=output.index, dtype="string")
        state.loc[event.eq("adverse_first_or_conflict")] = "adverse_first_or_conflict"
        state.loc[event.eq("timeout")] = "timeout"
        state.loc[retained] = "clear_then_retained"
        state.loc[giveback] = "clear_then_giveback"
        output[f"postcost_{hurdle}_clear_first"] = clear.astype("int8")
        output[f"postcost_{hurdle}_persistence_target_valid"] = clear.astype("int8")
        output[f"postcost_{hurdle}_retained_net"] = retained.astype("int8")
        output[f"postcost_{hurdle}_giveback_after_clear"] = giveback.astype("int8")
        output[f"postcost_{hurdle}_four_state"] = state
    return output.loc[:, [
        "candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts",
        "execution_policy_id", "cost_model_id", "exact_h12_net_bps",
        *[f"postcost_{hurdle}_{suffix}" for hurdle in ("h0", "h25") for suffix in ("clear_first", "persistence_target_valid", "retained_net", "giveback_after_clear", "four_state")],
    ]]


def run(*, events_path: Path, alignment_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    events = pd.read_parquet(events_path)
    alignment = pd.read_parquet(alignment_path, columns=["candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "exact_h12_net_bps", "execution_policy_id", "cost_model_id"])
    labels = build_labels(events, alignment)
    for column in ("decision_ts", "label_end_ts", "label_available_ts"):
        labels[column] = pd.to_datetime(labels[column], utc=True, errors="raise")
    if not labels.label_end_ts.eq(labels.decision_ts + pd.Timedelta(hours=12)).all() or not labels.label_available_ts.eq(labels.label_end_ts).all():
        raise ValueError("persistence labels lose exact H12 availability")
    if labels.execution_policy_id.nunique() != 1 or labels.execution_policy_id.iloc[0] != EXECUTION_POLICY_ID or labels.cost_model_id.nunique() != 1 or labels.cost_model_id.iloc[0] != COST_MODEL_ID:
        raise ValueError("persistence labels violate frozen policy/cost contract")
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        label_path = stage / "postcost_persistence_labels.parquet"
        labels.to_parquet(label_path, index=False, compression="zstd")
        support = []
        for hurdle in ("h0", "h25"):
            support.append(labels.groupby(["side", f"postcost_{hurdle}_four_state"], as_index=False).size().assign(hurdle=hurdle).rename(columns={f"postcost_{hurdle}_four_state": "state", "size": "rows"}))
        pd.concat(support, ignore_index=True).to_csv(stage / "support_by_side.csv", index=False)
        _write_json(stage / "contract.json", {
            "source_target_id": TARGET_ID,
            "policy_id": EXECUTION_POLICY_ID,
            "cost_model_id": COST_MODEL_ID,
            "roles": {
                "reachability": "postcost_h*_clear_first",
                "persistence": "postcost_h*_retained_net conditional on clear_first",
                "giveback": "postcost_h*_giveback_after_clear conditional on clear_first",
            },
            "availability": "exact H12 outcome; training target only, never a decision-time input",
        })
        manifest = {
            "schema": "historical_exact_h12_postcost_persistence_labels_v1",
            "status": "MATERIALIZED_RESEARCH_ONLY_OOF_HEAD_TARGETS_ONLY",
            "rows": len(labels),
            "inputs": {str(path): _sha256(path) for path in (events_path, alignment_path)},
            "outputs": {name: _sha256(stage / name) for name in ("postcost_persistence_labels.parquet", "support_by_side.csv", "contract.json")},
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(events_path=args.events, alignment_path=args.alignment, output=args.output), indent=2))


if __name__ == "__main__":
    main()
