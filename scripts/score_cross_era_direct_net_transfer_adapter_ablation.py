#!/usr/bin/env python3
"""Score a frozen direct-net transfer ablation without opening outcome labels.

The parser intentionally has no label argument.  This process can only verify
the frozen historical bundle and score a point-in-time feature pack.  The
separate audit script is the only component allowed to join exact outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY, sha256
from scripts.run_cross_era_direct_net_transfer_adapter_ablation import (
    SCHEMA,
    _assert_identity,
    _hash,
    _normalise_matrix,
    _write_json,
    add_corrected_transition_inputs,
    apply_corrections,
    score_parent,
)
from scripts.run_cross_era_tail_payoff_challenger import add_regime_composites


def _load_frozen(source_dir: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    frozen_path = source_dir / "frozen_before_current_evaluation.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("schema") != SCHEMA:
        raise ValueError(f"unexpected frozen schema: {frozen.get('schema')!r}")
    if frozen.get("current_outcomes_used_for_selection") is not False:
        raise ValueError("frozen selection does not prove label-free current selection")
    model_path = Path(frozen["model"]["path"])
    if sha256(model_path) != frozen["model"]["sha256"]:
        raise ValueError("frozen model hash mismatch")
    return frozen, joblib.load(model_path)


def prepare_current_frame(path: Path) -> pd.DataFrame:
    """Normalise the declared feature pack without reading any labels."""

    frame = pd.read_parquet(path)
    _assert_identity(frame, "current feature pack")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame = frame.rename(columns={
        "base_candidate_group_rows": "candidate_group_size",
        "base_margin_to_cutoff": "base_margin_to_candidate_cutoff",
    })
    if "era" not in frame:
        frame["era"] = "2026_may_jul19"
    frame, _ = add_regime_composites(frame)
    return add_corrected_transition_inputs(frame)


def score_frame(bundle: Mapping[str, Any], current: pd.DataFrame) -> pd.DataFrame:
    matrix = _normalise_matrix(current, bundle["parent_columns"])
    parent = score_parent(current, matrix, bundle["parent"])
    result = apply_corrections(parent, current, bundle["corrections"])
    _assert_identity(result, "current predictions")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frozen, bundle = _load_frozen(args.source_dir)
    current = prepare_current_frame(args.current_pack)
    scored = score_frame(bundle, current)
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "current_predictions_before_outcomes.parquet"
    scored.to_parquet(prediction_path, index=False)
    report = {
        "schema": SCHEMA,
        "status": "scored_label_free_before_current_outcomes",
        "promotion_eligible": False,
        "label_free_contract": "this scorer has no label input and performs no retraining, selection, calibration or mapping",
        "frozen_source": _hash(args.source_dir / "frozen_before_current_evaluation.json"),
        "current_pack": _hash(args.current_pack),
        "outputs": {"predictions": {**_hash(prediction_path), "rows": int(len(scored))}},
    }
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    _write_json(args.output_dir / "manifest.json", {"schema": SCHEMA, "status": report["status"], "report": _hash(report_path), "outputs": report["outputs"]})
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-dir", type=Path, required=True)
    value.add_argument("--current-pack", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True, default=str))
