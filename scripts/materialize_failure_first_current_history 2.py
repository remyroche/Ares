#!/usr/bin/env python3
"""Retire resolved current forward OOS into explicit detector-training history.

This materializer never rewrites provenance.  Original OOF and frozen-forward
flags remain present, while a new combined flag identifies rows that are now
eligible as resolved strict model-OOS history for a *future* failure detector.
The already-opened forward evaluation report is mandatory evidence before any
forward row can be retired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LEDGER = Path(
    "data_perp/artifacts/"
    "execution_ev_context_clean_recent_mapping_forward_july19_20260726_v1/"
    "mapped_oof.parquet"
)
DEFAULT_STATE = Path(
    "data_perp/artifacts/raw_market_state_backward_recurrence_20260726_v1/"
    "weekly_raw_state_diagnostic_rows.parquet"
)
DEFAULT_FORWARD_REPORT = Path(
    "data_perp/artifacts/failure_first_binary_forward_july19_20260726_v1/"
    "report.json"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/"
    "failure_first_current_strict_model_oos_history_20260726_v1"
)
OOF_FLAG = "causal_recent_side_isotonic_ev__is_oof"
FORWARD_FLAG = "causal_recent_side_isotonic_ev__is_forward_oos"
COMBINED_FLAG = "failure_first_score_is_strict_model_oos"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--state-source", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--forward-evaluation-report",
        type=Path,
        default=DEFAULT_FORWARD_REPORT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    ledger_path = Path(args.ledger)
    state_path = Path(args.state_source)
    report_path = Path(args.forward_evaluation_report)
    ledger_hash = _sha256(ledger_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reported_hash = report.get("source_hashes", {}).get(str(ledger_path))
    if reported_hash != ledger_hash:
        raise ValueError(
            "forward evaluation report is not bound to the supplied ledger"
        )
    if report.get("status") != "CROSS_MODEL_FORWARD_TRANSFER_DIAGNOSTIC":
        raise ValueError("forward cohort has not completed the required audit")
    if bool(report.get("promotion_allowed", True)):
        raise ValueError("forward report must remain non-promotional")
    ledger = pd.read_parquet(ledger_path)
    required = (
        "candidate_id",
        "execution_decision_utc",
        "execution_label_end_utc",
        "evaluation_origin",
        "causal_recent_side_isotonic_ev",
        "execution_gross_ev_12h",
        "execution_net_ev_12h",
        OOF_FLAG,
        FORWARD_FLAG,
    )
    missing = [name for name in required if name not in ledger]
    if missing:
        raise KeyError("current history ledger missing: " + ", ".join(missing))
    if ledger["candidate_id"].isna().any() or ledger["candidate_id"].duplicated().any():
        raise ValueError("current history candidate IDs must be unique")
    oof = ledger[OOF_FLAG].fillna(False).astype(bool)
    forward = ledger[FORWARD_FLAG].fillna(False).astype(bool)
    if (oof & forward).any():
        raise ValueError("OOF and forward provenance flags must be exclusive")
    selected = ledger.loc[oof | forward].copy()
    selected[COMBINED_FLAG] = True
    selected["failure_first_history_role"] = np.where(
        selected[OOF_FLAG].fillna(False).astype(bool),
        "outer_oof",
        "retired_resolved_forward_oos",
    )
    selected["execution_decision_utc"] = pd.to_datetime(
        selected["execution_decision_utc"], utc=True, errors="raise"
    )
    selected["execution_label_end_utc"] = pd.to_datetime(
        selected["execution_label_end_utc"], utc=True, errors="raise"
    )
    numeric = (
        selected.loc[
            :,
            [
                "causal_recent_side_isotonic_ev",
                "execution_gross_ev_12h",
                "execution_net_ev_12h",
            ],
        ]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(np.float64)
    )
    if not np.isfinite(numeric).all():
        raise ValueError("strict model-OOS history requires finite score/outcomes")
    retired = selected.loc[
        selected["failure_first_history_role"].eq(
            "retired_resolved_forward_oos"
        )
    ]
    if len(retired) != int(report["forward_rows"]):
        raise ValueError("retired forward row count differs from opened report")
    if retired["execution_decision_utc"].max() != pd.Timestamp(
        report["forward_end"]
    ):
        raise ValueError("retired forward end differs from opened report")
    state_ids = pd.read_parquet(
        state_path, columns=["candidate_id"]
    )["candidate_id"]
    forward_state_matches = int(
        retired["candidate_id"].isin(state_ids).sum()
    )
    if forward_state_matches != len(retired):
        raise ValueError("retired forward cohort lacks exact raw-H0 identity")
    output_path = output / "strict_model_oos_history.parquet"
    selected.to_parquet(output_path, index=False)
    role_counts = (
        selected["failure_first_history_role"].value_counts().to_dict()
    )
    manifest = {
        "schema": "failure_first_current_strict_model_oos_history_v1",
        "status": "MATERIALIZED_RESOLVED_STRICT_MODEL_OOS_HISTORY",
        "combined_score_valid_flag": COMBINED_FLAG,
        "rows": int(len(selected)),
        "role_counts": role_counts,
        "start_utc": selected["execution_decision_utc"].min(),
        "end_utc": selected["execution_decision_utc"].max(),
        "observed_utc_days": int(
            selected["execution_decision_utc"].dt.normalize().nunique()
        ),
        "evaluation_origins": (
            selected.groupby(
                ["failure_first_history_role", "evaluation_origin"],
                observed=True,
            )
            .size()
            .rename("rows")
            .reset_index()
            .to_dict("records")
        ),
        "forward_raw_h0_identity_matches": forward_state_matches,
        "forward_report_status": report["status"],
        "forward_report_promotion_allowed": report["promotion_allowed"],
        "policy": (
            "Retired forward rows may train only a later detector. They remain "
            "forbidden from evaluation of a detector fitted on this history."
        ),
        "source_hashes": {
            str(ledger_path): ledger_hash,
            str(state_path): _sha256(state_path),
            str(report_path): _sha256(report_path),
        },
        "output": {
            "path": str(output_path),
            "rows": int(len(selected)),
            "sha256": _sha256(output_path),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return {
        "status": manifest["status"],
        "rows": int(len(selected)),
        "role_counts": role_counts,
        "end_utc": manifest["end_utc"],
        "output_dir": str(output),
    }


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2))


if __name__ == "__main__":
    main()
