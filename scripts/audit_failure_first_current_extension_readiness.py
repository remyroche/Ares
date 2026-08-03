#!/usr/bin/env python3
"""Audit whether any local artifact can extend failure-first current history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PIPELINE = Path(
    "data_perp/artifacts/failure_first_regime_pipeline_20260726_v6"
)
DEFAULT_HISTORY = Path(
    "data_perp/artifacts/"
    "failure_first_current_strict_model_oos_history_20260726_v1/"
    "strict_model_oos_history.parquet"
)
DEFAULT_STATE = Path(
    "data_perp/artifacts/raw_market_state_backward_recurrence_20260726_v1/"
    "weekly_raw_state_diagnostic_rows.parquet"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/"
    "failure_first_current_extension_readiness_20260726_v1"
)
DEFAULT_CANDIDATES = (
    Path(
        "data_perp/artifacts/"
        "failure_first_current_strict_model_oos_history_20260726_v1/"
        "strict_model_oos_history.parquet"
    ),
    Path(
        "data_perp/artifacts/execution_ev_alpha_oof_july20_20260726_v1/"
        "alpha_oof.parquet"
    ),
    Path(
        "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/"
        "execution_ev_policy_labels.parquet"
    ),
    Path(
        "data_perp/artifacts/"
        "packb_side_local_outer_oof_july20_20260726_v1_31_8/"
        "oof_predictions.parquet"
    ),
    Path(
        "data_perp/artifacts/"
        "packb_side_local_residual_oof_july20_20260726_v1_31_8/"
        "oof_predictions.parquet"
    ),
)
COMBINED_FLAG = "failure_first_score_is_strict_model_oos"
REQUIRED_LEDGER_COLUMNS = (
    "candidate_id",
    "execution_decision_utc",
    "execution_label_end_utc",
    "evaluation_origin",
    "causal_recent_side_isotonic_ev",
    "execution_gross_ev_12h",
    "execution_net_ev_12h",
    COMBINED_FLAG,
)


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _decision_time(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    for name in (
        "execution_decision_utc",
        "__decision_ts__",
        "decision_utc",
    ):
        if name in frame:
            return (
                pd.to_datetime(frame[name], utc=True, errors="coerce"),
                name,
            )
    if "__ts__" in frame:
        return (
            pd.to_datetime(
                frame["__ts__"], utc=True, errors="coerce"
            )
            + pd.Timedelta(hours=1),
            "__ts__ + 1h",
        )
    return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"), ""


def _audit_candidate(
    path: Path,
    *,
    state_ids: pd.Index,
    history_end: pd.Timestamp,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "status": "MISSING",
            "usable_extension_rows": 0,
        }
    frame = pd.read_parquet(path)
    timestamp, timestamp_contract = _decision_time(frame)
    missing = sorted(set(REQUIRED_LEDGER_COLUMNS).difference(frame))
    later = timestamp.gt(history_end)
    state_matches = (
        frame["candidate_id"].isin(state_ids)
        if "candidate_id" in frame
        else pd.Series(False, index=frame.index)
    )
    complete = pd.Series(True, index=frame.index)
    if not missing:
        complete &= frame[COMBINED_FLAG].fillna(False).astype(bool)
        complete &= frame[
            [
                "causal_recent_side_isotonic_ev",
                "execution_gross_ev_12h",
                "execution_net_ev_12h",
            ]
        ].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        complete &= pd.to_datetime(
            frame["execution_label_end_utc"],
            utc=True,
            errors="coerce",
        ).notna()
        complete &= state_matches
    else:
        complete &= False
    usable = later & complete
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": "READY_EXTENSION" if usable.any() else "NOT_READY",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "decision_timestamp_contract": timestamp_contract,
        "decision_start_utc": timestamp.min(),
        "decision_end_utc": timestamp.max(),
        "rows_later_than_current_history": int(later.sum()),
        "missing_failure_first_columns": missing,
        "candidate_id_present": "candidate_id" in frame,
        "raw_h0_identity_matches": int(state_matches.sum()),
        "raw_h0_identity_match_rate": float(state_matches.mean())
        if len(frame)
        else 0.0,
        "usable_extension_rows": int(usable.sum()),
        "diagnosis": (
            "Complete strict model-OOS failure-first ledger."
            if not missing
            else "Upstream or label-only artifact; execution-EV mapped score "
            "and/or explicit combined provenance is absent."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-pipeline", type=Path, default=DEFAULT_PIPELINE
    )
    parser.add_argument("--current-history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--state-source", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--candidate",
        action="append",
        type=Path,
        default=None,
        help="Optional candidate parquet; repeat to override default inventory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    gate_path = Path(args.current_pipeline) / "sufficiency_gate.json"
    manifest_path = Path(args.current_pipeline) / "manifest.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    pipeline_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = pd.read_parquet(args.current_history)
    history_time, contract = _decision_time(history)
    history_end = history_time.max()
    state_ids = pd.Index(
        pd.read_parquet(
            args.state_source, columns=["candidate_id"]
        )["candidate_id"]
    )
    candidates = (
        tuple(args.candidate) if args.candidate else DEFAULT_CANDIDATES
    )
    inventory = [
        _audit_candidate(
            path, state_ids=state_ids, history_end=history_end
        )
        for path in candidates
    ]
    criteria = gate["criteria"]

    def deficit(name: str) -> int:
        item = criteria[name]
        return max(0, int(item["required"]) - int(item["observed"]))

    deficits = {
        "additional_observed_days": deficit("observed_calendar_days"),
        "additional_failure_episodes": deficit("failure_episodes"),
        "additional_complete_window_episodes": deficit(
            "complete_window_episodes"
        ),
        "additional_failure_bins": deficit("failure_bins"),
    }
    ready_sources = [
        item for item in inventory if item["usable_extension_rows"] > 0
    ]
    ready_to_extend = bool(ready_sources)
    ready_to_fit = bool(gate["taxonomy_training_allowed"])
    report = {
        "schema": "failure_first_current_extension_readiness_v1",
        "status": (
            "READY_TO_REFIT"
            if ready_to_fit
            else "READY_TO_EXTEND_AND_RECOMPUTE_GATES"
            if ready_to_extend
            else "WAITING_FOR_NEW_SAME_MODEL_HISTORY"
        ),
        "current_history": {
            "path": str(Path(args.current_history)),
            "sha256": _sha256(Path(args.current_history)),
            "rows": int(len(history)),
            "decision_timestamp_contract": contract,
            "start_utc": history_time.min(),
            "end_utc": history_end,
            "observed_utc_days": int(
                history_time.dt.normalize().nunique()
            ),
            "score_valid_flag": pipeline_manifest["score_valid_flag"],
        },
        "current_gate_status": gate["status"],
        "current_gate_criteria": criteria,
        "minimum_remaining_deficits": deficits,
        "candidate_inventory": inventory,
        "ready_extension_sources": ready_sources,
        "ready_to_extend": ready_to_extend,
        "ready_to_fit": ready_to_fit,
        "next_source_contract": {
            "must_begin_after_utc": history_end,
            "required_columns": list(REQUIRED_LEDGER_COLUMNS),
            "required_properties": [
                "same frozen score/model lineage or a new explicit origin",
                "strict OOF or previously opened and retired forward OOS",
                "finite mapped score and exact 12h gross/net outcomes",
                "label availability resolved before detector fitting",
                "candidate-identity raw-H0 join",
                "no reuse as evaluation after retirement into training",
            ],
        },
        "source_hashes": {
            str(gate_path): _sha256(gate_path),
            str(manifest_path): _sha256(manifest_path),
            str(Path(args.state_source)): _sha256(Path(args.state_source)),
        },
    }
    _write_json(output / "readiness.json", report)
    return {
        "status": report["status"],
        "history_end_utc": history_end,
        "ready_extension_sources": int(len(ready_sources)),
        "minimum_remaining_deficits": deficits,
        "output_dir": str(output),
    }


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2))


if __name__ == "__main__":
    main()
