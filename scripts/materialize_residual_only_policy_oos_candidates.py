#!/usr/bin/env python3
"""Materialize strict OOS residual-meta rows for simple policy optimisation.

The materializer deliberately does not refit, rerank, or recalibrate predictions.
It preserves the train-derived hierarchical EV map and percentile emitted by the
meta fold model, then joins only causal path geometry from the matching base
handoff ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["__ts__", "__symbol__", "side_name"]
PREDICTION_COLUMNS = [
    *KEYS,
    "__label_path_end_ts__",
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
    "score_base",
    "score_base_ev_mapped",
    "score_base_ev_residual_expert",
    "score_base_ev_residual_expert_hier_mapped",
    "meta_residual_expert_delta_ev",
    "score_base_ev_rank_train_reference",
    "score_base_residual_ev_rank_train_reference",
    "archetype_policy_key",
    "calendar_month",
    "week_start",
]
LEDGER_COLUMNS = [
    *KEYS,
    "__signal_ts__",
    "__decision_ts__",
    "__first_path_ts__",
    "__entry_ts__",
    "__label_path_end_ts__",
    "__barrier_pct__",
    "__archetype_policy_key__",
    "__archetype_policy_tp_r__",
    "__archetype_policy_sl_r__",
    "__archetype_policy_trail_r__",
    "__archetype_policy_confidence__",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _require_columns(frame: pd.DataFrame, required: list[str], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def materialize_candidates(
    predictions: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    side_name: str = "long",
    min_rank: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    side_name = str(side_name).strip().lower()
    if side_name not in {"long", "short"}:
        raise ValueError(f"unsupported side_name: {side_name}")
    _require_columns(predictions, PREDICTION_COLUMNS, "predictions")
    _require_columns(ledger, LEDGER_COLUMNS, "ledger")

    pred = predictions.loc[
        predictions["side_name"].astype(str).str.lower().eq(str(side_name).lower()),
        PREDICTION_COLUMNS,
    ].copy()
    paths = ledger.loc[
        ledger["side_name"].astype(str).str.lower().eq(str(side_name).lower()),
        LEDGER_COLUMNS,
    ].copy()
    for frame in (pred, paths):
        for column in (
            "__ts__",
            "__signal_ts__",
            "__decision_ts__",
            "__first_path_ts__",
            "__entry_ts__",
            "__label_path_end_ts__",
        ):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        frame["__symbol__"] = frame["__symbol__"].astype(str)
        frame["side_name"] = frame["side_name"].astype(str).str.lower()

    if pred.duplicated(KEYS).any():
        raise ValueError("predictions contain duplicate timestamp/symbol/side keys")
    if paths.duplicated(KEYS).any():
        raise ValueError("ledger contains duplicate timestamp/symbol/side keys")

    merged = pred.merge(
        paths,
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("_pred", "_path"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = int(merged["_merge"].ne("both").sum())
        raise ValueError(f"{missing} prediction rows have no matching causal path row")
    merged = merged.drop(columns="_merge")

    pred_end = merged.pop("__label_path_end_ts___pred")
    path_end = merged.pop("__label_path_end_ts___path")
    if not pred_end.eq(path_end).all():
        raise ValueError("prediction and path ledgers disagree on label resolution time")
    merged["__label_path_end_ts__"] = path_end

    signal = merged["__signal_ts__"]
    decision = merged["__decision_ts__"]
    first_path = merged["__first_path_ts__"]
    entry = merged["__entry_ts__"]
    expected_decision = signal + pd.Timedelta(hours=1)
    timing_valid = (
        signal.notna()
        & decision.eq(expected_decision)
        & first_path.ge(decision)
        & entry.ge(decision)
        & merged["__label_path_end_ts__"].gt(decision)
    )
    if not timing_valid.all():
        raise ValueError(f"causal path timing invalid for {int((~timing_valid).sum())} rows")

    archetype_match = merged["archetype_policy_key"].astype(str).eq(
        merged["__archetype_policy_key__"].astype(str)
    )
    if not archetype_match.all():
        raise ValueError(f"archetype mismatch for {int((~archetype_match).sum())} rows")

    finite_columns = [
        "score_base_residual_ev_rank_train_reference",
        "score_base_ev_residual_expert_hier_mapped",
        "__barrier_pct__",
    ]
    finite = np.ones(len(merged), dtype=bool)
    for column in finite_columns:
        finite &= np.isfinite(pd.to_numeric(merged[column], errors="coerce").to_numpy())
    finite &= pd.to_numeric(merged["__barrier_pct__"], errors="coerce").to_numpy() > 0.0
    if not finite.all():
        raise ValueError(f"required policy values invalid for {int((~finite).sum())} rows")

    out = pd.DataFrame(
        {
            "timestamp": merged["__ts__"],
            "signal_timestamp": signal,
            "decision_timestamp": decision,
            "first_path_timestamp": first_path,
            "entry_timestamp": entry,
            "label_path_end_timestamp": merged["__label_path_end_ts__"],
            "symbol": merged["__symbol__"],
            "side": np.float32(1.0 if side_name == "long" else -1.0),
            "side_name": side_name,
            "strategy_id": f"{side_name}_s59_residual_only_oos",
            "policy_archetype": merged["archetype_policy_key"].astype(str),
            "archetype_policy_key": merged["archetype_policy_key"].astype(str),
            "local_side_archetype": side_name
            + "__"
            + merged["archetype_policy_key"]
            .astype(str)
            .str.removeprefix(f"{side_name}__"),
            "rank_pct": pd.to_numeric(
                merged["score_base_residual_ev_rank_train_reference"], errors="coerce"
            ).astype(np.float32),
            "calibrated_score": pd.to_numeric(
                merged["score_base_ev_residual_expert_hier_mapped"], errors="coerce"
            ).astype(np.float32),
            "expected_net_ev_after_1pct": pd.to_numeric(
                merged["score_base_ev_residual_expert_hier_mapped"], errors="coerce"
            ).astype(np.float32),
            "base_score_oof": pd.to_numeric(merged["score_base"], errors="coerce").astype(
                np.float32
            ),
            "meta_score_oof": pd.to_numeric(
                merged["score_base_ev_residual_expert"], errors="coerce"
            ).astype(np.float32),
            "barrier_pct": pd.to_numeric(merged["__barrier_pct__"], errors="coerce").astype(
                np.float32
            ),
            "archetype_tp_r": pd.to_numeric(
                merged["__archetype_policy_tp_r__"], errors="coerce"
            ).astype(np.float32),
            "archetype_sl_r": pd.to_numeric(
                merged["__archetype_policy_sl_r__"], errors="coerce"
            ).astype(np.float32),
            "archetype_trail_r": pd.to_numeric(
                merged["__archetype_policy_trail_r__"], errors="coerce"
            ).astype(np.float32),
            "archetype_policy_confidence": pd.to_numeric(
                merged["__archetype_policy_confidence__"], errors="coerce"
            ).astype(np.float32),
            "ev_after_1pct": pd.to_numeric(merged["ev_after_1pct"], errors="coerce").astype(
                np.float32
            ),
            "clean_exec": pd.to_numeric(merged["clean_exec"], errors="coerce").astype(
                np.float32
            ),
            "dirty_positive": pd.to_numeric(
                merged["dirty_positive"], errors="coerce"
            ).astype(np.float32),
            "full_path_bad_mae_1r": pd.to_numeric(
                merged["full_path_bad_mae_1r"], errors="coerce"
            ).astype(np.float32),
            "timeout": pd.to_numeric(merged["timeout"], errors="coerce").astype(np.float32),
            "meta_residual_expert_delta_ev": pd.to_numeric(
                merged["meta_residual_expert_delta_ev"], errors="coerce"
            ).astype(np.float32),
            "score_base_ev_mapped": pd.to_numeric(
                merged["score_base_ev_mapped"], errors="coerce"
            ).astype(np.float32),
            "calendar_month": merged["calendar_month"].astype(str),
            "week_start": merged["week_start"].astype(str),
        }
    )
    out = out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    rows_before_rank_filter = int(len(out))
    if min_rank is not None:
        out = out.loc[out["rank_pct"].ge(float(min_rank))].reset_index(drop=True)
        if out.empty:
            raise ValueError(f"no candidates remain after rank_pct >= {min_rank}")
    audit = {
        "schema": "residual_only_policy_oos_candidates_v1",
        "side": side_name,
        "rows": int(len(out)),
        "rows_before_rank_filter": rows_before_rank_filter,
        "min_rank": None if min_rank is None else float(min_rank),
        "symbols": int(out["symbol"].nunique()),
        "timestamp_min": out["timestamp"].min(),
        "timestamp_max": out["timestamp"].max(),
        "label_path_end_max": out["label_path_end_timestamp"].max(),
        "archetypes": out["policy_archetype"].value_counts().to_dict(),
        "causal_timing": {
            "decision_equals_signal_plus_timeframe": True,
            "timeframe": "1h",
            "first_path_at_or_after_decision": True,
            "entry_at_or_after_decision": True,
        },
        "score_contract": {
            "rank_pct": "frozen train-reference percentile of hierarchical residual EV",
            "calibrated_score": "train-only side_x_archetype hierarchical expected EV after 1pct",
            "no_materializer_rerank": True,
            "no_materializer_refit": True,
        },
        "cost_contract": {
            "diagnostic_outcome": "ev_after_1pct includes the fixed 1pct round-trip cost once",
            "optimizer_candidate_has_precomputed_net_return": False,
        },
    }
    return out, audit


def repair_prediction_path_end_from_ledger(
    predictions: pd.DataFrame,
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repair the known pre-fix DuckDB timezone cast from authoritative paths."""
    pred = predictions.copy()
    source_end = pd.to_datetime(pred["__label_path_end_ts__"], utc=True, errors="coerce")
    lookup = ledger.loc[:, [*KEYS, "__label_path_end_ts__"]].copy()
    lookup["__label_path_end_ts__"] = pd.to_datetime(
        lookup["__label_path_end_ts__"], utc=True, errors="coerce"
    )
    if lookup.duplicated(KEYS).any():
        raise ValueError("cannot repair path end from a duplicate ledger")
    joined = pred.loc[:, KEYS].merge(
        lookup,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    repaired_end = joined["__label_path_end_ts__"]
    if repaired_end.isna().any():
        raise ValueError("cannot repair path end: authoritative ledger rows are missing")
    delta_seconds = (source_end - repaired_end).dt.total_seconds()
    mismatch = ~source_end.eq(repaired_end)
    pred["__label_path_end_ts__"] = repaired_end.to_numpy()
    return pred, {
        "enabled": True,
        "source_mismatch_rows": int(mismatch.sum()),
        "source_rows": int(len(pred)),
        "delta_seconds_min": float(delta_seconds.min()),
        "delta_seconds_max": float(delta_seconds.max()),
        "repair_source": "authoritative corrected causal scored ledger",
        "reason": "pre-fix DuckDB TIMESTAMPTZ-to-TIMESTAMP host-timezone cast",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--scored-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument(
        "--min-rank",
        type=float,
        default=None,
        help="Optional frozen train-reference rank floor for a bounded replay universe.",
    )
    parser.add_argument(
        "--repair-prediction-path-end-from-ledger",
        action="store_true",
        help=(
            "Repair only the known pre-fix timezone-shifted prediction path-end "
            "metadata from the authoritative causal ledger. Scores are unchanged."
        ),
    )
    args = parser.parse_args()

    predictions = pd.read_parquet(args.predictions, columns=PREDICTION_COLUMNS)
    ledger = pd.read_parquet(args.scored_ledger, columns=LEDGER_COLUMNS)
    path_end_repair: dict[str, Any] = {"enabled": False}
    if args.repair_prediction_path_end_from_ledger:
        predictions, path_end_repair = repair_prediction_path_end_from_ledger(
            predictions,
            ledger,
        )
    candidates, audit = materialize_candidates(
        predictions,
        ledger,
        side_name=args.side,
        min_rank=args.min_rank,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.out, index=False, compression="zstd")
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    manifest = {
        **audit,
        "predictions_path": str(args.predictions),
        "predictions_sha256": _sha256(args.predictions),
        "scored_ledger_path": str(args.scored_ledger),
        "scored_ledger_sha256": _sha256(args.scored_ledger),
        "prediction_path_end_repair": path_end_repair,
        "output_path": str(args.out),
        "output_sha256": _sha256(args.out),
    }
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
