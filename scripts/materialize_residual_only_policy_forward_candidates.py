#!/usr/bin/env python3
"""Materialize final-refit forward long candidates for policy replay diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _load_long_label_history(labels_dir: Path) -> pd.DataFrame:
    files = sorted(labels_dir.glob("train_global_long_5_*.parquet"))
    wanted = [
        "__ts__",
        "__symbol__",
        "__archetype_policy_key__",
        "__barrier_pct__",
        "__archetype_policy_tp_r__",
        "__archetype_policy_sl_r__",
        "__archetype_policy_trail_r__",
        "__archetype_policy_confidence__",
    ]
    parts = [pd.read_parquet(path, columns=wanted) for path in files]
    if not parts:
        raise FileNotFoundError(f"no long label files under {labels_dir}")
    frame = pd.concat(parts, ignore_index=True, copy=False)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    return frame.sort_values(["__ts__", "__symbol__"], kind="mergesort")


def materialize_forward_candidates(
    scorer: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_rank: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "score",
        "score_base_ev_residual_expert",
        "score_base_ev_residual_expert_hier_mapped",
        "score_base_residual_ev_rank_train_reference",
        "ae_gmm_input_complete",
        "base_input_complete",
        "meta_residual_expert_complete_case",
    }
    missing = sorted(required - set(scorer.columns))
    if missing:
        raise ValueError(f"forward scorer missing columns: {missing}")
    rows = scorer.copy()
    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True, errors="coerce")
    rows = rows.loc[
        rows["side_name"].astype(str).str.lower().eq("long")
        & rows["__ts__"].ge(start)
        & rows["__ts__"].lt(end)
    ].copy()
    complete = np.ones(len(rows), dtype=bool)
    for column in (
        "ae_gmm_input_complete",
        "base_input_complete",
        "meta_residual_expert_complete_case",
    ):
        complete &= pd.to_numeric(rows[column], errors="coerce").fillna(0.0).to_numpy() > 0.5
    rows = rows.loc[complete].copy()
    rows["rank_pct"] = pd.to_numeric(
        rows["score_base_residual_ev_rank_train_reference"], errors="coerce"
    )
    rows = rows.loc[rows["rank_pct"].ge(float(min_rank))].copy()
    if rows.empty:
        raise ValueError("no complete forward candidates pass the frozen rank floor")

    # Point-in-time barrier state: the most recent labelled row for the symbol
    # must precede the forward decision. Archetype geometry itself is mapped
    # separately from train-label history and is not taken from the future row.
    barrier = labels.loc[:, ["__ts__", "__symbol__", "__barrier_pct__"]].copy()
    rows = pd.merge_asof(
        rows.sort_values(["__ts__", "__symbol__"], kind="mergesort"),
        barrier.sort_values(["__ts__", "__symbol__"], kind="mergesort"),
        on="__ts__",
        by="__symbol__",
        direction="backward",
        allow_exact_matches=True,
    )
    geometry_cols = [
        "__archetype_policy_tp_r__",
        "__archetype_policy_sl_r__",
        "__archetype_policy_trail_r__",
    ]
    geometry = (
        labels.groupby("__archetype_policy_key__", observed=True)[geometry_cols]
        .median(numeric_only=True)
        .reset_index()
    )
    confidence = (
        labels.groupby("__archetype_policy_key__", observed=True)[
            "__archetype_policy_confidence__"
        ]
        .agg(lambda values: values.dropna().astype(str).mode().iloc[0] if len(values.dropna()) else "unknown")
        .reset_index()
    )
    rows = rows.merge(
        geometry,
        left_on="archetype_policy_key",
        right_on="__archetype_policy_key__",
        how="left",
        validate="many_to_one",
    ).merge(
        confidence,
        left_on="archetype_policy_key",
        right_on="__archetype_policy_key__",
        how="left",
        validate="many_to_one",
        suffixes=("", "_confidence"),
    )
    barrier_values = pd.to_numeric(rows["__barrier_pct__"], errors="coerce")
    fallback_barrier = float(pd.to_numeric(labels["__barrier_pct__"], errors="coerce").median())
    rows["__barrier_pct__"] = barrier_values.fillna(fallback_barrier)
    if not np.isfinite(rows["__barrier_pct__"]).all() or not rows["__barrier_pct__"].gt(0).all():
        raise ValueError("forward barrier mapping produced invalid values")

    signal = rows["__ts__"]
    decision = signal + pd.Timedelta(hours=1)
    out = pd.DataFrame(
        {
            "timestamp": signal,
            "signal_timestamp": signal,
            "decision_timestamp": decision,
            "first_path_timestamp": decision,
            "entry_timestamp": decision,
            "label_path_end_timestamp": decision + pd.Timedelta(hours=24),
            "symbol": rows["__symbol__"].astype(str),
            "side": np.float32(1.0),
            "side_name": "long",
            "strategy_id": "long_s59_residual_only_oos",
            "policy_archetype": rows["archetype_policy_key"].astype(str),
            "archetype_policy_key": rows["archetype_policy_key"].astype(str),
            "local_side_archetype": "long__"
            + rows["archetype_policy_key"].astype(str).str.removeprefix("long__"),
            "rank_pct": rows["rank_pct"].astype(np.float32),
            "calibrated_score": pd.to_numeric(
                rows["score_base_ev_residual_expert_hier_mapped"], errors="coerce"
            ).astype(np.float32),
            "expected_net_ev_after_1pct": pd.to_numeric(
                rows["score_base_ev_residual_expert_hier_mapped"], errors="coerce"
            ).astype(np.float32),
            "base_score_oof": pd.to_numeric(rows["score"], errors="coerce").astype(np.float32),
            "meta_score_oof": pd.to_numeric(
                rows["score_base_ev_residual_expert"], errors="coerce"
            ).astype(np.float32),
            "barrier_pct": rows["__barrier_pct__"].astype(np.float32),
            "archetype_tp_r": pd.to_numeric(
                rows["__archetype_policy_tp_r__"], errors="coerce"
            ).astype(np.float32),
            "archetype_sl_r": pd.to_numeric(
                rows["__archetype_policy_sl_r__"], errors="coerce"
            ).astype(np.float32),
            "archetype_trail_r": pd.to_numeric(
                rows["__archetype_policy_trail_r__"], errors="coerce"
            ).astype(np.float32),
            "archetype_policy_confidence": rows[
                "__archetype_policy_confidence__"
            ].astype(str),
            "prediction_provenance": "final_refit_forward",
        }
    )
    out = out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    audit: dict[str, object] = {
        "schema": "residual_only_policy_forward_candidates_v1",
        "prediction_provenance": "final_refit_forward",
        "oos_claim": False,
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()),
        "timestamp_min": out["timestamp"].min().isoformat(),
        "timestamp_max": out["timestamp"].max().isoformat(),
        "min_rank": float(min_rank),
        "causal_clock": "decision=signal+1h; first_path>=decision",
        "barrier_source": "latest causal labelled symbol barrier; global train median fallback",
    }
    return out, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--min-rank", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorer = pd.read_parquet(args.scorer)
    labels = _load_long_label_history(args.labels_dir)
    out, audit = materialize_forward_candidates(
        scorer,
        labels,
        start=_utc(args.start),
        end=_utc(args.end_exclusive),
        min_rank=float(args.min_rank),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False, compression="zstd")
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
