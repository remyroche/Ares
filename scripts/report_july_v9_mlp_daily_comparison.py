#!/usr/bin/env python3
"""Compare the prior 8-day policy with frozen V9+MLP on July OOS rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
)
from extreme_price_movements.inference.threshold_basis_policy import (
    load_threshold_basis_policy,
)
from extreme_price_movements.regime_ev_calibration import (
    apply_regime_ev_calibration,
    load_regime_ev_calibration,
    required_feature_columns,
)
from scripts.score_compare_meta_residual_july_oos import (
    _append_store_features,
    _apply_threshold_policy,
    _policy_rank_current,
)


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOMES = [
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consistent-predictions",
        type=Path,
        default=None,
        help=(
            "Preferred single-ledger source scored with one frozen model/rank "
            "contract for the complete evaluation period."
        ),
    )
    parser.add_argument("--early-predictions", type=Path, default=None)
    parser.add_argument("--complete-08-10", type=Path, default=None)
    parser.add_argument("--complete-11-12", type=Path, default=None)
    parser.add_argument("--july-state", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--old-regime-calibration", type=Path, required=True)
    parser.add_argument("--old-threshold-policy", type=Path, required=True)
    parser.add_argument("--old-threshold-reference", type=Path, required=True)
    parser.add_argument("--new-regime-calibration", type=Path, required=True)
    parser.add_argument("--policy-rank-run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_sources(args: argparse.Namespace) -> pd.DataFrame:
    if args.consistent_predictions is not None:
        frame = pd.read_parquet(args.consistent_predictions)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        if "historical_rank" not in frame.columns:
            raise RuntimeError(
                "Consistent prediction ledger is missing frozen historical_rank"
            )
        frame["historical_rank_pre_v9"] = pd.to_numeric(
            frame["historical_rank"], errors="coerce"
        )
        return frame.sort_values(KEYS, kind="stable").drop_duplicates(
            KEYS, keep="last"
        )

    required = {
        "--early-predictions": args.early_predictions,
        "--complete-08-10": args.complete_08_10,
        "--complete-11-12": args.complete_11_12,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "Use --consistent-predictions, or provide all legacy stitched sources: "
            + ", ".join(missing)
        )
    early = pd.read_parquet(args.early_predictions)
    early["__ts__"] = pd.to_datetime(early["__ts__"], utc=True, errors="coerce")
    early = early.loc[early["__ts__"].lt(pd.Timestamp("2026-07-08", tz="UTC"))]
    early["historical_rank_pre_v9"] = pd.to_numeric(
        early.get("historical_rank_alternative"), errors="coerce"
    )
    parts = [early]
    for path in (args.complete_08_10, args.complete_11_12):
        part = pd.read_parquet(path)
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        part["historical_rank_pre_v9"] = pd.to_numeric(
            part.get("historical_rank"), errors="coerce"
        )
        parts.append(part)
    common = set(parts[0].columns)
    for part in parts[1:]:
        common.update(part.columns)
    aligned = [part.reindex(columns=sorted(common)) for part in parts]
    frame = pd.concat(aligned, ignore_index=True, sort=False, copy=False)
    return frame.sort_values(KEYS, kind="stable").drop_duplicates(KEYS, keep="last")


def _merge_july_state(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    state = pd.read_parquet(path)
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True, errors="coerce")
    state_cols = [name for name in state if name.startswith("resid_event_")]
    early = frame["__ts__"].lt(pd.Timestamp("2026-07-11", tz="UTC"))
    left = frame.loc[early].drop(columns=state_cols, errors="ignore")
    left = left.merge(
        state[KEYS + state_cols].drop_duplicates(KEYS, keep="last"),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    return pd.concat([left, frame.loc[~early]], ignore_index=True, sort=False)


def _top_fraction_mask(frame: pd.DataFrame, score_col: str, fraction: float) -> np.ndarray:
    selected = np.zeros(len(frame), dtype=bool)
    score = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=np.float64)
    for positions in frame.groupby("__ts__", sort=False).indices.values():
        pos = np.asarray(positions, dtype=np.int64)
        finite = pos[np.isfinite(score[pos])]
        count = max(1, int(np.ceil(float(fraction) * len(finite)))) if len(finite) else 0
        if count:
            order = finite[np.argsort(-score[finite], kind="stable")[:count]]
            selected[order] = True
    return selected


def _metric_rows(frame: pd.DataFrame, selector: str, selected: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    day = frame["__ts__"].dt.strftime("%Y-%m-%d")
    for date in pd.date_range("2026-07-01", "2026-07-13", freq="D", tz="UTC"):
        date_key = date.strftime("%Y-%m-%d")
        day_mask = day.eq(date_key).to_numpy()
        chosen = day_mask & selected
        ev = pd.to_numeric(frame.loc[chosen, "ev_after_1pct"], errors="coerce")
        finite = ev.notna()
        realized_idx = ev.index[finite]
        row = {
            "day": date_key,
            "selector": selector,
            "candidate_rows": int(day_mask.sum()),
            "selected_rows": int(chosen.sum()),
            "realized_rows": int(finite.sum()),
            "mean_net_ev_per_trade": float(ev.loc[finite].mean()) if finite.any() else np.nan,
            "sum_net_ev_notional": float(ev.loc[finite].sum()) if finite.any() else np.nan,
            "positive_ev_rate": float((ev.loc[finite] > 0).mean()) if finite.any() else np.nan,
        }
        for name, output in (
            ("clean_exec", "clean_exec_rate"),
            ("dirty_positive", "dirty_positive_rate"),
            ("first_touch_bad_mae_1r", "first_touch_bad_mae_rate"),
            ("full_path_bad_mae_1r", "full_path_bad_mae_rate"),
            ("timeout", "timeout_rate"),
        ):
            values = pd.to_numeric(frame.loc[realized_idx, name], errors="coerce")
            row[output] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(row)
    return rows


def main() -> int:
    args = _args()
    frame = _merge_july_state(_load_sources(args), args.july_state)

    old_artifact = load_regime_ev_calibration(args.old_regime_calibration)
    new_artifact = load_regime_ev_calibration(args.new_regime_calibration)
    store_features = [
        name
        for name in required_feature_columns(old_artifact)
        if name not in frame.columns
    ]
    if store_features:
        frame, _ = _append_store_features(frame, args.feature_root, store_features)

    frame["calibrated_score"] = pd.to_numeric(
        frame["historical_rank_pre_v9"], errors="coerce"
    ).astype(np.float32)
    new_required = required_feature_columns(new_artifact)
    missing = sorted(name for name in new_required if name not in frame.columns)
    if missing:
        raise RuntimeError("Missing frozen V9+MLP features: " + ", ".join(missing))
    frame = apply_regime_ev_calibration(
        frame,
        new_artifact,
        source_score_col="calibrated_score",
        adjusted_score_col="score_v9_mlp",
        copy=False,
    )

    frame = apply_regime_ev_calibration(
        frame,
        old_artifact,
        source_score_col="score_meta_base_soft_label",
        adjusted_score_col="score_old_regime_raw_meta",
        copy=False,
    )
    frame = apply_regime_ev_calibration(
        frame,
        old_artifact,
        source_score_col="score_shock_adjusted",
        adjusted_score_col="score_old_regime_sparse_shock",
        copy=False,
    )
    rank_store = PolicyRankReferenceStore(
        data_root="data_perp", run_id=args.policy_rank_run_id
    )
    frame["policy_rank_old_baseline"] = _policy_rank_current(
        frame,
        store=rank_store,
        raw_score_col="score_meta_base_soft_label",
        adjusted_score_col="score_old_regime_raw_meta",
    )
    old_policy = load_threshold_basis_policy(args.old_threshold_policy)
    old_policy["reference_candidates_path"] = str(args.old_threshold_reference)
    frame = pd.concat(
        [
            frame,
            _apply_threshold_policy(
                frame,
                policy=old_policy,
                score_col="score_old_regime_sparse_shock",
                baseline_rank_col="policy_rank_old_baseline",
                prefix="old_8d",
            ),
        ],
        axis=1,
        copy=False,
    )

    masks = {
        "previous_8d_policy": frame["old_8d_selected"].fillna(False).to_numpy(bool),
        "v9_mlp_fixed_rank_ge_090": pd.to_numeric(
            frame["expected_ev_rank_score"], errors="coerce"
        ).ge(0.90).to_numpy(),
        "previous_equal_budget_top10": _top_fraction_mask(
            frame, "score_old_regime_sparse_shock", 0.10
        ),
        "v9_mlp_equal_budget_top10": _top_fraction_mask(
            frame, "expected_ev_rank_score", 0.10
        ),
    }
    metrics = pd.DataFrame(
        [row for name, mask in masks.items() for row in _metric_rows(frame, name, mask)]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        args.output_dir / "july_01_13_v9_mlp_comparison_ledger.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(args.output_dir / "daily_metrics.csv", index=False)
    pivot = metrics.pivot(index="day", columns="selector", values="mean_net_ev_per_trade")
    pivot.to_csv(args.output_dir / "daily_mean_net_ev_pivot.csv")
    comparison_metrics = [
        "selected_rows",
        "realized_rows",
        "mean_net_ev_per_trade",
        "sum_net_ev_notional",
        "positive_ev_rate",
        "clean_exec_rate",
        "full_path_bad_mae_rate",
        "timeout_rate",
    ]
    policy_rows = metrics.loc[
        metrics["selector"].isin(
            ["previous_8d_policy", "v9_mlp_fixed_rank_ge_090"]
        )
    ]
    comparison = policy_rows.pivot(
        index="day", columns="selector", values=comparison_metrics
    )
    comparison.columns = [f"{metric}__{selector}" for metric, selector in comparison]
    for metric in comparison_metrics:
        old_col = f"{metric}__previous_8d_policy"
        new_col = f"{metric}__v9_mlp_fixed_rank_ge_090"
        comparison[f"delta_{metric}__v9_mlp_minus_previous"] = (
            comparison[new_col] - comparison[old_col]
        )
    comparison.reset_index().to_csv(
        args.output_dir / "daily_policy_comparison.csv", index=False
    )
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
