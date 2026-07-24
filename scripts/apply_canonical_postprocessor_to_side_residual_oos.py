#!/usr/bin/env python3
"""Apply the frozen V9/MLP/hier-EV chain to side-residual meta OOS rows.

Postprocessor inputs are refreshed through the repository's logical
point-in-time feature-store reader. That reader combines physical symbol files
with canonical static/delta blocks, matching train and live consumers. The
scorer then fails closed on every remaining non-finite input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.features import _batch_roll_zscore
from extreme_price_movements.features_oi import rolling_robust_zscore_by_symbol
from extreme_price_movements.inference.canonical_meta_postprocessor import (
    CanonicalMetaPostprocessor,
    V9TailPostprocessor,
)
from scripts.run_meta_v9_ev_mapped_side_residual_ablation import (
    _augment_from_feature_store,
)


KEYS = ["__ts__", "__symbol__", "side_name"]
DERIVED_INPUTS = {
    "carry_adj_ret_self_z_10h": "carry_adj_ret_10h",
    "oi_chg_2h_robust_z": "oi_chg_2h",
    "oi_chg_4h_robust_z": "oi_chg_4h",
    "oi_chg_8h_robust_z": "oi_chg_8h",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _old_score_from_rank(postprocessor: CanonicalMetaPostprocessor, rank: pd.Series) -> np.ndarray:
    """Map the new frozen train-reference rank onto V9's trained score domain."""
    values = np.clip(pd.to_numeric(rank, errors="coerce").to_numpy(np.float64), 0.0, 1.0)
    reference = np.asarray(
        postprocessor.predecessor_bundle.historical_rank_reference.sorted_scores_global,
        dtype=np.float64,
    )
    if reference.size < 2:
        raise RuntimeError("V9 predecessor has no usable historical score reference")
    position = values * float(reference.size - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.ceil(position).astype(np.int64)
    return (reference[lower] + (position - lower) * (reference[upper] - reference[lower])).astype(
        np.float32
    )


def _derive_supported_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=False)

    def _current(name: str) -> pd.Series:
        if name not in out:
            return pd.Series(np.nan, index=out.index, dtype=np.float32)
        return pd.to_numeric(out[name], errors="coerce")

    if "carry_adj_ret_10h" in out:
        raw = pd.to_numeric(out["carry_adj_ret_10h"], errors="coerce").astype(np.float32)
        derived = _batch_roll_zscore(raw, 14 * 24).clip(-6.0, 6.0).astype(np.float32)
        current = _current("carry_adj_ret_self_z_10h")
        out["carry_adj_ret_self_z_10h"] = current.where(current.notna(), derived)
    for hours in (2, 4, 8):
        raw_name = f"oi_chg_{hours}h"
        output_name = f"oi_chg_{hours}h_robust_z"
        if raw_name not in out:
            continue
        raw = pd.to_numeric(out[raw_name], errors="coerce").astype(np.float32).to_frame()
        derived = (
            rolling_robust_zscore_by_symbol(raw, 30 * 24, min_periods=7 * 24)
            .iloc[:, 0]
            .clip(-10.0, 10.0)
            .astype(np.float32)
        )
        current = _current(output_name)
        out[output_name] = current.where(current.notna(), derived)
    return out


def _load_symbol_context(
    feature_dir: Path,
    symbol: str,
    wanted: pd.DataFrame,
    required: list[str],
) -> pd.DataFrame:
    path = _feature_file_for_symbol(feature_dir, symbol)
    if not path.exists():
        return wanted.assign(**{name: np.nan for name in required})
    schema = set(pq.read_schema(path).names)
    read_columns = [
        name
        for name in dict.fromkeys([*required, *DERIVED_INPUTS.values()])
        if name in schema
    ]
    stored = pd.read_parquet(path, columns=read_columns)
    stored.index = pd.to_datetime(stored.index, utc=True, errors="coerce")
    stored = stored.loc[~stored.index.duplicated(keep="last")].sort_index()
    stored = _derive_supported_inputs(stored)
    selected = stored.reindex(pd.DatetimeIndex(wanted["__ts__"]))
    selected["__ts__"] = wanted["__ts__"].to_numpy()
    selected["__symbol__"] = symbol
    for name in required:
        if name not in selected:
            selected[name] = np.nan
    return selected[["__ts__", "__symbol__", *required]].reset_index(drop=True)


def _topk_metrics(frame: pd.DataFrame, score_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    timestamp = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    for score_col in score_columns:
        score = pd.to_numeric(frame[score_col], errors="coerce")
        eligible = frame.loc[np.isfinite(score)].copy()
        eligible["__score__"] = score.loc[eligible.index].to_numpy(np.float64)
        for fraction in (0.10, 0.20, 0.30):
            count = max(1, int(np.ceil(len(eligible) * fraction)))
            selected = eligible.nlargest(count, "__score__")
            selected_ts = timestamp.loc[selected.index]
            weekly = selected.groupby(
                selected_ts.dt.to_period("W-SUN").astype(str), observed=True
            )["ev_after_1pct"].mean()
            monthly = selected.groupby(selected_ts.dt.strftime("%Y-%m"), observed=True)[
                "ev_after_1pct"
            ].mean()
            rows.append(
                {
                    "model": score_col,
                    "top_k": int(round(fraction * 100)),
                    "candidate_rows": int(len(eligible)),
                    "selected_rows": int(len(selected)),
                    "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
                    "worst_week_ev_after_1pct": float(weekly.min()),
                    "worst_month_ev_after_1pct": float(monthly.min()),
                    "clean_exec_precision": float(selected["clean_exec"].mean()),
                    "dirty_positive_rate": float(selected["dirty_positive"].mean()),
                    "full_path_bad_mae_rate": float(selected["full_path_bad_mae_1r"].mean()),
                    "timeout_rate": float(selected["timeout"].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-oos", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--start",
        default=None,
        help="Optional inclusive UTC decision timestamp; limits the input ledger before joins.",
    )
    parser.add_argument(
        "--end-exclusive",
        default=None,
        help="Optional exclusive UTC decision timestamp; limits the input ledger before joins.",
    )
    parser.add_argument(
        "--source-rank-col",
        default="score_base_residual_ev_rank_train_reference",
    )
    parser.add_argument(
        "--postprocessor-mode",
        choices=("auto", "v9_only", "canonical"),
        default="auto",
        help=(
            "auto reads policy_config.json. v9_only applies V9 tail plus the "
            "hierarchical side/archetype EV map without MLP/regime effects."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predecessor_path = args.policy_root / "v9_tail95_predecessor_bundle.joblib"
    residual_path = args.policy_root / "residual_event_state.joblib"
    calibration_path = args.policy_root / "composite_policy_regime_ev_calibration.json"
    policy_config = {}
    for policy_config_path in (
        args.policy_root / "optimized_portfolio_policy_config.json",
        args.policy_root / "hit_surprise_archetype_portfolio_policy.json",
        args.policy_root / "policy_config.json",
    ):
        if policy_config_path.is_file():
            policy_config = json.loads(policy_config_path.read_text(encoding="utf-8"))
            break
    mode = str(args.postprocessor_mode)
    if mode == "auto":
        mode = (
            "canonical"
            if bool(policy_config.get("mlp_postprocessor_enabled", True))
            else "v9_only"
        )
    if mode == "v9_only":
        postprocessor = V9TailPostprocessor.load(
            predecessor_bundle_path=predecessor_path,
            residual_event_state_path=residual_path,
            hierarchical_ev_artifact_path=calibration_path,
        )
    else:
        postprocessor = CanonicalMetaPostprocessor.load(
            predecessor_bundle_path=predecessor_path,
            residual_event_state_path=residual_path,
            regime_ev_artifact_path=calibration_path,
        )
    frame = pd.read_parquet(args.input_oos)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    if frame.duplicated(KEYS).any():
        raise ValueError("Input OOS rows must be unique by timestamp/symbol/side")
    if args.source_rank_col not in frame:
        raise KeyError(f"Missing source rank column: {args.source_rank_col}")
    if args.start is not None:
        start = pd.Timestamp(args.start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        frame = frame.loc[frame["__ts__"].ge(start)].copy()
    else:
        start = None
    if args.end_exclusive is not None:
        end = pd.Timestamp(args.end_exclusive)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        frame = frame.loc[frame["__ts__"].lt(end)].copy()
    else:
        end = None
    if frame.empty:
        raise ValueError("No OOS rows remain after the requested time window")

    required = postprocessor.required_input_features()
    # Preserve already materialized point-in-time columns from the OOS ledger
    # and use the logical store only for absent fields. Generated residual-state
    # features are not raw static-store columns and must not be overwritten.
    joined = _augment_from_feature_store(frame, args.feature_dir, required)
    old_score = _old_score_from_rank(postprocessor, joined[args.source_rank_col])
    for name in ("score_meta_base_soft_label", "hit_probability", "calibrated_score"):
        joined[name] = old_score
    complete_report = postprocessor.complete_case_report(joined)
    complete = complete_report["complete_case"].astype(bool)
    scored_input = joined.loc[complete].copy()
    scored = postprocessor.transform(scored_input, copy=False)

    complete_report = pd.concat([joined[KEYS], complete_report], axis=1)
    complete_report.to_parquet(
        args.output_dir / "complete_case_report.parquet", index=False, compression="zstd"
    )
    scored.to_parquet(
        args.output_dir / "postprocessed_oos_predictions.parquet",
        index=False,
        compression="zstd",
    )
    score_columns = [
        args.source_rank_col,
        "historical_rank",
        "market_state_mlp_expected_ev_rank_score",
    ]
    metrics = _topk_metrics(scored, score_columns)
    metrics.to_csv(args.output_dir / "topk_metrics.csv", index=False)
    missing = (
        complete_report.loc[~complete, "missing_features"]
        .str.split(",")
        .explode()
        .loc[lambda values: values.ne("")]
        .value_counts()
        .rename_axis("feature")
        .reset_index(name="rejected_rows")
    )
    missing.to_csv(args.output_dir / "missing_feature_attrition.csv", index=False)
    manifest = {
        "schema": "canonical_postprocessor_side_residual_oos_v1",
        "input_oos": str(args.input_oos),
        "feature_dir": str(args.feature_dir),
        "source_rank_col": args.source_rank_col,
        "input_rows": int(len(joined)),
        "start": start.isoformat() if start is not None else None,
        "end_exclusive": end.isoformat() if end is not None else None,
        "complete_rows": int(complete.sum()),
        "complete_fraction": float(complete.mean()),
        "strict_complete_case": True,
        "derived_inputs": DERIVED_INPUTS,
        "score_domain_bridge": "new_train_reference_rank_to_v9_historical_score_quantile",
        "postprocessor_mode": mode,
        "artifacts": {
            "predecessor": {"path": str(predecessor_path), "sha256": _sha256(predecessor_path)},
            "residual_state": {"path": str(residual_path), "sha256": _sha256(residual_path)},
            "calibration": {"path": str(calibration_path), "sha256": _sha256(calibration_path)},
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
