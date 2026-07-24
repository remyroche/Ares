#!/usr/bin/env python3
"""Score frozen current and residual-meta models on the aligned July handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.data_store import read_symbol_features
from extreme_price_movements.inference.policy_rank_reference import (
    PolicyRankReferenceStore,
)
from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
    load_threshold_basis_policy,
)
from extreme_price_movements.meta_historical_rank import HistoricalScoreRankReference
from extreme_price_movements.regime_ev_calibration import (
    apply_regime_ev_calibration,
    load_regime_ev_calibration,
)
from extreme_price_movements.regime_ev_calibration import (
    required_feature_columns as regime_required_feature_columns,
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    _add_reference_fold_features,
)

ROOT = Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1")
RUN_ROOT = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v6_15mchart_base_frozenfs_fixedparams_"
    "may_july_combined_20260708"
)
DEFAULT_HANDOFF = (
    RUN_ROOT
    / "s52_trailing_regime_meta_handoff_top30_allsafe_20260708"
    / "train_meta_regime_handoff.parquet"
)
DEFAULT_OLD_PREDICTIONS = (
    RUN_ROOT
    / "train_meta_frozenfs_fixedparams_train_may_june_score_july_20260709_savedmodels"
    / "s52_train_meta_regime_handoff_smoke_predictions.parquet"
)
DEFAULT_AEGMM = (
    RUN_ROOT
    / "threshold_basis_ablation_may_june_july_weighted_gmm_posterior_8d_v2_overlay_parity"
    / "combined_may_june_july_candidates_with_parity_frozen_aegmm_posteriors.parquet"
)
DEFAULT_BUNDLE = (
    ROOT
    / "inference_bundle_residual_pca8_globaloverlay_shock"
    / "alternative_meta_residual_pca8_shock_bundle.joblib"
)
DEFAULT_FEATURE_ROOT = Path("data_perp/features/20260711_070000")
DEFAULT_TRAIN_REFERENCE = ROOT / "cache" / "compact_reference_with_lifecycle.parquet"
DEFAULT_OUTPUT = ROOT / "july_oos_comparison"
DEFAULT_POLICY_ROOT = Path(
    "data_perp/artifacts/s59_s52_frozen_inference_bundle_20260709/policy_params"
)
DEFAULT_THRESHOLD_POLICY = DEFAULT_POLICY_ROOT / "threshold_basis_policy.json"
DEFAULT_REGIME_CALIBRATION = DEFAULT_POLICY_ROOT / "regime_ev_calibration.json"
DEFAULT_POLICY_RANK_RUN_ID = "s59_s52_frozen_native_shadow_20260709"
DEFAULT_HISTORICAL_ALTERNATIVE = (
    ROOT
    / "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay_sparse_shock_composite"
    / "oos_predictions_historical_rank.parquet"
)

KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOMES = [
    "ev_after_1pct",
    "exec_margin",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
]
MARKET_PREFIXES = ("mkt_", "market_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _columns(path: Path) -> list[str]:
    return [str(name) for name in pq.ParquetFile(path).schema_arrow.names]


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=False)
    if "timestamp" in out.columns and "__ts__" not in out.columns:
        out = out.rename(columns={"timestamp": "__ts__"})
    if "symbol" in out.columns and "__symbol__" not in out.columns:
        out = out.rename(columns={"symbol": "__symbol__"})
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["side_name"] = out["side_name"].astype(str).str.lower()
    if "archetype_policy_key" not in out.columns:
        for fallback in ("__archetype_policy_key__", "policy_archetype"):
            if fallback in out.columns:
                out["archetype_policy_key"] = out[fallback].astype(str)
                break
    out["archetype_policy_key"] = out["archetype_policy_key"].astype(str)
    return out


def _read_july_handoff(
    path: Path, required: Iterable[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    available = set(_columns(path))
    requested = list(
        dict.fromkeys(
            KEYS
            + [name for name in ("score", "selected_top30") if name in available]
            + [name for name in required if name in available]
        )
    )
    frame = _canonicalize(pd.read_parquet(path, columns=requested))
    frame = frame.loc[frame["__ts__"].ge(start) & frame["__ts__"].lt(end)].copy()
    return (
        frame.drop_duplicates(KEYS, keep="last")
        .sort_values(KEYS, kind="stable")
        .reset_index(drop=True)
    )


def _merge_old_predictions(
    frame: pd.DataFrame, path: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    available = set(_columns(path))
    requested = [
        name
        for name in KEYS
        + OUTCOMES
        + [
            "score_meta_base_soft_label",
            "rel_rankband_clean_rate",
            "rel_rankband_bad_mae_rate",
            "rel_rankband_timeout_rate",
            "rel_rankband_exec_margin_mean",
            "rel_rankband_edge",
            "rel_marginband_clean_rate",
            "rel_marginband_timeout_rate",
            "rel_marginband_exec_margin_mean",
        ]
        if name in available
    ]
    old = _canonicalize(pd.read_parquet(path, columns=requested))
    old = old.loc[old["__ts__"].ge(start) & old["__ts__"].lt(end)]
    old = old.drop_duplicates(KEYS, keep="last")
    return frame.merge(
        old, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "__old")
    )


def _merge_frozen_aegmm(
    frame: pd.DataFrame, path: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    available = _columns(path)
    state_cols = [
        name
        for name in available
        if any(
            token in name.lower()
            for token in (
                "gmm",
                "mahal",
                "reconstruction",
                "cluster_speed",
                "cluster_acceleration",
            )
        )
    ]
    requested = [
        name
        for name in [
            "timestamp",
            "symbol",
            "side_name",
            "archetype_policy_key",
            *state_cols,
        ]
        if name in available
    ]
    state = _canonicalize(pd.read_parquet(path, columns=requested))
    state = state.loc[state["__ts__"].ge(start) & state["__ts__"].lt(end)]
    state = state.drop_duplicates(KEYS, keep="last")
    add = [name for name in state_cols if name not in frame.columns]
    return frame.merge(state[KEYS + add], on=KEYS, how="left", validate="one_to_one")


def _append_store_features(
    frame: pd.DataFrame, feature_root: Path, requested: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    names = [name for name in dict.fromkeys(requested) if name not in frame.columns]
    if not names:
        return frame, {}
    values = np.full((len(frame), len(names)), np.nan, dtype=np.float32)
    grouped = frame.groupby("__symbol__", sort=False).indices
    for symbol, raw_positions in grouped.items():
        positions = np.asarray(raw_positions, dtype=np.int64)
        path = feature_root / f"symbol={str(symbol).replace('/', '_')}.parquet"
        if not path.exists():
            continue
        timestamps = frame.iloc[positions]["__ts__"]
        features = read_symbol_features(
            str(path),
            columns=names,
            start_ts=timestamps.min(),
            end_ts=timestamps.max(),
        )
        if features.empty:
            continue
        features.index = pd.to_datetime(features.index, utc=True, errors="coerce")
        features = features[~features.index.duplicated(keep="last")]
        aligned = features.reindex(timestamps.to_numpy())
        available = [name for name in names if name in aligned.columns]
        if available:
            dest = [names.index(name) for name in available]
            values[np.ix_(positions, dest)] = aligned[available].to_numpy(
                dtype=np.float32, copy=False
            )
    appended = pd.DataFrame(values, columns=names, index=frame.index)
    out = pd.concat([frame, appended], axis=1, copy=False)
    # Market-wide columns are broadcast features. Recover isolated symbol gaps
    # from the same timestamp's cross-sectional median without using outcomes.
    market_names = [name for name in names if name.startswith(MARKET_PREFIXES)]
    if market_names:
        out[market_names] = out.groupby("__ts__", sort=False)[market_names].transform(
            "median"
        )
    coverage = {
        name: float(pd.to_numeric(out[name], errors="coerce").notna().mean())
        for name in names
    }
    return out, coverage


def _fit_old_rank_reference(path: Path) -> HistoricalScoreRankReference:
    available = set(_columns(path))
    required = ["__ts__", "side_name", "score_meta_base_soft_label"]
    if not set(required).issubset(available):
        raise ValueError(
            f"Current-reference rank source is missing {sorted(set(required) - available)}"
        )
    train = pd.read_parquet(path, columns=required)
    train["__ts__"] = pd.to_datetime(train["__ts__"], utc=True, errors="coerce")
    train = train.loc[train["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC"))]
    train = train.dropna(subset=["score_meta_base_soft_label"])
    return HistoricalScoreRankReference(
        score_col="score_meta_base_soft_label",
        side_col="side_name",
    ).fit(train)


def _apply_reference_priors(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    available = set(_columns(path))
    requested = [
        name
        for name in KEYS
        + [
            "score",
            "selected_top30",
            "clean_exec",
            "full_path_bad_mae_1r",
            "timeout",
            "dirty_positive",
            "exec_margin",
        ]
        if name in available
    ]
    train = _canonicalize(pd.read_parquet(path, columns=requested))
    train = train.loc[train["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC"))]
    _train_enriched, valid_enriched = _add_reference_fold_features(train, frame)
    return valid_enriched


def _metric_rows(frame: pd.DataFrame, selector: str, rank_col: str) -> pd.DataFrame:
    selected = frame.loc[
        pd.to_numeric(frame[rank_col], errors="coerce").ge(0.90)
    ].copy()
    selected["selector"] = selector
    return selected


def _policy_archetype(frame: pd.DataFrame) -> pd.Series:
    side = frame["side_name"].astype(str).str.lower()
    arch = frame["archetype_policy_key"].astype(str)
    prefixed = arch.str.startswith("long__") | arch.str.startswith("short__")
    return arch.where(prefixed, side + "__" + arch)


def _policy_rank_current(
    frame: pd.DataFrame,
    *,
    store: PolicyRankReferenceStore,
    raw_score_col: str,
    adjusted_score_col: str,
) -> np.ndarray:
    out = np.full(len(frame), np.nan, dtype=np.float32)
    floor = 0.90
    retained = 0.50
    for side, positions in frame.groupby("side_name", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        strategy_id = f"{str(side).lower()}_s52_meta_threshold_handoff"
        raw = pd.to_numeric(frame.iloc[pos][raw_score_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
        adjusted = pd.to_numeric(
            frame.iloc[pos][adjusted_score_col], errors="coerce"
        ).to_numpy(dtype=np.float64)
        for local_idx, row_idx in enumerate(pos):
            if not np.isfinite(adjusted[local_idx]):
                continue
            adjusted_rank = store.lookup(
                strategy_id=strategy_id,
                side=str(side),
                calibrated_score=float(adjusted[local_idx]),
            ).policy_rank_pct
            raw_rank = (
                store.lookup(
                    strategy_id=strategy_id,
                    side=str(side),
                    calibrated_score=float(raw[local_idx]),
                ).policy_rank_pct
                if np.isfinite(raw[local_idx])
                else np.nan
            )
            if (
                np.isfinite(raw_rank)
                and raw_rank >= floor
                and adjusted_rank < floor + retained * max(raw_rank - floor, 0.0)
            ):
                adjusted_rank = floor + retained * max(raw_rank - floor, 0.0)
            out[row_idx] = np.float32(np.clip(adjusted_rank, 0.0, 1.0))
    return out


def _build_alternative_threshold_reference(
    *,
    policy: dict[str, Any],
    july_scored: pd.DataFrame,
    historical_path: Path,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    reference_path = Path(str(policy["reference_candidates_path"]))
    reference = pd.read_parquet(reference_path)
    reference["timestamp"] = pd.to_datetime(
        reference["timestamp"], utc=True, errors="coerce"
    )
    reference["archetype_policy_key"] = (
        reference["policy_archetype"]
        .astype(str)
        .str.replace(r"^(long|short)__", "", regex=True)
    )

    historical = pd.read_parquet(
        historical_path,
        columns=[
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "score_current_reference",
            "score_adjusted",
        ],
    ).rename(columns={"__ts__": "timestamp", "__symbol__": "symbol"})
    historical["timestamp"] = pd.to_datetime(
        historical["timestamp"], utc=True, errors="coerce"
    )
    july = july_scored[
        [
            "__ts__",
            "__symbol__",
            "side_name",
            "archetype_policy_key",
            "score_meta_base_soft_label",
            "score_shock_adjusted",
            "score_regime_alternative",
        ]
    ].rename(
        columns={
            "__ts__": "timestamp",
            "__symbol__": "symbol",
            "score_meta_base_soft_label": "score_current_reference",
            "score_shock_adjusted": "score_adjusted",
        }
    )
    base_keys = ["timestamp", "symbol", "side_name"]
    score_map = pd.concat([historical, july], ignore_index=True, sort=False)
    duplicate_scores = score_map.duplicated(base_keys, keep=False)
    if bool(duplicate_scores.any()):
        raise ValueError(
            "Alternative threshold score map is not unique by timestamp/symbol/side: "
            f"rows={int(duplicate_scores.sum())}"
        )
    merged = reference.merge(
        score_map,
        on=base_keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "__current_contract"),
    )
    current_arch = merged["archetype_policy_key__current_contract"].astype(str)
    valid_arch = merged["archetype_policy_key__current_contract"].notna()
    merged.loc[valid_arch, "archetype_policy_key"] = current_arch.loc[valid_arch]
    merged.loc[valid_arch, "policy_archetype"] = (
        merged.loc[valid_arch, "side_name"].astype(str)
        + "__"
        + current_arch.loc[valid_arch]
    )
    merged.loc[valid_arch, "local_side_archetype"] = merged.loc[
        valid_arch, "policy_archetype"
    ]

    old_raw = pd.to_numeric(merged["calibrated_score"], errors="coerce")
    old_adjusted = pd.to_numeric(merged["calibrated_score_regime_ev"], errors="coerce")
    new_raw = pd.to_numeric(merged["score_adjusted"], errors="coerce")
    direct = pd.to_numeric(merged.get("score_regime_alternative"), errors="coerce")
    # The deployed regime calibration is additive. On historical rows its exact
    # adjustment is preserved by applying the new-minus-old raw-score delta.
    alternative_adjusted = direct.where(
        direct.notna(),
        (old_adjusted + new_raw - old_raw).clip(0.0, 1.0),
    )
    matched = alternative_adjusted.notna()
    missing_rows = int((~matched).sum())
    if missing_rows / max(len(merged), 1) > 0.05:
        raise ValueError(
            "Alternative threshold reference alignment is below 95%: "
            f"missing={missing_rows} rows={len(merged)}"
        )
    merged = merged.loc[matched].copy()
    alternative_adjusted = alternative_adjusted.loc[matched]
    new_raw = new_raw.loc[matched]
    merged["calibrated_score_regime_ev"] = alternative_adjusted.astype(np.float32)
    merged["score_regime_calibrated"] = alternative_adjusted.astype(np.float32)
    merged["calibrated_score"] = new_raw.where(
        new_raw.notna(), alternative_adjusted
    ).astype(np.float32)
    merged = merged.drop(
        columns=[
            "archetype_policy_key",
            "archetype_policy_key__current_contract",
            "score_current_reference",
            "score_adjusted",
            "score_regime_alternative",
        ],
        errors="ignore",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False, compression="zstd")
    return output_path, {
        "reference_rows": int(len(merged)),
        "historical_scores": int(
            merged["timestamp"].lt(pd.Timestamp("2026-07-01", tz="UTC")).sum()
        ),
        "july_scores": int(
            merged["timestamp"].ge(pd.Timestamp("2026-07-01", tz="UTC")).sum()
        ),
        "alternative_score_coverage": float(matched.mean()),
        "excluded_unscored_reference_rows": missing_rows,
    }


def _apply_threshold_policy(
    frame: pd.DataFrame,
    *,
    policy: dict[str, Any],
    score_col: str,
    baseline_rank_col: str,
    prefix: str,
) -> pd.DataFrame:
    decisions: list[dict[str, Any]] = []
    policy_arch = _policy_archetype(frame)
    for idx, row in frame.iterrows():
        side = str(row["side_name"])
        decisions.append(
            {
                "_row_idx": int(idx),
                "signal_bar_ts": pd.Timestamp(row["__ts__"]).isoformat(),
                "symbol": str(row["__symbol__"]),
                "strategy_id": f"{side}_s52_meta_threshold_handoff",
                "side": side,
                "side_name": side,
                "policy_archetype": str(policy_arch.loc[idx]),
                "calibrated_score": float(row[score_col]),
                "policy_rank_pct": float(row[baseline_rank_col]),
            }
        )
    apply_threshold_basis_policy_to_decisions(decisions, policy=policy, store=None)
    payload = pd.DataFrame(
        {
            f"{prefix}_selected": [
                bool(item.get("threshold_basis_selected", False)) for item in decisions
            ],
            f"{prefix}_rank": [
                float(item.get("threshold_basis_rank_score", 0.0)) for item in decisions
            ],
            f"{prefix}_dynamic_ev_target": [
                item.get("threshold_basis_dynamic_ev_target", np.nan)
                for item in decisions
            ],
            f"{prefix}_dynamic_score_threshold": [
                item.get("threshold_basis_dynamic_score_threshold", np.nan)
                for item in decisions
            ],
            f"{prefix}_recent_reference_rows": [
                item.get("threshold_basis_recent_reference_rows", 0)
                for item in decisions
            ],
            f"{prefix}_baseline_activity_count": [
                item.get("threshold_basis_baseline_activity_count", 0)
                for item in decisions
            ],
        },
        index=frame.index,
    )
    return payload


def _breakdowns(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    selected = pd.concat(
        [
            frame.loc[frame["threshold_current_selected"]].assign(
                selector="current_policy_8d_ev_target"
            ),
            frame.loc[frame["threshold_alternative_selected"]].assign(
                selector="residual_policy_8d_ev_target"
            ),
            _metric_rows(frame, "current_reference", "historical_rank_current"),
            _metric_rows(
                frame, "residual_shock_alternative", "historical_rank_alternative"
            ),
            _metric_rows(frame, "current_reference_batch_top10", "batch_rank_current"),
            _metric_rows(frame, "residual_shock_batch_top10", "batch_rank_alternative"),
        ],
        ignore_index=True,
    )
    selected["day"] = selected["__ts__"].dt.strftime("%Y-%m-%d")
    selected["positive_ev"] = selected["ev_after_1pct"].gt(0.0).astype(np.float32)
    candidate = frame.copy(deep=False)
    candidate["day"] = candidate["__ts__"].dt.strftime("%Y-%m-%d")
    tables: dict[str, pd.DataFrame] = {}
    for name, groups in {
        "overall": ["selector"],
        "day": ["selector", "day"],
        "side": ["selector", "side_name"],
        "archetype": ["selector", "side_name", "archetype_policy_key"],
        "day_side_archetype": ["selector", "day", "side_name", "archetype_policy_key"],
    }.items():
        table = (
            selected.groupby(groups, sort=True, dropna=False)
            .agg(
                selected_rows=("__ts__", "size"),
                mean_ev_after_1pct=("ev_after_1pct", "mean"),
                sum_ev_after_1pct=("ev_after_1pct", "sum"),
                positive_ev_rate=("positive_ev", "mean"),
                clean_exec_precision=("clean_exec", "mean"),
                dirty_positive_rate=("dirty_positive", "mean"),
                first_touch_bad_mae_rate=("first_touch_bad_mae_1r", "mean"),
                full_path_bad_mae_rate=("full_path_bad_mae_1r", "mean"),
                timeout_rate=("timeout", "mean"),
            )
            .reset_index()
        )
        candidate_groups = [group for group in groups if group != "selector"]
        if candidate_groups:
            counts = (
                candidate.groupby(candidate_groups, sort=True, dropna=False)
                .size()
                .rename("candidate_rows")
                .reset_index()
            )
            table = table.merge(
                counts, on=candidate_groups, how="left", validate="many_to_one"
            )
        else:
            table["candidate_rows"] = int(len(candidate))
        day_divisor = (
            1 if "day" in candidate_groups else max(int(candidate["day"].nunique()), 1)
        )
        table["trades_per_day"] = table["selected_rows"] / day_divisor
        tables[name] = table
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--old-predictions", type=Path, default=DEFAULT_OLD_PREDICTIONS)
    parser.add_argument("--aegmm", type=Path, default=DEFAULT_AEGMM)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--train-reference", type=Path, default=DEFAULT_TRAIN_REFERENCE)
    parser.add_argument(
        "--threshold-policy", type=Path, default=DEFAULT_THRESHOLD_POLICY
    )
    parser.add_argument(
        "--regime-calibration", type=Path, default=DEFAULT_REGIME_CALIBRATION
    )
    parser.add_argument("--policy-rank-run-id", default=DEFAULT_POLICY_RANK_RUN_ID)
    parser.add_argument(
        "--historical-alternative", type=Path, default=DEFAULT_HISTORICAL_ALTERNATIVE
    )
    parser.add_argument(
        "--skip-reference-priors",
        action="store_true",
        help="Use reliability/prior columns already materialized in the handoff.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-07-09")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    bundle = joblib.load(args.bundle)
    regime_artifact = load_regime_ev_calibration(args.regime_calibration)
    required = list(
        dict.fromkeys(
            bundle.required_input_features()
            + regime_required_feature_columns(regime_artifact)
        )
    )
    frame = _read_july_handoff(args.handoff, required, start, end)
    handoff_rows = len(frame)
    frame = _merge_old_predictions(frame, args.old_predictions, start, end)
    frame = _merge_frozen_aegmm(frame, args.aegmm, start, end)
    frame, store_coverage = _append_store_features(frame, args.feature_root, required)
    if not args.skip_reference_priors:
        frame = _apply_reference_priors(frame, args.train_reference)

    old_rank = _fit_old_rank_reference(args.train_reference)
    old_rank_frame = pd.DataFrame(
        {
            "side_name": frame["side_name"],
            "score_meta_base_soft_label": frame["score_meta_base_soft_label"],
        },
        index=frame.index,
    )
    frame = pd.concat(
        [
            frame,
            pd.Series(
                old_rank.transform(old_rank_frame).to_numpy(dtype=np.float32),
                index=frame.index,
                name="historical_rank_current",
            ),
        ],
        axis=1,
        copy=False,
    )
    new = bundle.predict(frame)
    new = new.set_axis(frame.index)
    frame = pd.concat([frame, new], axis=1, copy=False)
    frame["historical_rank_alternative"] = frame["historical_rank"].astype(np.float32)
    frame = apply_regime_ev_calibration(
        frame,
        regime_artifact,
        source_score_col="score_meta_base_soft_label",
        adjusted_score_col="score_regime_current",
        copy=False,
    )
    frame = apply_regime_ev_calibration(
        frame,
        regime_artifact,
        source_score_col="score_shock_adjusted",
        adjusted_score_col="score_regime_alternative",
        copy=False,
    )
    rank_store = PolicyRankReferenceStore(
        data_root="data_perp", run_id=args.policy_rank_run_id
    )
    frame["policy_rank_current"] = _policy_rank_current(
        frame,
        store=rank_store,
        raw_score_col="score_meta_base_soft_label",
        adjusted_score_col="score_regime_current",
    )

    threshold_policy_current = load_threshold_basis_policy(args.threshold_policy)
    alternative_reference, alternative_reference_manifest = (
        _build_alternative_threshold_reference(
            policy=threshold_policy_current,
            july_scored=frame,
            historical_path=args.historical_alternative,
            output_path=args.output_dir
            / "threshold_basis_reference_candidates_alternative.parquet",
        )
    )
    threshold_policy_alternative = dict(threshold_policy_current)
    threshold_policy_alternative["reference_candidates_path"] = str(
        alternative_reference
    )
    threshold_policy_alternative["reference_columns"] = list(
        pd.read_parquet(alternative_reference, columns=None).columns
    )
    frame = pd.concat(
        [
            frame,
            _apply_threshold_policy(
                frame,
                policy=threshold_policy_current,
                score_col="score_regime_current",
                baseline_rank_col="policy_rank_current",
                prefix="threshold_current",
            ),
            _apply_threshold_policy(
                frame,
                policy=threshold_policy_alternative,
                score_col="score_regime_alternative",
                baseline_rank_col="policy_rank_current",
                prefix="threshold_alternative",
            ),
        ],
        axis=1,
        copy=False,
    )
    frame["batch_rank_current"] = (
        frame.groupby("__ts__", sort=False)["score_meta_base_soft_label"]
        .rank(method="average", pct=True)
        .astype(np.float32)
    )
    frame["batch_rank_alternative"] = (
        frame.groupby("__ts__", sort=False)["score_shock_adjusted"]
        .rank(method="average", pct=True)
        .astype(np.float32)
    )

    keep = list(
        dict.fromkeys(
            KEYS
            + OUTCOMES
            + [
                "score_meta_base_soft_label",
                "historical_rank_current",
                "batch_rank_current",
                "score_lifecycle_only",
                "score_residual_overlay",
                "score_shock_adjusted",
                "shock_composite_raw",
                "shock_composite_local",
                "hit_probability",
                "historical_rank_alternative",
                "batch_rank_alternative",
                "score_regime_current",
                "score_regime_alternative",
                "policy_rank_current",
                "threshold_current_selected",
                "threshold_current_rank",
                "threshold_current_dynamic_ev_target",
                "threshold_current_dynamic_score_threshold",
                "threshold_current_recent_reference_rows",
                "threshold_current_baseline_activity_count",
                "threshold_alternative_selected",
                "threshold_alternative_rank",
                "threshold_alternative_dynamic_ev_target",
                "threshold_alternative_dynamic_score_threshold",
                "threshold_alternative_recent_reference_rows",
                "threshold_alternative_baseline_activity_count",
            ]
        )
    )
    scored = frame[[name for name in keep if name in frame.columns]].copy()
    scored.to_parquet(
        args.output_dir / "july_oos_old_new_aligned_predictions.parquet",
        index=False,
        compression="zstd",
    )
    tables = _breakdowns(scored)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"metrics_{name}.csv", index=False)

    current = scored["batch_rank_current"].ge(0.90)
    alternative = scored["batch_rank_alternative"].ge(0.90)
    overlap = pd.DataFrame(
        {
            "selection_bucket": [
                "retained",
                "dropped_by_alternative",
                "added_by_alternative",
            ],
            "rows": [
                int((current & alternative).sum()),
                int((current & ~alternative).sum()),
                int((~current & alternative).sum()),
            ],
            "mean_ev_after_1pct": [
                float(scored.loc[current & alternative, "ev_after_1pct"].mean()),
                float(scored.loc[current & ~alternative, "ev_after_1pct"].mean()),
                float(scored.loc[~current & alternative, "ev_after_1pct"].mean()),
            ],
        }
    )
    overlap.to_csv(args.output_dir / "selection_reallocation.csv", index=False)
    required_coverage = {
        name: float(pd.to_numeric(frame[name], errors="coerce").notna().mean())
        if name in frame.columns
        else 0.0
        for name in required
    }
    manifest = {
        "schema": "july_oos_meta_residual_comparison_v1",
        "start": start,
        "end_exclusive": end,
        "handoff_rows": handoff_rows,
        "aligned_rows": len(scored),
        "days": int(scored["__ts__"].dt.floor("D").nunique()),
        "timestamps": int(scored["__ts__"].nunique()),
        "symbols": int(scored["__symbol__"].nunique()),
        "old_model_source": str(args.old_predictions),
        "new_bundle_source": str(args.bundle),
        "new_bundle_fit_through": str(bundle.fit_through),
        "old_rank_reference_source": str(args.train_reference),
        "reference_priors_recomputed": not bool(args.skip_reference_priors),
        "new_rank_reference_source": str(args.bundle),
        "threshold_policy_source": str(args.threshold_policy),
        "threshold_policy_id": threshold_policy_current.get("policy_id"),
        "threshold_policy_family": threshold_policy_current.get("family"),
        "threshold_policy_window_days": threshold_policy_current.get("window_days"),
        "threshold_policy_activity_basis": "current frozen policy-rank count per timestamp",
        "threshold_policy_alternative_reference": alternative_reference_manifest,
        "regime_calibration_source": str(args.regime_calibration),
        "regime_calibration_policy_id": regime_artifact.get("policy_id")
        or regime_artifact.get("artifact_id"),
        "feature_root": str(args.feature_root),
        "required_feature_count": len(required),
        "required_feature_mean_coverage": float(
            np.mean(list(required_coverage.values()))
        ),
        "required_features_below_90pct": sorted(
            name for name, rate in required_coverage.items() if rate < 0.90
        ),
        "store_feature_coverage": store_coverage,
        "leakage_contract": (
            "Both models and rank references are frozen through June. July outcomes are joined only after "
            "prediction and are used solely for reporting. No feature selection, HPO, AE/GMM fit, meta fit, "
            "threshold fit, or policy optimization is performed on July."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    joblib.dump(
        old_rank,
        args.output_dir / "current_reference_historical_rank.joblib",
        compress=3,
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    print(tables["overall"].to_string(index=False))
    print(tables["day"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
