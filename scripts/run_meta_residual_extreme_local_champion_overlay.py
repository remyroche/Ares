#!/usr/bin/env python3
"""Sparse side x archetype extreme-state overlay over the residual champion.

The parent score/rank is preserved for ordinary rows.  Train-only residual-state
semantics select a very small local feature set and define adverse/opportunity
tail intensities.  The overlay can only demote parent top-10 rows or promote
parent top-10--20 rows when a local state is extreme.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.residual_event_archetypes import _binned_mi  # noqa: E402
from extreme_price_movements.features_negative_residuals import (  # noqa: E402
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
)
from scripts.report_meta_residual_archetype_final import (  # noqa: E402
    _autocorr_components,
    _calendar_components_preselected,
)

KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
PARENT = (
    "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay_"
    "sparse_shock_composite"
)
FEATURES = [
    "resid_event_aegmm_expected_adverse_path_event",
    "resid_event_aegmm_expected_negative_residual_event",
    "resid_event_aegmm_expected_positive_residual_event",
    "resid_event_aegmm_expected_favorable_near_miss_event",
    "resid_event_aegmm_expected_ev_after_1pct",
    "resid_event_aegmm_expected_ev_timestamp_neutral_surprise",
    "resid_event_aegmm_expected_persistence_strength",
    "resid_event_aegmm_expected_directional_ev_divergence",
    "resid_event_aegmm_expected_bullish_tape_adverse_ev",
    "resid_event_aegmm_expected_timestamp_ev_sign_disagreement",
    "resid_event_aegmm_expected_persistent_subthreshold_damage",
    "resid_event_aegmm_expected_persistent_material_nontail",
    "resid_event_aegmm_gmm_entropy",
    "resid_event_aegmm_gmm_posterior_margin",
    "resid_event_aegmm_dae_reconstruction_error_zscore",
    "resid_event_aegmm_posterior_speed",
    "resid_event_aegmm_posterior_acceleration",
    "resid_event_aegmm_reconstruction_recent_max_24h",
    "resid_event_aegmm_reconstruction_recent_max_48h",
    "resid_event_aegmm_reconstruction_recent_max_96h",
    "resid_event_aegmm_hours_since_ood_spike_96h_norm",
]
STATE_NATIVE_FEATURES = tuple(FEATURES)
THRESHOLDS = (0.95, 0.96, 0.975, 0.99, 0.995)
ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
TOP_COUNTS = (1,)
FEATURE_SCREEN_TAIL = 0.95
CLUSTER_COLUMN = "resid_event_aegmm_gmm_cluster_id"
CLUSTER_CONDITIONED = False


def _local_group_columns(*, include_event: bool = False) -> list[str]:
    columns = ["side_name", "archetype_policy_key"]
    if CLUSTER_CONDITIONED:
        columns.append(CLUSTER_COLUMN)
    if include_event:
        columns.append("event")
    return columns


def _empirical_midrank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    percentile = np.full(len(values), 0.5, dtype=np.float32)
    finite = np.isfinite(values)
    left = np.searchsorted(reference, values[finite], side="left")
    right = np.searchsorted(reference, values[finite], side="right")
    percentile[finite] = (left + right) / (2.0 * float(len(reference)))
    return percentile


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _week_start(values: pd.Series) -> pd.Series:
    day = pd.to_datetime(values, utc=True, errors="coerce").dt.floor("D")
    return day - pd.to_timedelta(day.dt.weekday.to_numpy(), unit="D")


def _metric_row(frame: pd.DataFrame, mask: np.ndarray, selector: str) -> dict[str, Any]:
    selected = frame.loc[mask].copy()
    selected["week_start"] = _week_start(selected["__ts__"])
    selected["month"] = selected["__ts__"].dt.strftime("%Y-%m")
    weekly = selected.groupby("week_start", observed=True)["ev_after_1pct"].mean()
    monthly = selected.groupby("month", observed=True)["ev_after_1pct"].mean()
    return {
        "selector": selector,
        "selected_rows": int(len(selected)),
        "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
        "clean_exec_precision": float(selected["clean_exec"].mean()),
        "dirty_positive_rate": float(selected["dirty_positive"].mean()),
        "bad_mae_rate": float(selected["full_path_bad_mae_1r"].mean()),
        "timeout_rate": float(selected["timeout"].mean()),
        "worst_week_ev": float(weekly.min()),
        "worst_month_ev": float(monthly.min()),
        "positive_weeks": int(weekly.gt(0.0).sum()),
        "weeks": int(len(weekly)),
    }


def _breakdown(frame: pd.DataFrame, mask: np.ndarray, selector: str) -> pd.DataFrame:
    selected = frame.loc[mask].copy()
    selected["month"] = selected["__ts__"].dt.strftime("%Y-%m")
    selected["week_start"] = _week_start(selected["__ts__"])
    reports: list[pd.DataFrame] = []
    for scope, groups in (
        ("month", ["month"]),
        ("week", ["week_start"]),
        ("side_archetype", ["side_name", "archetype_policy_key"]),
        ("month_side_archetype", ["month", "side_name", "archetype_policy_key"]),
    ):
        report = (
            selected.groupby(groups, observed=True, dropna=False)
            .agg(
                selected_rows=("ev_after_1pct", "size"),
                mean_ev_after_1pct=("ev_after_1pct", "mean"),
                clean_exec_precision=("clean_exec", "mean"),
                dirty_positive_rate=("dirty_positive", "mean"),
                bad_mae_rate=("full_path_bad_mae_1r", "mean"),
                timeout_rate=("timeout", "mean"),
            )
            .reset_index()
        )
        report["scope"] = scope
        report["selector"] = selector
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def _load_joined(
    *,
    champion_path: Path,
    parent_eval_path: Path,
    state_path: Path | Sequence[Path],
    train_oof_predictions_dir: Path | None,
    train_oof_rank_cache: Path | None,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_end: pd.Timestamp,
    negative_residual_features: Path | None = None,
    strict_eval_state_coverage: bool = False,
    state_group_filter: tuple[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    champion_columns = [
        *KEYS,
        "historical_rank",
        "hit_probability",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    # With a cached OOF rank stream, train rows are assembled from the compact
    # residual state artifact below.  Avoid loading a second full champion
    # ledger unless it is genuinely required as the rank source or fallback
    # for the observable hit-probability column.
    champion_train: pd.DataFrame | None = None
    if train_oof_predictions_dir is None:
        champion_train = pd.read_parquet(champion_path, columns=champion_columns)
        champion_train["__ts__"] = pd.to_datetime(champion_train["__ts__"], utc=True)
        champion_train = champion_train.loc[
            champion_train["__ts__"].ge(train_start)
            & champion_train["__ts__"].lt(train_end)
        ]
    eval_schema = set(pq.read_schema(parent_eval_path).names)
    rank_column = next(
        (
            name
            for name in (
                "historical_rank_adjusted",
                "historical_rank",
                "score_base_residual_ev_rank_train_reference",
            )
            if name in eval_schema
        ),
        None,
    )
    probability_column = next(
        (
            name
            for name in ("hit_prob_adjusted", "hit_probability", "score")
            if name in eval_schema
        ),
        None,
    )
    if rank_column is None or probability_column is None:
        raise ValueError(
            "Parent meta predictions do not expose a supported frozen rank and "
            "observable soft-hit score."
        )
    eval_columns = [
        *KEYS,
        rank_column,
        probability_column,
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    parent_eval = pd.read_parquet(parent_eval_path, columns=eval_columns).rename(
        columns={
            rank_column: "historical_rank",
            probability_column: "hit_probability",
        }
    )
    parent_eval["__ts__"] = pd.to_datetime(parent_eval["__ts__"], utc=True)
    parent_eval = parent_eval.loc[
        parent_eval["__ts__"].ge(train_end) & parent_eval["__ts__"].lt(eval_end)
    ]
    if state_group_filter is not None:
        parent_eval = parent_eval.loc[
            parent_eval["side_name"].astype(str).eq(str(state_group_filter[0]))
            & parent_eval["archetype_policy_key"].astype(str).eq(
                str(state_group_filter[1])
            )
        ]
    state_columns = [
        *KEYS,
        "resid_event_class",
        CLUSTER_COLUMN,
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
        "hit_probability",
        *STATE_NATIVE_FEATURES,
    ]
    state_paths = [state_path] if isinstance(state_path, Path) else list(state_path)
    if not state_paths:
        raise ValueError("At least one residual-state artifact is required")
    state_parts: list[pd.DataFrame] = []
    state_filters = None
    if state_group_filter is not None:
        state_filters = [
            ("side_name", "=", str(state_group_filter[0])),
            ("archetype_policy_key", "=", str(state_group_filter[1])),
        ]
    for path in state_paths:
        available_state_columns = set(pq.read_schema(path).names)
        state_parts.append(
            pd.read_parquet(
                path,
                columns=[name for name in state_columns if name in available_state_columns],
                filters=state_filters,
            )
        )
    state = pd.concat(state_parts, ignore_index=True, copy=False)
    for name in STATE_NATIVE_FEATURES:
        if name not in state.columns:
            state[name] = np.float32(0.0)
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
    state = state.loc[state["__ts__"].ge(train_start) & state["__ts__"].lt(eval_end)]
    duplicate_state_keys = state.duplicated(KEYS, keep=False)
    if bool(duplicate_state_keys.any()):
        examples = state.loc[duplicate_state_keys, KEYS].head(5).to_dict("records")
        raise ValueError(
            "Residual-state artifacts overlap on decision rows; provide disjoint "
            f"OOS coverage. examples={examples}"
        )
    # Compact residual-state artifacts intentionally omit realized path fields
    # in some generations. Those fields are labels/reporting only, but the
    # chronological residual calendar still needs them. Recover them from the
    # frozen champion ledger by key rather than silently dropping the rows or
    # synthesizing a substitute clean-path definition.
    outcome_columns = [
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    missing_outcomes = [name for name in outcome_columns if name not in state.columns]
    if missing_outcomes:
        outcomes = pd.read_parquet(champion_path, columns=[*KEYS, *missing_outcomes])
        outcomes["__ts__"] = pd.to_datetime(outcomes["__ts__"], utc=True)
        outcomes = outcomes.loc[
            outcomes["__ts__"].ge(train_start) & outcomes["__ts__"].lt(eval_end)
        ].drop_duplicates(KEYS, keep="last")
        state = state.merge(
            outcomes,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    train_rank_contract = "frozen_champion_backcast_rank"
    train_rank_rows = int(len(champion_train)) if champion_train is not None else 0
    if train_oof_predictions_dir is None:
        assert champion_train is not None
        train = champion_train.merge(
            state.drop(
                columns=[
                    "ev_after_1pct",
                    "clean_exec",
                    "dirty_positive",
                    "full_path_bad_mae_1r",
                    "timeout",
                ],
                errors="ignore",
            ),
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
    else:
        if train_oof_rank_cache is not None and train_oof_rank_cache.exists():
            oof = pd.read_parquet(train_oof_rank_cache)
            oof["__ts__"] = pd.to_datetime(oof["__ts__"], utc=True)
            oof = oof.loc[
                oof["__ts__"].ge(train_start) & oof["__ts__"].lt(train_end)
            ]
        else:
            shard_paths = sorted(
                Path(path)
                for path in glob.glob(str(train_oof_predictions_dir / "*.parquet"))
            )
            if not shard_paths:
                raise FileNotFoundError(
                    f"No meta OOF prediction shards under {train_oof_predictions_dir}"
                )
            oof_parts: list[pd.DataFrame] = []
            oof_columns = [*KEYS, "score_meta_base_soft_label"]
            for shard_path in shard_paths:
                part = pd.read_parquet(shard_path, columns=oof_columns)
                part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True)
                part = part.loc[
                    part["__ts__"].ge(train_start) & part["__ts__"].lt(train_end)
                ]
                if not part.empty:
                    oof_parts.append(part)
            if not oof_parts:
                raise ValueError("Meta OOF shards have no rows in the training period")
            oof = pd.concat(oof_parts, ignore_index=True, copy=False)
            oof = oof.drop_duplicates(KEYS, keep="last")
            score = pd.to_numeric(oof["score_meta_base_soft_label"], errors="coerce")
            finite = score.notna()
            oof = oof.loc[finite].copy()
            oof["historical_rank"] = score.loc[finite].rank(
                method="average", pct=True
            ).astype(np.float32)
            if train_oof_rank_cache is not None:
                train_oof_rank_cache.parent.mkdir(parents=True, exist_ok=True)
                oof.loc[:, [*KEYS, "historical_rank"]].to_parquet(
                    train_oof_rank_cache, index=False, compression="zstd"
                )
        train = state.merge(
            oof.loc[:, [*KEYS, "historical_rank"]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        if "hit_probability" not in train:
            if champion_train is None:
                champion_train = pd.read_parquet(champion_path, columns=[*KEYS, "hit_probability"])
                champion_train["__ts__"] = pd.to_datetime(champion_train["__ts__"], utc=True)
                champion_train = champion_train.loc[
                    champion_train["__ts__"].ge(train_start)
                    & champion_train["__ts__"].lt(train_end)
                ]
            train = train.merge(
                champion_train.loc[:, [*KEYS, "hit_probability"]],
                on=KEYS,
                how="left",
                validate="one_to_one",
            )
        train_rank_contract = "global_train_meta_oof_score_percentile"
        train_rank_rows = int(len(oof))
    valid = parent_eval.merge(
        state.drop(
            columns=[
                "ev_after_1pct",
                "clean_exec",
                "dirty_positive",
                "full_path_bad_mae_1r",
                "timeout",
            ],
            errors="ignore",
        ),
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator="__state_join__",
    )
    missing_state = valid["__state_join__"].ne("both")
    missing_state_rows = int(missing_state.sum())
    missing_state_days = int(
        valid.loc[missing_state, "__ts__"].dt.floor("D").nunique()
    )
    valid = valid.drop(columns="__state_join__")
    if strict_eval_state_coverage and missing_state_rows:
        raise ValueError(
            "Residual-state OOS coverage is incomplete: "
            f"missing_rows={missing_state_rows:,}, "
            f"missing_days={missing_state_days}, parent_rows={len(parent_eval):,}. "
            "Materialize frozen train-only state assignments before scoring."
        )
    negative_manifest: dict[str, Any] | None = None
    if negative_residual_features is not None:
        market = pd.read_parquet(
            negative_residual_features,
            columns=NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
        )
        market.index = pd.to_datetime(market.index, utc=True, errors="coerce")
        market = market.loc[~market.index.duplicated(keep="last")].sort_index()
        market.index.name = "__ts__"
        market = market.reset_index()
        train = train.merge(market, on="__ts__", how="left", validate="many_to_one")
        valid = valid.merge(market, on="__ts__", how="left", validate="many_to_one")
        negative_manifest = {
            "path": str(negative_residual_features),
            "features": list(NEGATIVE_RESIDUAL_META_FEATURE_KEYS),
            "train_match_rate": float(
                train[NEGATIVE_RESIDUAL_META_FEATURE_KEYS].notna().any(axis=1).mean()
            ),
            "valid_match_rate": float(
                valid[NEGATIVE_RESIDUAL_META_FEATURE_KEYS].notna().any(axis=1).mean()
            ),
        }
    return train, valid, {
        "champion_train_rows": int(len(champion_train)) if champion_train is not None else 0,
        "champion_train_loaded": bool(champion_train is not None),
        "parent_eval_rows": int(len(parent_eval)),
        "parent_eval_rank_column": str(rank_column),
        "parent_eval_probability_column": str(probability_column),
        "state_rows": int(len(state)),
        "state_group_filter": (
            list(state_group_filter) if state_group_filter is not None else None
        ),
        "train_rank_contract": train_rank_contract,
        "train_rank_rows": train_rank_rows,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "eval_state_missing_rows": missing_state_rows,
        "eval_state_missing_days": missing_state_days,
        "eval_state_match_rate": float(1.0 - missing_state_rows / max(len(valid), 1)),
        "negative_residual_features": negative_manifest,
    }


def _feature_catalog(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top20 = pd.to_numeric(train["historical_rank"], errors="coerce").ge(0.80)
    adverse = train["resid_event_class"].astype(str).eq("adverse_path_event")
    positive = train["resid_event_class"].astype(str).isin(
        ["positive_residual_event", "favorable_near_miss_event"]
    )
    for local_key, group in train.loc[top20].groupby(
        _local_group_columns(), observed=True, sort=True
    ):
        local_key = local_key if isinstance(local_key, tuple) else (local_key,)
        side, archetype = local_key[:2]
        cluster_id = local_key[2] if CLUSTER_CONDITIONED else None
        idx = group.index
        y_adverse = adverse.loc[idx].to_numpy(dtype=np.float32)
        y_positive = positive.loc[idx].to_numpy(dtype=np.float32)
        for feature in FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(
                dtype=np.float32
            )
            finite = np.isfinite(values)
            if int(finite.sum()) < 200:
                continue
            for event, target in (("adverse", y_adverse), ("positive", y_positive)):
                if np.unique(target[finite]).size < 2:
                    continue
                mi = _binned_mi(values[finite], target[finite], 8)
                event_mean = float(np.nanmean(values[target > 0.5]))
                other_mean = float(np.nanmean(values[target <= 0.5]))
                prevalence = float(np.mean(target[finite]))
                reference = np.sort(values[finite])
                best_tail: dict[str, float] | None = None
                for direction in (-1.0, 1.0):
                    directed = np.float32(direction) * values
                    directed_reference = np.sort(directed[finite])
                    percentile = _empirical_midrank(directed, directed_reference)
                    tail = finite & (percentile >= FEATURE_SCREEN_TAIL)
                    tail_rows = int(tail.sum())
                    if tail_rows < 30:
                        continue
                    tail_rate = float(np.mean(target[tail]))
                    tail_delta = tail_rate - prevalence
                    payload = {
                        "direction": direction,
                        "tail_rows": float(tail_rows),
                        "tail_rate": tail_rate,
                        "tail_delta": tail_delta,
                        "tail_lift": tail_rate / max(prevalence, 1e-6),
                    }
                    if best_tail is None or (
                        payload["tail_delta"], payload["tail_rate"]
                    ) > (best_tail["tail_delta"], best_tail["tail_rate"]):
                        best_tail = payload
                if best_tail is None or best_tail["tail_delta"] <= 0.0:
                    continue
                rows.append(
                    {
                        "side_name": str(side),
                        "archetype_policy_key": str(archetype),
                        **(
                            {CLUSTER_COLUMN: int(cluster_id)}
                            if CLUSTER_CONDITIONED
                            else {}
                        ),
                        "event": event,
                        "feature": feature,
                        "mi": float(mi),
                        "direction": float(best_tail["direction"]),
                        "event_mean": event_mean,
                        "other_mean": other_mean,
                        "event_prevalence": prevalence,
                        "extreme_screen_quantile": FEATURE_SCREEN_TAIL,
                        "extreme_tail_rows": int(best_tail["tail_rows"]),
                        "extreme_tail_event_rate": float(best_tail["tail_rate"]),
                        "extreme_tail_event_delta": float(best_tail["tail_delta"]),
                        "extreme_tail_event_lift": float(best_tail["tail_lift"]),
                        "feature_unique_values": int(np.unique(reference).size),
                        "rows": int(finite.sum()),
                    }
                )
    sort_columns = [
        "side_name",
        "archetype_policy_key",
        *([CLUSTER_COLUMN] if CLUSTER_CONDITIONED else []),
        "event",
        "extreme_tail_event_delta",
        "mi",
    ]
    return pd.DataFrame(rows).sort_values(
        sort_columns,
        ascending=[True] * (len(sort_columns) - 2) + [False, False],
        kind="stable",
    )


def _fit_references(
    train: pd.DataFrame, catalog: pd.DataFrame, top_count: int
) -> dict[tuple[Any, ...], list[tuple[str, float, np.ndarray]]]:
    references: dict[tuple[Any, ...], list[tuple[str, float, np.ndarray]]] = {}
    for local_key, group in catalog.groupby(
        _local_group_columns(include_event=True), observed=True, sort=True
    ):
        local_key = local_key if isinstance(local_key, tuple) else (local_key,)
        side, archetype = local_key[:2]
        cluster_id = local_key[2] if CLUSTER_CONDITIONED else None
        event = local_key[-1]
        selected = group.head(int(top_count))
        local = train.loc[
            train["side_name"].astype(str).eq(str(side))
            & train["archetype_policy_key"].astype(str).eq(str(archetype))
        ]
        if CLUSTER_CONDITIONED:
            local = local.loc[
                pd.to_numeric(local[CLUSTER_COLUMN], errors="coerce").eq(
                    float(cluster_id)
                )
            ]
        payload: list[tuple[str, float, np.ndarray]] = []
        for row in selected.itertuples(index=False):
            values = float(row.direction) * pd.to_numeric(
                local[row.feature], errors="coerce"
            ).to_numpy(dtype=np.float32)
            reference = np.sort(values[np.isfinite(values)])
            if len(reference) >= 200:
                payload.append((str(row.feature), float(row.direction), reference))
        if payload:
            key: tuple[Any, ...] = (str(side), str(archetype))
            if CLUSTER_CONDITIONED:
                key += (int(cluster_id),)
            references[key + (str(event),)] = payload
    return references


def _composite(
    frame: pd.DataFrame,
    references: dict[tuple[Any, ...], list[tuple[str, float, np.ndarray]]],
    event: str,
) -> np.ndarray:
    output = np.full(len(frame), 0.5, dtype=np.float32)
    side = frame["side_name"].astype(str).to_numpy()
    archetype = frame["archetype_policy_key"].astype(str).to_numpy()
    cluster_source = (
        frame[CLUSTER_COLUMN]
        if CLUSTER_COLUMN in frame
        else pd.Series(np.nan, index=frame.index, dtype=np.float32)
    )
    cluster = pd.to_numeric(cluster_source, errors="coerce").to_numpy()
    for local_key, payload in references.items():
        side_key, archetype_key = local_key[:2]
        cluster_key = local_key[2] if CLUSTER_CONDITIONED else None
        event_key = local_key[-1]
        if event_key != event:
            continue
        mask = (side == side_key) & (archetype == archetype_key)
        if CLUSTER_CONDITIONED:
            mask &= cluster == float(cluster_key)
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        components: list[np.ndarray] = []
        for feature, direction, reference in payload:
            values = direction * pd.to_numeric(
                frame.iloc[idx][feature], errors="coerce"
            ).to_numpy(dtype=np.float32)
            pct = np.full(len(idx), 0.5, dtype=np.float32)
            finite = np.isfinite(values)
            # Midranks prevent a large point mass (commonly zero) from becoming
            # an entire nominal 95-99.5% tail through right-tie assignment.
            pct = _empirical_midrank(values, reference)
            components.append(pct)
        output[idx] = np.mean(np.column_stack(components), axis=1).astype(np.float32)
    return output


def _intensity(composite: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip(
        (composite - float(threshold)) / max(1.0 - float(threshold), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)


def _adjust_rank(
    parent_rank: np.ndarray,
    adverse: np.ndarray,
    positive: np.ndarray,
    *,
    threshold: float,
    alpha_down: float,
    alpha_up: float,
) -> np.ndarray:
    adjusted = parent_rank.astype(np.float32, copy=True)
    adverse_intensity = _intensity(adverse, threshold)
    positive_intensity = _intensity(positive, threshold)
    top10 = parent_rank >= 0.90
    near = (parent_rank >= 0.80) & (parent_rank < 0.90)
    adjusted[top10] -= np.float32(alpha_down) * adverse_intensity[top10]
    adjusted[near] += np.float32(alpha_up) * positive_intensity[near]
    return np.clip(adjusted, 0.0, 1.0)


def _search(
    train: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    parent_rank = pd.to_numeric(train["historical_rank"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    parent_selected = parent_rank >= 0.90
    base_count = max(int(parent_selected.sum()), 1)
    month_code, month_labels = pd.factorize(
        train["__ts__"].dt.strftime("%Y-%m"), sort=True
    )
    month_code = month_code.astype(np.int16, copy=False)
    month_count = int(len(month_labels))
    ev = pd.to_numeric(train["ev_after_1pct"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    clean = pd.to_numeric(train["clean_exec"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for top_count in TOP_COUNTS:
        references = _fit_references(train, catalog, top_count)
        adverse = _composite(train, references, "adverse")
        positive = _composite(train, references, "positive")
        for threshold in THRESHOLDS:
            for alpha_down in ALPHAS:
                for alpha_up in ALPHAS:
                    adjusted = _adjust_rank(
                        parent_rank,
                        adverse,
                        positive,
                        threshold=threshold,
                        alpha_down=alpha_down,
                        alpha_up=alpha_up,
                    )
                    selected = adjusted >= 0.90
                    finite_ev = np.isfinite(ev)
                    finite_clean = np.isfinite(clean)
                    selected_ev = selected & finite_ev
                    selected_clean = selected & finite_clean
                    counts = np.bincount(
                        month_code,
                        weights=selected_ev.astype(np.float32, copy=False),
                        minlength=month_count,
                    )
                    clean_counts = np.bincount(
                        month_code,
                        weights=selected_clean.astype(np.float32, copy=False),
                        minlength=month_count,
                    )
                    ev_sum = np.bincount(
                        month_code,
                        weights=np.where(selected_ev, ev, 0.0),
                        minlength=month_count,
                    )
                    clean_sum = np.bincount(
                        month_code,
                        weights=np.where(selected_clean, clean, 0.0),
                        minlength=month_count,
                    )
                    valid_month = (counts > 0) & (clean_counts > 0)
                    values = ev_sum[valid_month] / counts[valid_month]
                    clean_values = clean_sum[valid_month] / clean_counts[valid_month]
                    count_ratio = float(selected.sum()) / float(base_count)
                    activity_penalty = abs(np.log(max(count_ratio, 1e-6)))
                    objective = (
                        float(np.mean(values))
                        - 0.5 * float(np.std(values))
                        + 0.25 * float(np.min(values))
                        + 0.002 * float(np.mean(clean_values))
                        - 0.01 * activity_penalty
                    )
                    record = {
                        "top_feature_count": int(top_count),
                        "threshold": float(threshold),
                        "alpha_down": float(alpha_down),
                        "alpha_up": float(alpha_up),
                        "selected_rows": int(selected.sum()),
                        "activity_ratio": count_ratio,
                        "mean_month_ev": float(np.mean(values)),
                        "std_month_ev": float(np.std(values)),
                        "worst_month_ev": float(np.min(values)),
                        "clean_exec_precision": float(np.mean(clean_values)),
                        "objective": objective,
                    }
                    rows.append(record)
                    if best is None or objective > float(best["objective"]):
                        best = record
    if best is None:
        raise RuntimeError("Extreme local overlay search produced no valid arm")
    search = pd.DataFrame(rows)
    strict = search.loc[
        search["threshold"].ge(0.95)
        & search["alpha_up"].eq(0.0)
        & search["activity_ratio"].between(0.95, 1.05)
    ].sort_values(
        ["objective", "top_feature_count", "threshold"],
        ascending=[False, True, False],
        kind="stable",
    )
    if strict.empty:
        raise RuntimeError("No strict extreme/activity-matched overlay arm")
    return search, best, strict.iloc[0].to_dict()


def _rank_for_params(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    catalog: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    references = _fit_references(train, catalog, int(params["top_feature_count"]))
    adverse = _composite(valid, references, "adverse")
    positive = _composite(valid, references, "positive")
    parent_rank = pd.to_numeric(valid["historical_rank"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    adjusted = _adjust_rank(
        parent_rank,
        adverse,
        positive,
        threshold=float(params["threshold"]),
        alpha_down=float(params["alpha_down"]),
        alpha_up=float(params["alpha_up"]),
    )
    return adjusted, adverse, positive


def _local_tail_diagnostics(
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Record the exact train-only threshold fitted in each local cell."""

    top_count = int(params["top_feature_count"])
    threshold = float(params["threshold"])
    references = _fit_references(train, catalog, top_count)
    rows: list[dict[str, Any]] = []
    for local_key, payload in sorted(references.items()):
        side, archetype = local_key[:2]
        cluster_id = local_key[2] if CLUSTER_CONDITIONED else None
        event = local_key[-1]
        local = train.loc[
            train["side_name"].astype(str).eq(side)
            & train["archetype_policy_key"].astype(str).eq(archetype)
        ]
        if CLUSTER_CONDITIONED:
            local = local.loc[
                pd.to_numeric(local[CLUSTER_COLUMN], errors="coerce").eq(
                    float(cluster_id)
                )
            ]
        for feature, direction, reference in payload:
            cutoff = float(np.quantile(reference, threshold))
            values = direction * pd.to_numeric(
                local[feature], errors="coerce"
            ).to_numpy(dtype=np.float32)
            finite = np.isfinite(values)
            left = np.searchsorted(reference, values[finite], side="left")
            right = np.searchsorted(reference, values[finite], side="right")
            percentile = _empirical_midrank(values, reference)[finite]
            tail = percentile >= threshold
            rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "event": event,
                    **(
                        {CLUSTER_COLUMN: int(cluster_id)}
                        if CLUSTER_CONDITIONED
                        else {}
                    ),
                    "feature": feature,
                    "direction": float(direction),
                    "tail_quantile": threshold,
                    "local_reference_rows": int(len(reference)),
                    "local_tail_cutoff": cutoff,
                    "local_tail_rows": int(np.sum(tail)),
                    "local_tail_fraction": float(np.mean(tail)),
                    "largest_tie_fraction": float(
                        np.max(right - left) / float(len(reference))
                    ),
                    "percentile_method": "empirical_midrank",
                    "scope": "side_x_archetype",
                }
            )
    report = pd.DataFrame(rows)
    if report.empty or not report["scope"].eq("side_x_archetype").all():
        raise RuntimeError("Local tail diagnostics did not preserve cell locality")
    return report


def _autocorrelation_report(
    frame: pd.DataFrame, masks: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendars: list[pd.DataFrame] = []
    for selector, mask in masks.items():
        selected = frame.loc[mask].copy()
        calendars.append(
            _calendar_components_preselected(
                selected,
                prob_col="hit_probability",
                arm=selector,
            )
        )
    calendar = pd.concat(calendars, ignore_index=True)
    return calendar, _autocorr_components(calendar)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--champion-ledger",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "champion_frozen_single_source_202501_20260710/"
            "frozen_champion_single_source_ledger.parquet"
        ),
    )
    parser.add_argument(
        "--train-oof-predictions-dir",
        type=Path,
        default=Path(
            "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_"
            "mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_"
            "payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_"
            "lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/"
            "best_full_oos_fixedfs_streamed_v1/prediction_shards"
        ),
    )
    parser.add_argument(
        "--train-oof-rank-cache",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_"
            "compactlocal_market_20260712_v3/meta_oof_global_rank_"
            "202504_202603.parquet"
        ),
    )
    parser.add_argument(
        "--state-artifact",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_"
            "compactlocal_market_20260712_v3/oos_residual_event_states.parquet"
        ),
    )
    parser.add_argument(
        "--parent-eval-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_"
            "globaloverlay_sparse_shock_composite/"
            "oos_predictions_historical_rank.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_20260712_v1"
        ),
    )
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--train-end", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument(
        "--min-valid-rows",
        type=int,
        default=50_000,
        help=(
            "Minimum joined OOS rows required for the requested evaluation "
            "window. Lower this explicitly for a bounded untouched holdout."
        ),
    )
    parser.add_argument(
        "--negative-residual-features",
        type=Path,
        default=None,
        help="Optional hardened market-context parquet indexed by decision timestamp.",
    )
    parser.add_argument(
        "--cluster-conditioned",
        action="store_true",
        help="Fit feature tails independently inside frozen residual GMM clusters.",
    )
    args = parser.parse_args()

    global CLUSTER_CONDITIONED
    CLUSTER_CONDITIONED = bool(args.cluster_conditioned)
    if args.negative_residual_features is not None:
        FEATURES.extend(
            name for name in NEGATIVE_RESIDUAL_META_FEATURE_KEYS if name not in FEATURES
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_start = pd.Timestamp(args.train_start, tz="UTC")
    train_end = pd.Timestamp(args.train_end, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    train, valid, coverage = _load_joined(
        champion_path=args.champion_ledger,
        parent_eval_path=args.parent_eval_predictions,
        state_path=args.state_artifact,
        train_oof_predictions_dir=args.train_oof_predictions_dir,
        train_oof_rank_cache=args.train_oof_rank_cache,
        train_start=train_start,
        train_end=train_end,
        eval_end=eval_end,
        negative_residual_features=args.negative_residual_features,
    )
    if len(train) < 100_000 or len(valid) < int(args.min_valid_rows):
        raise ValueError(f"Insufficient joined support: train={len(train)} valid={len(valid)}")
    catalog = _feature_catalog(train)
    search, best, strict_best = _search(train, catalog)
    adjusted, adverse, positive = _rank_for_params(train, valid, catalog, best)
    strict_adjusted, strict_adverse, strict_positive = _rank_for_params(
        train, valid, catalog, strict_best
    )
    parent_rank = pd.to_numeric(valid["historical_rank"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    valid["resid_extreme_adverse_composite"] = adverse
    valid["resid_extreme_positive_composite"] = positive
    valid["historical_rank_extreme_local"] = adjusted
    valid["resid_strict_extreme_adverse_composite"] = strict_adverse
    valid["resid_strict_extreme_positive_composite"] = strict_positive
    valid["historical_rank_strict_extreme_local"] = strict_adjusted
    valid["selected_parent"] = parent_rank >= 0.90
    valid["selected_extreme_local"] = adjusted >= 0.90
    valid["selected_strict_extreme_local"] = strict_adjusted >= 0.90
    valid.to_parquet(
        args.output_dir / "oos_predictions.parquet", index=False, compression="zstd"
    )
    search.to_csv(args.output_dir / "train_search.csv", index=False)
    catalog.to_csv(args.output_dir / "local_feature_catalog.csv", index=False)
    selected_catalog = (
        catalog.groupby(
            _local_group_columns(include_event=True),
            observed=True,
            sort=True,
        )
        .head(int(best["top_feature_count"]))
        .copy()
    )
    selected_catalog.to_csv(args.output_dir / "selected_local_features.csv", index=False)
    strict_selected_catalog = (
        catalog.groupby(
            _local_group_columns(include_event=True),
            observed=True,
            sort=True,
        )
        .head(int(strict_best["top_feature_count"]))
        .copy()
    )
    strict_selected_catalog.to_csv(
        args.output_dir / "selected_local_features_strict.csv", index=False
    )
    strict_tail_diagnostics = _local_tail_diagnostics(
        train, strict_selected_catalog, strict_best
    )
    strict_tail_diagnostics.to_csv(
        args.output_dir / "strict_local_tail_thresholds.csv", index=False
    )

    parent_mask = parent_rank >= 0.90
    adjusted_mask = adjusted >= 0.90
    strict_adjusted_mask = strict_adjusted >= 0.90
    parent_selector = PARENT
    adjusted_selector = f"{PARENT}_extreme_local"
    strict_selector = f"{PARENT}_strict_extreme_local"
    summary = pd.DataFrame(
        [
            _metric_row(valid, parent_mask, parent_selector),
            _metric_row(valid, adjusted_mask, adjusted_selector),
            _metric_row(valid, strict_adjusted_mask, strict_selector),
        ]
    )
    calendar, autocorrelation = _autocorrelation_report(
        valid,
        {
            parent_selector: parent_mask,
            adjusted_selector: adjusted_mask,
            strict_selector: strict_adjusted_mask,
        },
    )
    calendar.to_csv(args.output_dir / "hit_surprise_calendar.csv", index=False)
    autocorrelation.to_csv(
        args.output_dir / "hit_surprise_autocorrelation.csv", index=False
    )
    mean_abs_ac = (
        autocorrelation.groupby("arm", observed=True)[
            "signed_surprise_autocorr_lag1"
        ]
        .apply(lambda values: float(values.abs().mean()))
        .rename("mean_abs_signed_surprise_autocorr_lag1")
    )
    summary = summary.merge(mean_abs_ac, left_on="selector", right_index=True, how="left")
    overlap = pd.DataFrame(
        [
            {
                "selector": adjusted_selector,
                "parent_rows": int(parent_mask.sum()),
                "selected_rows": int(adjusted_mask.sum()),
                "kept_rows": int((parent_mask & adjusted_mask).sum()),
                "dropped_rows": int((parent_mask & ~adjusted_mask).sum()),
                "added_rows": int((~parent_mask & adjusted_mask).sum()),
            },
            {
                "selector": strict_selector,
                "parent_rows": int(parent_mask.sum()),
                "selected_rows": int(strict_adjusted_mask.sum()),
                "kept_rows": int((parent_mask & strict_adjusted_mask).sum()),
                "dropped_rows": int((parent_mask & ~strict_adjusted_mask).sum()),
                "added_rows": int((~parent_mask & strict_adjusted_mask).sum()),
            },
        ]
    )
    overlap.to_csv(args.output_dir / "selection_overlap.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat(
        [
            _breakdown(valid, parent_mask, parent_selector),
            _breakdown(valid, adjusted_mask, adjusted_selector),
            _breakdown(valid, strict_adjusted_mask, strict_selector),
        ],
        ignore_index=True,
    ).to_csv(args.output_dir / "breakdowns.csv", index=False)
    forced_rows: list[dict[str, Any]] = []
    forced_breakdowns: list[pd.DataFrame] = []
    forced_masks: dict[str, np.ndarray] = {}
    for tail_threshold in THRESHOLDS:
        candidates = search.loc[
            search["threshold"].eq(float(tail_threshold))
            & search["alpha_down"].gt(0.0)
            & search["alpha_up"].eq(0.0)
        ].sort_values(
            ["objective", "activity_ratio", "alpha_down"],
            ascending=[False, False, True],
            kind="stable",
        )
        if candidates.empty:
            continue
        candidate_params = candidates.iloc[0].to_dict()
        forced_rank, _, _ = _rank_for_params(
            train, valid, catalog, candidate_params
        )
        forced_mask = forced_rank >= 0.90
        selector = f"{PARENT}_forced_local_tail_{tail_threshold:.3f}"
        metric = _metric_row(valid, forced_mask, selector)
        metric.update(
            {
                f"train_{name}": candidate_params[name]
                for name in (
                    "threshold",
                    "alpha_down",
                    "activity_ratio",
                    "objective",
                )
            }
        )
        forced_rows.append(metric)
        forced_masks[selector] = forced_mask
        forced_breakdowns.append(_breakdown(valid, forced_mask, selector))
    if forced_rows:
        forced_summary = pd.DataFrame(forced_rows)
        _, forced_autocorrelation = _autocorrelation_report(valid, forced_masks)
        forced_mean_abs_ac = (
            forced_autocorrelation.groupby("arm", observed=True)[
                "signed_surprise_autocorr_lag1"
            ]
            .apply(lambda values: float(values.abs().mean()))
            .rename("mean_abs_signed_surprise_autocorr_lag1")
        )
        forced_summary = forced_summary.merge(
            forced_mean_abs_ac, left_on="selector", right_index=True, how="left"
        )
        forced_summary.to_csv(
            args.output_dir / "forced_nonzero_tail_summary.csv", index=False
        )
        pd.concat(forced_breakdowns, ignore_index=True).to_csv(
            args.output_dir / "forced_nonzero_tail_breakdowns.csv", index=False
        )
        forced_autocorrelation.to_csv(
            args.output_dir / "forced_nonzero_tail_autocorrelation.csv", index=False
        )
    _write_json(
        args.output_dir / "manifest.json",
        {
            "schema": "meta_residual_extreme_local_champion_overlay_v1",
            "parent": PARENT,
            "champion_ledger": str(args.champion_ledger),
            "train_oof_predictions_dir": str(args.train_oof_predictions_dir),
            "train_oof_rank_cache": str(args.train_oof_rank_cache),
            "parent_eval_predictions": str(args.parent_eval_predictions),
            "state_artifact": str(args.state_artifact),
            "negative_residual_features": (
                str(args.negative_residual_features)
                if args.negative_residual_features is not None
                else None
            ),
            "cluster_conditioned": bool(CLUSTER_CONDITIONED),
            "coverage": coverage,
            "train_start": train_start,
            "train_end": train_end,
            "eval_end": eval_end,
            "best": best,
            "strict_best": strict_best,
            "candidate_features": FEATURES,
            "selected_local_features": selected_catalog.to_dict("records"),
            "selected_local_features_strict": strict_selected_catalog.to_dict(
                "records"
            ),
            "leakage_contract": (
                "Feature selection, directions, references, and overlay parameters use "
                "only chronological meta OOF predictions and residual-state rows from "
                "2025-04 through 2026-03. April-June 2026 is OOS. All inference "
                "features and cluster assignments are frozen pre-entry outputs."
            ),
            "scope_contract": (
                "Parent rank is unchanged outside extreme local states. Every feature "
                "percentile and tail cutoff is fitted independently inside its side x "
                "archetype x frozen residual-GMM-cluster cell when cluster conditioning "
                "is enabled."
            ),
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
