#!/usr/bin/env python3
"""Run or report the fixed-window base side/target ablation matrix.

This is deliberately an orchestration layer.  It does not fit a model itself,
does not import the base training runner, and never starts a subprocess on
import.  The matrix holds labels, cost contract, train/OOS split, and frozen
AE/GMM representation fixed while it varies only the requested model-side and
feature-selection contracts:

* ``A0``: shared L2 base model with global economic MDA;
* ``A``: long/short L2 models with archetype-aware pre-screening followed by
  independent side MDA;
* ``B``: Pack-B-style correlation-first selector, run with independent side
  models only when A beats A0 on common-row top-10 net EV;
* ``C``: hierarchical train-only target/geometry/sample-weight HPO using the
  selected A/B payload and then one untouched Apr-Jun scoring pass.

Pack B is a selector-only experiment.  It always retains the corrected base
soft target and base L2 LightGBM recipe.  It must never inherit the meta
residual target or meta-model parameters from
``run_meta_v9_ev_mapped_side_residual_ablation.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_materialized_trailing_label_topk_lgbm_hpo.py"
STAGE_C_RUNNER = ROOT / "scripts" / "run_base_side_target_geometry_hpo.py"
DEFAULT_L2_PARAMS = ROOT / "docs" / "promoted_s59_singlecycle_base_params.json"
TOP_FRACS = (0.01, 0.05, 0.10, 0.20, 0.30)
KEYS = ("__ts__", "__symbol__", "side_name")
DEFAULT_TRAIN_START = "2025-04-01T00:00:00Z"
DEFAULT_TRAIN_END = "2026-04-01T00:00:00Z"
DEFAULT_OOS_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_PURGE_HOURS = 25.0


@dataclass(frozen=True)
class MatrixConfig:
    labels_path: Path
    feature_dir: Path
    feature_list_csv: Path
    output_dir: Path
    fixed_params_json: Path
    oos_months: tuple[str, ...] = DEFAULT_OOS_MONTHS
    train_window_days: int = 365
    purge_hours: float = DEFAULT_PURGE_HOURS
    feature_selection_sample_rows: int = 300_000
    max_train_rows: int = 0
    seed: int = 42
    python: str = sys.executable
    frozen_ae_gmm_state: Path | None = None
    frozen_ae_gmm_output_sidecar: Path | None = None


def _iso_months(value: str | Iterable[str]) -> tuple[str, ...]:
    values = value.split(",") if isinstance(value, str) else value
    months = tuple(str(item).strip() for item in values if str(item).strip())
    periods = [pd.Period(month, freq="M") for month in months]
    if len(periods) != 3 or periods != [periods[0] + idx for idx in range(3)]:
        raise ValueError(
            "The fixed ablation requires exactly three contiguous OOS months; "
            f"got={months}"
        )
    return months


def _side_name(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().str.lower()
    return pd.Series(
        np.where(
            text.str.contains("short", regex=False).to_numpy()
            | numeric.lt(0.0).fillna(False).to_numpy(),
            "short",
            "long",
        ),
        index=values.index,
        dtype="string",
    )


def _state_path_for_a0(config: MatrixConfig) -> Path:
    return (
        config.output_dir
        / "A0_shared_l2"
        / "_feature_selection_phase"
        / "ae_gmm_states"
        / "cycle__global_state.pkl"
    )


def _command_base(
    config: MatrixConfig,
    *,
    arm_name: str,
    selector: str,
    model_side_scope: str,
    frozen_ae_gmm_state: Path | None,
    fit_reference_ae_gmm: bool,
) -> list[str]:
    """Build one fixed-window base command without executing it."""
    if model_side_scope not in {"shared", "per_side"}:
        raise ValueError(f"Unsupported model_side_scope={model_side_scope}")
    if fit_reference_ae_gmm and frozen_ae_gmm_state is not None:
        raise ValueError("A reference AE/GMM fit and a frozen state are mutually exclusive")
    command = [
        str(config.python),
        str(RUNNER),
        "--labels-path",
        str(config.labels_path),
        "--feature-dir",
        str(config.feature_dir),
        "--feature-list-csv",
        str(config.feature_list_csv),
        "--output-dir",
        str(config.output_dir / arm_name),
        "--months",
        ",".join(config.oos_months),
        "--single-fit-oos-window",
        "--train-window-days",
        str(int(config.train_window_days)),
        "--label-path-purge-hours",
        str(float(config.purge_hours)),
        "--model-side-scope",
        model_side_scope,
        "--feature-selection-method",
        selector,
        "--feature-selection-target-mode",
        "target_soft",
        "--fresh-feature-selection",
        "--fixed-params-json",
        str(config.fixed_params_json),
        "--n-trials",
        "0",
        "--max-train-rows",
        str(int(config.max_train_rows)),
        "--feature-selection-sample-rows",
        str(int(config.feature_selection_sample_rows)),
        "--seed",
        str(int(config.seed)),
        "--no-save-final-model",
    ]
    if frozen_ae_gmm_state is not None:
        command.extend(["--fixed-ae-gmm-state-pkl", str(frozen_ae_gmm_state)])
        if config.frozen_ae_gmm_output_sidecar is not None:
            command.extend(
                [
                    "--frozen-ae-gmm-output-sidecar",
                    str(config.frozen_ae_gmm_output_sidecar),
                ]
            )
    elif fit_reference_ae_gmm:
        command.append("--refit-cycle-ae-gmm")
    else:
        raise ValueError("Every arm must receive the shared AE/GMM state")
    return command


def build_arm_commands(config: MatrixConfig) -> dict[str, list[str]]:
    """Build A0/A commands and both possible B commands.

    ``B`` cannot be chosen before the A0/A comparison exists, so the returned
    dictionary makes both branches visible in the provenance manifest.
    """
    _validate_config(config)
    state = config.frozen_ae_gmm_state
    a0_state = state or _state_path_for_a0(config)
    return {
        "A0_shared_l2": _command_base(
            config,
            arm_name="A0_shared_l2",
            selector="mda",
            model_side_scope="shared",
            frozen_ae_gmm_state=state,
            fit_reference_ae_gmm=state is None,
        ),
        "A_per_side_l2": _command_base(
            config,
            arm_name="A_per_side_l2",
            selector="archetype_prescreen_side_mda",
            model_side_scope="per_side",
            frozen_ae_gmm_state=a0_state,
            fit_reference_ae_gmm=False,
        ),
        "B_corrfirst_if_per_side": _command_base(
            config,
            arm_name="B_corrfirst_pack_b",
            selector="archetype_prescreen_side_mda_corrfirst",
            model_side_scope="per_side",
            frozen_ae_gmm_state=a0_state,
            fit_reference_ae_gmm=False,
        ),
        "B_corrfirst_if_shared": _command_base(
            config,
            arm_name="B_corrfirst_pack_b",
            selector="archetype_prescreen_side_mda_corrfirst",
            model_side_scope="shared",
            frozen_ae_gmm_state=a0_state,
            fit_reference_ae_gmm=False,
        ),
    }


def build_stage_c_command(
    config: MatrixConfig,
    *,
    source_arm_name: str,
    model_side_scope: str,
) -> list[str]:
    """Build the isolated target/geometry HPO command for the selected A/B arm."""
    if model_side_scope not in {"shared", "per_side"}:
        raise ValueError(f"Unsupported model_side_scope={model_side_scope}")
    return [
        str(config.python),
        str(STAGE_C_RUNNER),
        "--source-arm-dir",
        str(config.output_dir / source_arm_name),
        "--output-dir",
        str(config.output_dir / "C_hierarchical_target_weight_hpo"),
        "--fixed-params-json",
        str(config.fixed_params_json),
        "--model-side-scope",
        str(model_side_scope),
        "--label-path-purge-hours",
        str(float(config.purge_hours)),
        "--seed",
        str(int(config.seed)),
    ]


def stage_c_contract(*, source_arm_name: str, model_side_scope: str) -> dict[str, Any]:
    return {
        "source_arm": str(source_arm_name),
        "model_side_scope": str(model_side_scope),
        "target": "continuous corrected base soft target, side-specific geometry",
        "objective": "gross timestamp-side top10/top20/top30 stable ranking",
        "weight_search": {
            "target_exponent": [1.0, 1.25, 1.5, 1.75, 2.0],
            "strength_ratio_continuous": [3.0, 12.0],
            "normalization": "target_power_p99_clip_bounded_mean_1",
            "rebalance": "timestamp plus tempered archetype support",
        },
        "validation": "chronological internal train-only folds with 25h purge",
        "selection": "Optuna proxy stages with pruning, then full LightGBM finalists with early stopping",
        "oos": "Apr-Jun rows are scored once after target/weight selection",
    }


def _validate_config(config: MatrixConfig) -> None:
    months = _iso_months(config.oos_months)
    expected_start = pd.Timestamp(f"{months[0]}-01", tz="UTC")
    expected_end = expected_start - pd.Timedelta(days=int(config.train_window_days))
    if expected_start != pd.Timestamp(DEFAULT_TRAIN_END):
        raise ValueError(
            "The default ablation is intentionally fixed to Apr 2025-Mar 2026 "
            "train and Apr-Jun 2026 OOS. Use an explicit future extension rather "
            "than silently changing this controlled comparison."
        )
    if expected_end != pd.Timestamp(DEFAULT_TRAIN_START):
        raise ValueError("train_window_days must be 365 for the controlled 1-year arm")
    if not math.isclose(float(config.purge_hours), DEFAULT_PURGE_HOURS):
        raise ValueError("The controlled ablation requires the corrected 25h label purge")
    if not config.fixed_params_json.is_file():
        raise FileNotFoundError(f"Missing base L2 parameter contract: {config.fixed_params_json}")
    if config.frozen_ae_gmm_output_sidecar is not None:
        if config.frozen_ae_gmm_state is None:
            raise ValueError(
                "A precomputed AE/GMM output sidecar requires its frozen state"
            )
        if not config.frozen_ae_gmm_output_sidecar.is_file():
            raise FileNotFoundError(config.frozen_ae_gmm_output_sidecar)
    _load_base_l2_contract(config.fixed_params_json)


def _load_base_l2_contract(path: Path) -> dict[str, Any]:
    """Fail closed if a meta residual contract is supplied to the base matrix."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = dict(payload.get("params") or payload)
    if str(params.get("target_mode", "")) != "target_soft":
        raise ValueError(
            "The matrix requires the corrected base soft target (target_soft), "
            f"not {params.get('target_mode')!r} from {path}"
        )
    if str(params.get("loss_function", "")) != "regression":
        raise ValueError(
            "The matrix requires the base L2 regression loss, "
            f"not {params.get('loss_function')!r} from {path}"
        )
    missing = [
        name
        for name in (
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
        )
        if name not in params
    ]
    if missing:
        raise ValueError(f"Incomplete base L2 parameter contract {path}: missing={missing}")
    return params


def _read_ledger(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "__ts__" not in frame.columns:
        raise ValueError(f"{path} has no __ts__ column")
    if "__symbol__" not in frame.columns:
        raise ValueError(f"{path} has no __symbol__ column")
    frame = frame.copy(deep=False)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    if "side_name" not in frame.columns:
        if "side" not in frame.columns:
            raise ValueError(f"{path} has neither side_name nor side")
        frame["side_name"] = _side_name(frame["side"])
    else:
        frame["side_name"] = _side_name(frame["side_name"])
    frame = frame.loc[
        frame["__ts__"].ge(start) & frame["__ts__"].lt(end),
    ].copy()
    duplicate = frame.duplicated(list(KEYS), keep=False)
    if duplicate.any():
        raise ValueError(f"{path} has duplicate base keys in the requested OOS scope")
    return frame


def _first_column(frame: pd.DataFrame, candidates: Sequence[str], *, name: str) -> str:
    column = next((candidate for candidate in candidates if candidate in frame.columns), None)
    if column is None:
        raise ValueError(f"Could not find {name}; checked={list(candidates)}")
    return column


def _score_column(frame: pd.DataFrame) -> str:
    return _first_column(frame, ("score", "score_base", "prediction", "pred"), name="base score")


def _net_column(frame: pd.DataFrame) -> str:
    return _first_column(
        frame,
        ("__first_touch_net__", "first_touch_net", "ev_after_1pct", "__u_policy_net__", "__r_policy_net__"),
        name="stored net EV",
    )


def _gross_series(frame: pd.DataFrame, net: pd.Series, *, net_col: str | None = None) -> tuple[pd.Series, str]:
    if net_col == "stage_c_geometry_net_return" and "stage_c_geometry_gross_return" in frame.columns:
        return (
            pd.to_numeric(frame["stage_c_geometry_gross_return"], errors="coerce"),
            "stored:stage_c_geometry_gross_return",
        )
    gross_column = next(
        (
            name
            for name in (
                "__first_touch_gross__",
                "first_touch_gross",
                "gross_ev",
                "gross_return",
                "__u_policy_gross__",
            )
            if name in frame.columns
        ),
        None,
    )
    if gross_column is not None:
        return pd.to_numeric(frame[gross_column], errors="coerce"), f"stored:{gross_column}"
    cost_column = next(
        (
            name
            for name in (
                "__first_touch_round_trip_cost__",
                "first_touch_round_trip_cost",
                "round_trip_cost",
            )
            if name in frame.columns
        ),
        None,
    )
    if cost_column is None:
        raise ValueError("Cannot derive gross EV without a stored gross or round-trip cost column")
    return net + pd.to_numeric(frame[cost_column], errors="coerce"), f"net_plus_stored:{cost_column}"


def _rank_timestamp_side(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["__ts__", "side_name", score_col, "__symbol__"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).copy()
    grouped = ordered.groupby(["__ts__", "side_name"], observed=True, sort=False)
    ordered["_rank"] = grouped.cumcount().add(1).astype(np.int32)
    ordered["_group_rows"] = grouped[score_col].transform("size").astype(np.int32)
    return ordered


def _metric_rows(
    frame: pd.DataFrame,
    *,
    arm: str,
    score_col: str,
    net_col: str,
    net_override: pd.Series | None = None,
    gross_override: pd.Series | None = None,
) -> tuple[list[dict[str, Any]], str]:
    ranked = _rank_timestamp_side(frame, score_col)
    if net_override is None:
        net = pd.to_numeric(ranked[net_col], errors="coerce")
    else:
        net = pd.to_numeric(net_override.reindex(ranked.index), errors="coerce")
    if gross_override is None:
        gross, gross_source = _gross_series(ranked, net, net_col=net_col)
    else:
        gross = pd.to_numeric(gross_override.reindex(ranked.index), errors="coerce")
        gross_source = "cost_rebase:derived_gross_before_fee_and_spread"
    ranked = ranked.assign(_net=net, _gross=gross)
    calendar_ts = ranked["__ts__"].dt.tz_localize(None)
    ranked["month"] = calendar_ts.dt.to_period("M").astype(str)
    ranked["week_start"] = calendar_ts.dt.to_period("W-SUN").dt.start_time
    archetype_col = next(
        (
            column
            for column in (
                "__archetype_label_family__",
                "archetype_label_family",
                "__archetype_policy_key__",
                "policy_archetype",
                "local_side_archetype",
            )
            if column in ranked.columns
        ),
        None,
    )
    if archetype_col is not None:
        ranked["archetype"] = ranked[archetype_col].astype(str)

    scopes: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("overall", ()),
        ("side", ("side_name",)),
        ("month", ("month",)),
        ("side_month", ("side_name", "month")),
    )
    if archetype_col is not None:
        scopes += (("side_month_archetype", ("side_name", "month", "archetype")),)
    rows: list[dict[str, Any]] = []
    for top_frac in TOP_FRACS:
        selected = ranked.loc[
            ranked["_rank"].le(np.ceil(ranked["_group_rows"] * float(top_frac)))
        ].copy()
        for scope, group_cols in scopes:
            grouped: Iterable[tuple[Any, pd.DataFrame]]
            if group_cols:
                grouped = selected.groupby(list(group_cols), observed=True, sort=True)
            else:
                grouped = [((), selected)]
            for group_key, subset in grouped:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                values = dict(zip(group_cols, group_key))
                daily = subset.groupby(subset["__ts__"].dt.floor("D"), observed=True)["_net"].mean()
                weekly = subset.groupby("week_start", observed=True)["_net"].mean()
                monthly = subset.groupby("month", observed=True)["_net"].mean()
                rows.append(
                    {
                        "arm": arm,
                        "scope": scope,
                        "top_frac": float(top_frac),
                        "selection_basis": "timestamp_side_topk_on_identical_rows",
                        "candidate_rows": int(len(frame)),
                        "selected_rows": int(len(subset)),
                        "selected_days": int(subset["__ts__"].dt.floor("D").nunique()),
                        "trades_per_day": float(len(subset) / max(1, subset["__ts__"].dt.floor("D").nunique())),
                        "mean_gross_ev": float(subset["_gross"].mean()),
                        "sum_gross_ev": float(subset["_gross"].sum()),
                        "mean_net_ev": float(subset["_net"].mean()),
                        "sum_net_ev": float(subset["_net"].sum()),
                        "positive_net_ev_rate": float(subset["_net"].gt(0.0).mean()),
                        "worst_day_mean_net_ev": float(daily.min()) if len(daily) else float("nan"),
                        "worst_week_mean_net_ev": float(weekly.min()) if len(weekly) else float("nan"),
                        "worst_month_mean_net_ev": float(monthly.min()) if len(monthly) else float("nan"),
                        **values,
                    }
                )
    return rows, gross_source


def _common_frames(ledgers: Mapping[str, Path], *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    loaded = {arm: _read_ledger(path, start=start, end=end) for arm, path in ledgers.items()}
    key_frames = {arm: frame.loc[:, list(KEYS)] for arm, frame in loaded.items()}
    common = next(iter(key_frames.values())).copy()
    coverage_rows: list[dict[str, Any]] = []
    for arm, keys in key_frames.items():
        coverage_rows.append({"arm": arm, "oos_rows": int(len(keys))})
        common = common.merge(keys, on=list(KEYS), how="inner", validate="one_to_one")
    for row in coverage_rows:
        row["common_rows"] = int(len(common))
        row["common_share"] = float(len(common) / max(1, row["oos_rows"]))
    common = common.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    aligned: dict[str, pd.DataFrame] = {}
    reference_arm = next(iter(loaded))
    reference = loaded[reference_arm].set_index(list(KEYS), verify_integrity=True)
    reference_net = _net_column(reference.reset_index())
    reference_values = pd.to_numeric(reference.loc[pd.MultiIndex.from_frame(common), reference_net], errors="coerce").to_numpy()
    for arm, frame in loaded.items():
        indexed = frame.set_index(list(KEYS), verify_integrity=True)
        aligned_frame = indexed.loc[pd.MultiIndex.from_frame(common)].reset_index()
        net_col = _net_column(aligned_frame)
        if arm != reference_arm:
            candidate = pd.to_numeric(aligned_frame[net_col], errors="coerce").to_numpy()
            mask = np.isfinite(reference_values) & np.isfinite(candidate)
            if mask.any() and not np.allclose(reference_values[mask], candidate[mask], rtol=0.0, atol=1e-10):
                raise ValueError(f"Outcome mismatch on identical OOS rows: {reference_arm} vs {arm}")
        aligned[arm] = aligned_frame
    return aligned, common, pd.DataFrame(coverage_rows)


def report_matrix(
    ledgers: Mapping[str, Path],
    *,
    output_dir: Path,
    oos_months: Sequence[str],
    spread_snapshot_path: Path | None = None,
    spread_quantile: float = 0.90,
    fee_round_trip_pct: float = 0.0015,
) -> dict[str, Path]:
    """Report common-row top-k economics for completed arm ledgers."""
    months = _iso_months(oos_months)
    start = pd.Timestamp(f"{months[0]}-01", tz="UTC")
    end = pd.Timestamp((pd.Period(months[-1], freq="M") + 1).start_time, tz="UTC")
    aligned, common, coverage = _common_frames(ledgers, start=start, end=end)
    output_dir.mkdir(parents=True, exist_ok=True)
    spread_contract: dict[str, Any] | None = None
    spread_by_symbol: pd.Series | None = None
    if spread_snapshot_path is not None:
        if not 0.0 < float(spread_quantile) < 1.0:
            raise ValueError("spread_quantile must lie strictly between zero and one")
        if float(fee_round_trip_pct) < 0.0:
            raise ValueError("fee_round_trip_pct cannot be negative")
        spread_frame = pd.read_parquet(
            spread_snapshot_path,
            columns=["observed_ts", "symbol", "spread_bps"],
        )
        spread_frame["observed_ts"] = pd.to_datetime(
            spread_frame["observed_ts"], utc=True, errors="coerce"
        )
        spread_frame["symbol"] = spread_frame["symbol"].astype(str)
        spread_frame["spread_bps"] = pd.to_numeric(
            spread_frame["spread_bps"], errors="coerce"
        )
        spread_frame = spread_frame.loc[
            spread_frame["observed_ts"].notna()
            & spread_frame["spread_bps"].ge(0.0)
            & np.isfinite(spread_frame["spread_bps"])
        ].copy()
        if spread_frame.empty:
            raise ValueError(f"No valid spread observations in {spread_snapshot_path}")
        grouped_spread = spread_frame.groupby("symbol", observed=True)["spread_bps"]
        spread_by_symbol = grouped_spread.quantile(float(spread_quantile))
        spread_support = grouped_spread.size()
        required_symbols = sorted(
            set().union(*(set(frame["__symbol__"].astype(str)) for frame in aligned.values()))
        )
        missing_symbols = sorted(set(required_symbols).difference(spread_by_symbol.index))
        if missing_symbols:
            raise ValueError(
                "Spread-cost rebase requires exact symbol coverage; "
                f"missing={missing_symbols[:20]} count={len(missing_symbols)}"
            )
        snapshot_bytes = spread_snapshot_path.read_bytes()
        support_required = spread_support.reindex(required_symbols)
        spread_contract = {
            "schema": "asset_p90_spread_plus_round_trip_fee_v1",
            "snapshot_path": str(spread_snapshot_path),
            "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "snapshot_rows_valid": int(len(spread_frame)),
            "snapshot_symbols": int(spread_frame["symbol"].nunique()),
            "snapshot_min_ts": str(spread_frame["observed_ts"].min()),
            "snapshot_max_ts": str(spread_frame["observed_ts"].max()),
            "required_symbols": int(len(required_symbols)),
            "missing_symbols": [],
            "minimum_observations_per_required_symbol": int(support_required.min()),
            "median_observations_per_required_symbol": float(support_required.median()),
            "spread_quantile": float(spread_quantile),
            "fee_round_trip_pct": float(fee_round_trip_pct),
            "formula": "gross_return - fee_round_trip_pct - p90_full_bid_ask_spread_bps / 10000",
            "interpretation": "one full spread per round trip: half at entry plus half at exit",
            "evidence_scope": "post_hoc_cost_sensitivity_not_untouched_oos",
        }
    all_rows: list[dict[str, Any]] = []
    gross_sources: dict[str, str] = {}
    for arm, frame in aligned.items():
        evaluation_net_col = (
            "stage_c_geometry_net_return"
            if arm == "C_hierarchical_target_weight_hpo" and "stage_c_geometry_net_return" in frame.columns
            else _net_column(frame)
        )
        net_override: pd.Series | None = None
        gross_override: pd.Series | None = None
        if spread_by_symbol is not None:
            stored_net = pd.to_numeric(frame[evaluation_net_col], errors="coerce")
            gross_override, gross_source = _gross_series(
                frame, stored_net, net_col=evaluation_net_col
            )
            row_spread_bps = frame["__symbol__"].astype(str).map(spread_by_symbol)
            if row_spread_bps.isna().any():
                raise ValueError(f"Unexpected missing spread after coverage validation for arm={arm}")
            net_override = (
                gross_override
                - float(fee_round_trip_pct)
                - pd.to_numeric(row_spread_bps, errors="coerce") / 10_000.0
            )
        metric_rows, gross_source = _metric_rows(
            frame,
            arm=arm,
            score_col=_score_column(frame),
            net_col=evaluation_net_col,
            net_override=net_override,
            gross_override=gross_override,
        )
        all_rows.extend(metric_rows)
        gross_sources[arm] = gross_source
        if arm == "C_hierarchical_target_weight_hpo":
            rank_arm = "C_rank_on_incumbent_geometry"
            rank_rows, rank_gross_source = _metric_rows(
                frame,
                arm=rank_arm,
                score_col=_score_column(frame),
                net_col=_net_column(frame),
                net_override=(
                    _gross_series(
                        frame,
                        pd.to_numeric(frame[_net_column(frame)], errors="coerce"),
                        net_col=_net_column(frame),
                    )[0]
                    - float(fee_round_trip_pct)
                    - frame["__symbol__"].astype(str).map(spread_by_symbol) / 10_000.0
                    if spread_by_symbol is not None
                    else None
                ),
                gross_override=(
                    _gross_series(
                        frame,
                        pd.to_numeric(frame[_net_column(frame)], errors="coerce"),
                        net_col=_net_column(frame),
                    )[0]
                    if spread_by_symbol is not None
                    else None
                ),
            )
            all_rows.extend(rank_rows)
            gross_sources[rank_arm] = rank_gross_source
    metrics = pd.DataFrame(all_rows)
    baseline = metrics.loc[metrics["arm"].eq("A0_shared_l2")].copy()
    delta_keys = [column for column in ("scope", "top_frac", "side_name", "month", "archetype") if column in metrics.columns]
    delta_metric_cols = (
        "mean_gross_ev",
        "sum_gross_ev",
        "mean_net_ev",
        "sum_net_ev",
        "positive_net_ev_rate",
        "worst_day_mean_net_ev",
        "worst_week_mean_net_ev",
        "worst_month_mean_net_ev",
        "trades_per_day",
    )
    baseline = baseline.loc[:, [*delta_keys, *delta_metric_cols]].rename(
        columns={column: f"baseline_{column}" for column in delta_metric_cols}
    )
    comparison = metrics.merge(baseline, on=delta_keys, how="left", validate="many_to_one")
    for column in delta_metric_cols:
        comparison[f"delta_vs_a0_{column}"] = comparison[column] - comparison[f"baseline_{column}"]

    paths = {
        "common_rows": output_dir / "common_oos_rows.parquet",
        "coverage": output_dir / "arm_common_row_coverage.csv",
        "metrics": output_dir / "topk_metrics_identical_rows.csv",
        "comparison": output_dir / "topk_metrics_delta_vs_a0.csv",
        "side": output_dir / "topk_metrics_by_side.csv",
        "month": output_dir / "topk_metrics_by_month.csv",
        "side_month": output_dir / "topk_metrics_by_side_month.csv",
        "provenance": output_dir / "report_provenance.json",
    }
    common.to_parquet(paths["common_rows"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    metrics.loc[metrics["scope"].eq("side")].to_csv(paths["side"], index=False)
    metrics.loc[metrics["scope"].eq("month")].to_csv(paths["month"], index=False)
    metrics.loc[metrics["scope"].eq("side_month")].to_csv(paths["side_month"], index=False)
    if "archetype" in metrics.columns:
        archetype_path = output_dir / "topk_metrics_by_side_month_archetype.csv"
        metrics.loc[metrics["scope"].eq("side_month_archetype")].to_csv(archetype_path, index=False)
        paths["side_month_archetype"] = archetype_path
    paths["provenance"].write_text(
        json.dumps(
            {
                "oos_months": list(months),
                "common_rows": int(len(common)),
                "selection_basis": "timestamp_side_topk_on_identical_rows",
                "gross_ev_source_by_arm": gross_sources,
                "net_ev_contract": (
                    "gross minus the recorded p90 full spread and round-trip fee"
                    if spread_contract is not None
                    else (
                        "A/B and C_rank_on_incumbent_geometry use stored corrected-label net EV; "
                        "C_hierarchical_target_weight_hpo uses its exact selected geometry net EV; "
                        "costs are not subtracted again"
                    )
                ),
                "spread_cost_rebase": spread_contract,
                "ledger_paths": {arm: str(path) for arm, path in ledgers.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _top10_overall_net(metrics_path: Path, arm: str) -> float:
    metrics = pd.read_csv(metrics_path)
    row = metrics.loc[
        metrics["arm"].eq(arm)
        & metrics["scope"].eq("overall")
        & np.isclose(pd.to_numeric(metrics["top_frac"], errors="coerce"), 0.10)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one overall top10 row for {arm}")
    return float(row.iloc[0]["mean_net_ev"])


def choose_b_scope(shared_top10_net: float, per_side_top10_net: float) -> str:
    """Use per-side Pack B only if A strictly beats the shared A0 baseline."""
    if not math.isfinite(shared_top10_net) or not math.isfinite(per_side_top10_net):
        return "shared"
    return "per_side" if per_side_top10_net > shared_top10_net else "shared"


def _run_command(command: Sequence[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    # Local side x archetype samples are deliberately shorter than the global
    # base burn-in. Keep every selector CV chronological instead of failing or
    # falling back to shuffled folds.
    env["EPM_LGBM_FORWARD_ALLOW_SHORT_HISTORY_FALLBACK"] = "1"
    subprocess.run(list(command), cwd=cwd, env=env, check=True)


def _run_arm_command(
    command: Sequence[str],
    *,
    cwd: Path,
    ledger_path: Path,
) -> None:
    if ledger_path.is_file():
        print(f"Skipping completed ablation arm: {ledger_path.parent}", flush=True)
        return
    _run_command(command, cwd=cwd)


def _arm_ledger(config: MatrixConfig, arm_name: str) -> Path:
    return config.output_dir / arm_name / "best_oos_scored_ledger.parquet"


def _write_manifest(config: MatrixConfig, *, commands: Mapping[str, Sequence[str]], c_spec: Mapping[str, Any], status: str, b_scope: str | None = None) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "base_side_target_ablation_matrix_v1",
        "status": status,
        "split": {
            "train_start_utc": DEFAULT_TRAIN_START,
            "train_end_exclusive_utc": DEFAULT_TRAIN_END,
            "oos_months": list(config.oos_months),
            "single_fit_oos_window": True,
            "growing_windows_forbidden": True,
            "label_path_purge_hours": float(config.purge_hours),
        },
        "base_contract": {
            "target": "corrected base soft target",
            "params": str(config.fixed_params_json),
            "parameter_values": _load_base_l2_contract(config.fixed_params_json),
            "loss": "L2 regression",
            "pack_b_is_selector_only": True,
            "pack_b_excludes": ["meta residual target", "meta model parameters"],
        },
        "ae_gmm_contract": {
            "shared_frozen_state": str(config.frozen_ae_gmm_state or _state_path_for_a0(config)),
            "reference_state_owner": "provided_state" if config.frozen_ae_gmm_state else "A0_shared_l2",
            "all_following_arms_reuse_exact_state": True,
            "precomputed_output_sidecar": (
                str(config.frozen_ae_gmm_output_sidecar)
                if config.frozen_ae_gmm_output_sidecar is not None
                else None
            ),
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "commands": {name: list(command) for name, command in commands.items()},
        "conditional_b": {
            "rule": "run Pack-B per_side iff A per-side top10 net EV strictly beats A0 on identical rows; otherwise shared",
            "resolved_scope": b_scope,
        },
        "stage_c": dict(c_spec),
    }
    path = config.output_dir / "ablation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--feature-list-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-params-json", type=Path, default=DEFAULT_L2_PARAMS)
    parser.add_argument("--oos-months", default=",".join(DEFAULT_OOS_MONTHS))
    parser.add_argument("--train-window-days", type=int, default=365)
    parser.add_argument("--label-path-purge-hours", type=float, default=DEFAULT_PURGE_HOURS)
    parser.add_argument("--feature-selection-sample-rows", type=int, default=300_000)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--frozen-ae-gmm-state", type=Path, default=None)
    parser.add_argument("--frozen-ae-gmm-output-sidecar", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="Run A0, A, and conditional B via subprocess.")
    parser.add_argument("--skip-stage-c", action="store_true", help="Stop after A/B. Stage C otherwise runs on the A/B winner.")
    parser.add_argument("--report-only", action="store_true", help="Only build common-row reports from existing arm ledgers.")
    parser.add_argument("--report-output-dir", type=Path, default=None)
    parser.add_argument("--spread-snapshot-path", type=Path, default=None)
    parser.add_argument("--spread-quantile", type=float, default=0.90)
    parser.add_argument("--fee-round-trip-pct", type=float, default=0.0015)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.execute and args.report_only:
        raise ValueError("--execute and --report-only are mutually exclusive")
    config = MatrixConfig(
        labels_path=args.labels_path,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        output_dir=args.output_dir,
        fixed_params_json=args.fixed_params_json,
        oos_months=_iso_months(args.oos_months),
        train_window_days=int(args.train_window_days),
        purge_hours=float(args.label_path_purge_hours),
        feature_selection_sample_rows=int(args.feature_selection_sample_rows),
        max_train_rows=int(args.max_train_rows),
        seed=int(args.seed),
        python=str(args.python),
        frozen_ae_gmm_state=args.frozen_ae_gmm_state,
        frozen_ae_gmm_output_sidecar=args.frozen_ae_gmm_output_sidecar,
    )
    if args.report_only:
        ledgers = {
            "A0_shared_l2": _arm_ledger(config, "A0_shared_l2"),
            "A_per_side_l2": _arm_ledger(config, "A_per_side_l2"),
        }
        b_path = _arm_ledger(config, "B_corrfirst_pack_b")
        if b_path.is_file():
            ledgers["B_corrfirst_pack_b"] = b_path
        c_path = config.output_dir / "C_hierarchical_target_weight_hpo" / "best_oos_scored_ledger.parquet"
        if c_path.is_file():
            ledgers["C_hierarchical_target_weight_hpo"] = c_path
        report_matrix(
            ledgers,
            output_dir=args.report_output_dir or config.output_dir / "reports",
            oos_months=config.oos_months,
            spread_snapshot_path=args.spread_snapshot_path,
            spread_quantile=float(args.spread_quantile),
            fee_round_trip_pct=float(args.fee_round_trip_pct),
        )
        return 0

    commands = build_arm_commands(config)
    c_spec = stage_c_contract(source_arm_name="pending_a_b_winner", model_side_scope="pending_a_scope")
    _write_manifest(config, commands=commands, c_spec=c_spec, status="planned")

    if args.execute:
        _run_arm_command(
            commands["A0_shared_l2"],
            cwd=ROOT,
            ledger_path=_arm_ledger(config, "A0_shared_l2"),
        )
        if config.frozen_ae_gmm_state is None and not _state_path_for_a0(config).is_file():
            raise FileNotFoundError(
                "A0 completed without the expected shared AE/GMM state: "
                f"{_state_path_for_a0(config)}"
            )
        _run_arm_command(
            commands["A_per_side_l2"],
            cwd=ROOT,
            ledger_path=_arm_ledger(config, "A_per_side_l2"),
        )
        initial_report = report_matrix(
            {
                "A0_shared_l2": _arm_ledger(config, "A0_shared_l2"),
                "A_per_side_l2": _arm_ledger(config, "A_per_side_l2"),
            },
            output_dir=config.output_dir / "reports_after_a",
            oos_months=config.oos_months,
        )
        b_scope = choose_b_scope(
            _top10_overall_net(initial_report["metrics"], "A0_shared_l2"),
            _top10_overall_net(initial_report["metrics"], "A_per_side_l2"),
        )
        b_command = commands[f"B_corrfirst_if_{b_scope}"]
        _run_arm_command(
            b_command,
            cwd=ROOT,
            ledger_path=_arm_ledger(config, "B_corrfirst_pack_b"),
        )
        b_arm = "B_corrfirst_pack_b"
        ledgers = {
            "A0_shared_l2": _arm_ledger(config, "A0_shared_l2"),
            "A_per_side_l2": _arm_ledger(config, "A_per_side_l2"),
            "B_corrfirst_pack_b": _arm_ledger(config, "B_corrfirst_pack_b"),
        }
        report_matrix(ledgers, output_dir=config.output_dir / "reports", oos_months=config.oos_months)
        if not bool(args.skip_stage_c):
            report_path = config.output_dir / "reports" / "topk_metrics_identical_rows.csv"
            b_top10 = _top10_overall_net(report_path, b_arm)
            a0_top10 = _top10_overall_net(report_path, "A0_shared_l2")
            a_top10 = _top10_overall_net(report_path, "A_per_side_l2")
            source_arm = b_arm if b_top10 > max(a0_top10, a_top10) else (
                "A_per_side_l2" if a_top10 > a0_top10 else "A0_shared_l2"
            )
            c_scope = "per_side" if a_top10 > a0_top10 else "shared"
            c_command = build_stage_c_command(
                config, source_arm_name=source_arm, model_side_scope=c_scope
            )
            c_ledger = config.output_dir / "C_hierarchical_target_weight_hpo" / "best_oos_scored_ledger.parquet"
            _run_arm_command(c_command, cwd=ROOT, ledger_path=c_ledger)
            if not c_ledger.is_file():
                raise FileNotFoundError(f"Stage C did not materialize its OOS ledger: {c_ledger}")
            ledgers["C_hierarchical_target_weight_hpo"] = c_ledger
            report_matrix(ledgers, output_dir=config.output_dir / "reports", oos_months=config.oos_months)
            c_spec = stage_c_contract(source_arm_name=source_arm, model_side_scope=c_scope)
            c_spec["command"] = c_command
        _write_manifest(config, commands=commands, c_spec=c_spec, status="completed_abc" if not bool(args.skip_stage_c) else "completed_ab", b_scope=b_scope)
        return 0

    print("Planned only. Pass --execute to run subprocesses, or --report-only for existing ledgers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
