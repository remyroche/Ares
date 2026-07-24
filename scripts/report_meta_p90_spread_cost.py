"""Compare base and meta OOS ranking under a static Kraken p90 spread stress.

This is a ranking diagnostic, not a causal transaction-cost reconstruction.  It
pins one pooled, per-symbol p90 full-spread table and applies it equally to
base and meta rows.  The evaluator ranks each model only within decision
timestamp x side, which is required when long and short use separate models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
TOP_FRACS = (0.10, 0.20, 0.30)
SCHEMA = "meta_base_p90_spread_fee15bps_timestamp_side_v2"

OPTIONAL_METRIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "clean_positive_rate": ("clean_exec", "clean_exec_label", "clean_positive"),
    "dirty_positive_rate": ("dirty_positive", "dirty_positive_label"),
    "first_touch_bad_mae_rate": (
        "first_touch_bad_mae_1r",
        "__first_touch_bad_mae_1r__",
    ),
    "full_path_bad_mae_rate": (
        "full_path_bad_mae_1r",
        "__path_full_bad_mae_1r__",
    ),
    "timeout_rate": ("timeout", "first_touch_timeout", "__first_touch_timeout__"),
}
ARCHETYPE_COLUMNS = (
    "archetype_label_family",
    "__archetype_label_family__",
    "policy_archetype",
    "local_side_archetype",
    "source_archetype",
)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: pd.DataFrame
    deltas: pd.DataFrame
    eligible: pd.Series
    integrity: dict[str, Any]
    provenance: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    available = _parquet_columns(path)
    missing = sorted(set(columns).difference(available))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")
    return pd.read_parquet(path, columns=columns)


def _coerce_keys(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["__symbol__"] = out["__symbol__"].astype(str)
    out["side_name"] = out["side_name"].astype(str).str.lower()
    invalid = out["__ts__"].isna() | out["__symbol__"].eq("") | ~out["side_name"].isin(("long", "short"))
    if bool(invalid.any()):
        raise ValueError(f"{role} has {int(invalid.sum())} invalid decision-time key rows")
    duplicate_count = int(out.duplicated(list(KEY_COLUMNS)).sum())
    if duplicate_count:
        raise ValueError(f"{role} has {duplicate_count} duplicate timestamp x symbol x side rows")
    return out


def _first_present(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in columns), None)


def _archetype_column(columns: set[str]) -> str | None:
    return _first_present(columns, ARCHETYPE_COLUMNS)


def _static_eligible_symbols(
    candidates: pd.DataFrame,
    spread: pd.DataFrame,
    *,
    eligible_symbols: int,
    spread_quantile: float,
) -> tuple[pd.Series, dict[str, Any]]:
    spread = spread.copy()
    spread["symbol"] = spread["symbol"].astype(str)
    spread["spread_bps"] = pd.to_numeric(spread["spread_bps"], errors="coerce")
    spread["observed_ts"] = pd.to_datetime(spread["observed_ts"], utc=True, errors="coerce")
    spread = spread.loc[np.isfinite(spread["spread_bps"]) & spread["spread_bps"].ge(0)].copy()
    if spread.empty:
        raise ValueError("Spread history has no finite non-negative spread observations")
    p90 = spread.groupby("symbol", observed=True)["spread_bps"].quantile(float(spread_quantile))
    candidate_symbols = pd.Index(candidates["__symbol__"].astype(str).unique())
    available = p90.reindex(candidate_symbols).dropna()
    if len(available) < int(eligible_symbols):
        raise ValueError(
            "Base candidate ledger has only "
            f"{len(available)} symbols with pooled p90 spreads; requires {eligible_symbols}"
        )
    selected = available.nsmallest(int(eligible_symbols)).sort_index()
    return selected, {
        "base_candidate_symbol_count": int(len(candidate_symbols)),
        "symbols_with_p90_spread": int(len(available)),
        "eligible_symbol_count": int(len(selected)),
        "eligible_cutoff_bps": float(selected.max()),
        "pooled_spread_observed_min_ts": spread["observed_ts"].min().isoformat()
        if spread["observed_ts"].notna().any()
        else None,
        "pooled_spread_observed_max_ts": spread["observed_ts"].max().isoformat()
        if spread["observed_ts"].notna().any()
        else None,
    }


def _selected_timestamp_side(frame: pd.DataFrame, score_col: str, frac: float) -> pd.DataFrame:
    """Take a deterministic top fraction within each decision timestamp x side."""
    ordered = frame.sort_values(
        ["__ts__", "side_name", score_col, "__symbol__"], kind="mergesort"
    )
    group_sizes = ordered.groupby(["__ts__", "side_name"], observed=True)[score_col].transform("size")
    ordered["__group_rank__"] = ordered.groupby(["__ts__", "side_name"], observed=True).cumcount() + 1
    cutoffs = np.ceil(group_sizes.to_numpy(dtype=np.float64) * float(frac)).astype(np.int64)
    selected = ordered.loc[ordered["__group_rank__"].to_numpy(dtype=np.int64) > (group_sizes.to_numpy(dtype=np.int64) - cutoffs)].copy()
    return selected.drop(columns="__group_rank__")


def _rate(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def _scope_rows(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    scope: str,
    group_columns: tuple[str, ...],
    selector: str,
    top_frac: float,
    optional_sources: dict[str, str],
) -> list[dict[str, Any]]:
    if group_columns:
        candidate_groups = candidates.groupby(list(group_columns), observed=True, dropna=False)
        selected_groups = {
            key if isinstance(key, tuple) else (key,): group
            for key, group in selected.groupby(list(group_columns), observed=True, dropna=False)
        }
    else:
        candidate_groups = [((), candidates)]
        selected_groups = {(): selected}
    rows: list[dict[str, Any]] = []
    for key, candidate_subset in candidate_groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        subset = selected_groups.get(key_tuple, selected.iloc[:0])
        selected_days = int(subset["__ts__"].dt.floor("D").nunique())
        weekly = subset.groupby("week_start", observed=True)["net_ev_p90_spread_fee15bps"].mean()
        monthly = subset.groupby("month", observed=True)["net_ev_p90_spread_fee15bps"].mean()
        row: dict[str, Any] = {
            "selector": selector,
            "scope": scope,
            "top_frac": float(top_frac),
            "candidate_rows": int(len(candidate_subset)),
            "selected_rows": int(len(subset)),
            "selected_days": selected_days,
            "trades_per_day": float(len(subset) / max(selected_days, 1)),
            "mean_gross_ev": float(subset["first_touch_gross"].mean()) if len(subset) else float("nan"),
            "sum_gross_ev": float(subset["first_touch_gross"].sum()) if len(subset) else 0.0,
            "mean_net_ev": float(subset["net_ev_p90_spread_fee15bps"].mean()) if len(subset) else float("nan"),
            "sum_net_ev": float(subset["net_ev_p90_spread_fee15bps"].sum()) if len(subset) else 0.0,
            "positive_net_ev_rate": _rate(
                subset["net_ev_p90_spread_fee15bps"].gt(0.0)
            ),
            "worst_week_mean_net_ev": float(weekly.min()) if len(weekly) else float("nan"),
            "worst_month_mean_net_ev": float(monthly.min()) if len(monthly) else float("nan"),
            "long_share": _rate(subset["side_name"].eq("long")),
        }
        for metric, source in optional_sources.items():
            row[metric] = _rate(subset[source])
        row.update(dict(zip(group_columns, key_tuple)))
        rows.append(row)
    return rows


def _selector_metrics(
    frame: pd.DataFrame,
    score_col: str,
    *,
    selector: str,
    archetype_col: str | None,
    optional_sources: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, tuple[str, ...]]] = [
        ("overall", ()),
        ("month", ("month",)),
        ("week", ("week_start",)),
        ("side", ("side_name",)),
    ]
    if archetype_col is not None:
        scopes.extend(
            [
                ("archetype", (archetype_col,)),
                ("side_archetype", ("side_name", archetype_col)),
            ]
        )
    for frac in TOP_FRACS:
        selected = _selected_timestamp_side(frame, score_col, frac)
        for scope, groups in scopes:
            rows.extend(
                _scope_rows(
                    frame,
                    selected,
                    scope=scope,
                    group_columns=groups,
                    selector=selector,
                    top_frac=frac,
                    optional_sources=optional_sources,
                )
            )
    return rows


def _delta_vs_base(metrics: pd.DataFrame, *, archetype_col: str | None) -> pd.DataFrame:
    base = metrics.loc[metrics["selector"].eq("base_score")].copy()
    meta = metrics.loc[metrics["selector"].eq("meta_base_soft_label")].copy()
    group_columns = ["scope", "top_frac", "month", "week_start", "side_name"]
    if archetype_col is not None:
        group_columns.append(archetype_col)
    for frame in (base, meta):
        for column in group_columns:
            if column not in frame.columns:
                frame[column] = np.nan
    keys = group_columns
    base = base.drop(columns="selector").rename(columns={column: f"base_{column}" for column in metrics.columns if column not in keys + ["selector"]})
    meta = meta.drop(columns="selector")
    merged = meta.merge(base, on=keys, how="inner", validate="one_to_one")
    for column in ("mean_gross_ev", "sum_gross_ev", "mean_net_ev", "sum_net_ev", "positive_net_ev_rate", "worst_week_mean_net_ev", "worst_month_mean_net_ev", "clean_positive_rate", "dirty_positive_rate", "first_touch_bad_mae_rate", "full_path_bad_mae_rate", "timeout_rate"):
        base_column = f"base_{column}"
        if column in merged.columns and base_column in merged.columns:
            merged[f"delta_{column}_vs_base"] = merged[column] - merged[base_column]
    return merged.sort_values(keys, kind="mergesort").reset_index(drop=True)


def evaluate_frames(
    predictions: pd.DataFrame,
    base_candidates: pd.DataFrame,
    spread_history: pd.DataFrame,
    *,
    eligible_symbols: int = 170,
    spread_quantile: float = 0.90,
    fee_round_trip_pct: float = 0.0015,
    base_candidate_selector_column: str = "selected_top30",
) -> EvaluationResult:
    """Evaluate scores on exact OOS candidate keys under the fixed stress cost."""
    required_prediction = set(KEY_COLUMNS) | {"score_base", "score_meta_base_soft_label", "first_touch_gross"}
    missing_prediction = sorted(required_prediction.difference(predictions.columns))
    if missing_prediction:
        raise ValueError(f"Predictions missing required columns: {missing_prediction}")
    required_candidates = set(KEY_COLUMNS)
    missing_candidates = sorted(required_candidates.difference(base_candidates.columns))
    if missing_candidates:
        raise ValueError(f"Base candidate ledger missing required columns: {missing_candidates}")
    required_spread = {"observed_ts", "symbol", "spread_bps"}
    missing_spread = sorted(required_spread.difference(spread_history.columns))
    if missing_spread:
        raise ValueError(f"Spread history missing required columns: {missing_spread}")

    prediction = _coerce_keys(predictions, role="predictions")
    base = _coerce_keys(base_candidates, role="base candidate ledger")
    if base_candidate_selector_column in base.columns:
        base = base.loc[base[base_candidate_selector_column].fillna(False).astype(bool)].copy()
    if base.empty:
        raise ValueError("Base candidate ledger has no selected candidate rows")

    # Constrain the universe to the prediction OOS interval but never to meta score validity.
    prediction_min = prediction["__ts__"].min()
    prediction_max = prediction["__ts__"].max()
    base_scope = base.loc[base["__ts__"].between(prediction_min, prediction_max)].copy()
    if base_scope.empty:
        raise ValueError("No base candidate rows overlap the prediction OOS period")
    p90, p90_integrity = _static_eligible_symbols(
        base_scope,
        spread_history,
        eligible_symbols=int(eligible_symbols),
        spread_quantile=float(spread_quantile),
    )

    base_keys = base_scope.loc[base_scope["__symbol__"].isin(p90.index), list(KEY_COLUMNS)]
    duplicate_base_keys = int(base_keys.duplicated(list(KEY_COLUMNS)).sum())
    if duplicate_base_keys:
        raise ValueError(f"Selected base candidate ledger has {duplicate_base_keys} duplicate keys")
    before_candidate_join = int(len(prediction))
    frame = prediction.merge(base_keys, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("No prediction rows overlap the p90-eligible base candidate ledger")
    frame["p90_spread_bps"] = frame["__symbol__"].map(p90)
    frame["first_touch_gross"] = pd.to_numeric(frame["first_touch_gross"], errors="coerce")
    frame["score_base"] = pd.to_numeric(frame["score_base"], errors="coerce")
    frame["score_meta_base_soft_label"] = pd.to_numeric(frame["score_meta_base_soft_label"], errors="coerce")
    frame["net_ev_p90_spread_fee15bps"] = (
        frame["first_touch_gross"]
        - float(fee_round_trip_pct)
        - frame["p90_spread_bps"] / 10_000.0
    )
    before_score_valid = int(len(frame))
    valid = (
        np.isfinite(frame["first_touch_gross"])
        & np.isfinite(frame["p90_spread_bps"])
        & np.isfinite(frame["score_base"])
        & np.isfinite(frame["score_meta_base_soft_label"])
        & np.isfinite(frame["net_ev_p90_spread_fee15bps"])
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("No identical score-valid OOS rows remain after p90 cost stress")
    frame["month"] = frame["__ts__"].dt.tz_localize(None).dt.to_period("M").astype(str)
    frame["week_start"] = frame["__ts__"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time.astype(str)
    archetype_col = _archetype_column(set(frame.columns))
    if archetype_col is None:
        raise ValueError(
            "Predictions are missing the base archetype identity required for "
            "archetype and side x archetype reporting"
        )
    frame[archetype_col] = frame[archetype_col].astype(str).replace({"": "unknown", "nan": "unknown"})
    optional_sources = {
        metric: source
        for metric, candidates in OPTIONAL_METRIC_COLUMNS.items()
        if (source := _first_present(set(frame.columns), candidates)) is not None
    }
    metric_rows = _selector_metrics(
        frame,
        "score_base",
        selector="base_score",
        archetype_col=archetype_col,
        optional_sources=optional_sources,
    )
    metric_rows.extend(
        _selector_metrics(
            frame,
            "score_meta_base_soft_label",
            selector="meta_base_soft_label",
            archetype_col=archetype_col,
            optional_sources=optional_sources,
        )
    )
    metrics = pd.DataFrame(metric_rows)
    deltas = _delta_vs_base(metrics, archetype_col=archetype_col)
    integrity = {
        "prediction_rows_before_base_candidate_join": before_candidate_join,
        "base_candidate_rows_in_prediction_time_scope_before_p90": int(len(base_scope)),
        "p90_eligible_base_candidate_key_rows": int(len(base_keys)),
        "prediction_rows_after_base_candidate_join_before_score_validity": before_score_valid,
        "identical_score_valid_oos_rows": int(len(frame)),
        "dropped_for_nonfinite_base_or_meta_score_or_outcome": int(before_score_valid - len(frame)),
        "prediction_min_ts": prediction_min.isoformat(),
        "prediction_max_ts": prediction_max.isoformat(),
        "rank_scope": "timestamp_side",
        "rank_tie_breaker": "ascending_symbol_after_ascending_score; highest score retained",
        "base_candidate_selector_column": base_candidate_selector_column if base_candidate_selector_column in base_candidates.columns else None,
        "optional_metric_sources": optional_sources,
        "archetype_column": archetype_col,
        **p90_integrity,
    }
    provenance = {
        "schema": SCHEMA,
        "cost_formula": "first_touch_gross - fee_round_trip_pct - pooled_p90_full_spread_bps/10000",
        "fee_round_trip_pct": float(fee_round_trip_pct),
        "spread_quantile": float(spread_quantile),
        "selection_basis": "per_timestamp_x_side_top_fraction_on_identical_score_valid_oos_rows",
        "pooled_p90_disclosure": "Static post-hoc spread stress: p90 is pooled over the supplied history and is not a causal historical cost estimate.",
        "raw_side_score_disclosure": "Scores are ranked only within timestamp x side; raw long and short values are never globally compared.",
        "base_meta_row_contract": "Both selectors use exactly the same score-valid keys after joining the p90-eligible base candidate ledger.",
    }
    return EvaluationResult(metrics=metrics, deltas=deltas, eligible=p90, integrity=integrity, provenance=provenance)


def evaluate_paths(
    *,
    predictions_path: Path,
    base_candidate_ledger_path: Path,
    spread_history_path: Path,
    eligible_symbols: int,
    spread_quantile: float,
    fee_round_trip_pct: float,
    base_candidate_selector_column: str,
) -> EvaluationResult:
    prediction_columns = list(KEY_COLUMNS) + ["score_base", "score_meta_base_soft_label", "first_touch_gross"]
    prediction_available = _parquet_columns(predictions_path)
    prediction_columns.extend(
        column
        for column in (*ARCHETYPE_COLUMNS, *(source for candidates in OPTIONAL_METRIC_COLUMNS.values() for source in candidates))
        if column in prediction_available
    )
    predictions = _read_parquet_columns(predictions_path, list(dict.fromkeys(prediction_columns)))
    candidate_columns = list(KEY_COLUMNS)
    candidate_available = _parquet_columns(base_candidate_ledger_path)
    if base_candidate_selector_column in candidate_available:
        candidate_columns.append(base_candidate_selector_column)
    candidates = _read_parquet_columns(base_candidate_ledger_path, candidate_columns)
    spread = _read_parquet_columns(spread_history_path, ["observed_ts", "symbol", "spread_bps"])
    result = evaluate_frames(
        predictions,
        candidates,
        spread,
        eligible_symbols=int(eligible_symbols),
        spread_quantile=float(spread_quantile),
        fee_round_trip_pct=float(fee_round_trip_pct),
        base_candidate_selector_column=base_candidate_selector_column,
    )
    result.provenance.update(
        {
            "predictions_path": str(predictions_path),
            "predictions_sha256": _sha256_file(predictions_path),
            "base_candidate_ledger_path": str(base_candidate_ledger_path),
            "base_candidate_ledger_sha256": _sha256_file(base_candidate_ledger_path),
            "spread_history_path": str(spread_history_path),
            "spread_history_sha256": _sha256_file(spread_history_path),
        }
    )
    return result


def _write_result(result: EvaluationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out_dir / "metrics.csv", index=False)
    result.deltas.to_csv(out_dir / "delta_vs_base.csv", index=False)
    result.eligible.rename("p90_spread_bps").to_csv(out_dir / "eligible_symbols.csv")
    (out_dir / "integrity.json").write_text(json.dumps(result.integrity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "provenance.json").write_text(json.dumps(result.provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--base-candidate-ledger", type=Path, required=True)
    parser.add_argument("--spread-history", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eligible-symbols", type=int, default=170)
    parser.add_argument("--spread-quantile", type=float, default=0.90)
    parser.add_argument("--fee-round-trip-pct", type=float, default=0.0015)
    parser.add_argument("--base-candidate-selector-column", type=str, default="selected_top30")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = evaluate_paths(
        predictions_path=args.predictions,
        base_candidate_ledger_path=args.base_candidate_ledger,
        spread_history_path=args.spread_history,
        eligible_symbols=args.eligible_symbols,
        spread_quantile=args.spread_quantile,
        fee_round_trip_pct=args.fee_round_trip_pct,
        base_candidate_selector_column=args.base_candidate_selector_column,
    )
    _write_result(result, args.out_dir)


if __name__ == "__main__":
    main()
