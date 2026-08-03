"""Pooled-global attribution and stability reporting for Stage-III OOF ledgers.

Every tail is selected exactly once over the eligible common-bps population.
Month, week, side and day tables are *attributions of that unchanged set*;
they never re-rank locally.  This is intentionally separate from portfolio PnL,
which requires execution sizing and constraint replay rather than row returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "stage_iii_pooled_global_reporting_v1"


class StageIIIReportingError(ValueError):
    """Raised when a report would violate identity, cost, or ranking contracts."""


@dataclass(frozen=True)
class StageIIIReportingConfig:
    candidate_id_column: str = "candidate_id"
    symbol_column: str = "symbol"
    timestamp_column: str = "decision_ts"
    side_column: str = "side_name"
    exact_net_column: str = "exact_net_bps"
    exact_gross_column: str = "exact_gross_bps"
    admission_column: str = "causal_21d_admitted"
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    total_cost_bps: float = 100.0
    surprise_columns: Mapping[str, tuple[str, str]] = field(default_factory=lambda: {
        "3d": ("hit_rate_surprise_3d", "hit_rate_surprise_support_3d"),
        "7d": ("hit_rate_surprise_7d", "hit_rate_surprise_support_7d"),
        "14d": ("hit_rate_surprise_14d", "hit_rate_surprise_support_14d"),
    })
    require_hit_surprise: bool = True

    def validate(self) -> None:
        if not self.top_fractions or any(not 0.0 < value <= 1.0 for value in self.top_fractions):
            raise StageIIIReportingError("top fractions must lie in (0, 1]")
        if tuple(sorted(set(self.top_fractions))) != self.top_fractions:
            raise StageIIIReportingError("top fractions must be unique and increasing")
        if self.total_cost_bps <= 0:
            raise StageIIIReportingError("total_cost_bps must be positive")


@dataclass(frozen=True)
class StageIIIReportTables:
    schema: str
    tail_summary: pd.DataFrame
    selected_attribution: pd.DataFrame
    residual_diagnostics: pd.DataFrame
    time_concentration: pd.DataFrame
    hit_surprise: pd.DataFrame


def _canonical_boolean(series: pd.Series, *, name: str) -> np.ndarray:
    parsed: list[bool] = []
    for value in series.to_numpy(dtype=object):
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            parsed.append(bool(value))
        else:
            raise StageIIIReportingError(f"{name} must contain only bool or integer 0/1")
    return np.asarray(parsed, dtype=bool)


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        raise StageIIIReportingError(f"report ledger lacks {name!r}")
    value = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(value).all():
        raise StageIIIReportingError(f"{name!r} must be finite")
    return value


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(pd.Series(left).corr(pd.Series(right), method="spearman"))


def _lag_one_autocorrelation(value: np.ndarray) -> float:
    if len(value) < 3 or np.std(value[:-1]) <= 0 or np.std(value[1:]) <= 0:
        return float("nan")
    return float(np.corrcoef(value[:-1], value[1:])[0, 1])


def _validate_ledger(
    frame: pd.DataFrame,
    *,
    score_columns: Mapping[str, str],
    config: StageIIIReportingConfig,
) -> pd.DataFrame:
    config.validate()
    required = {
        config.candidate_id_column, config.symbol_column, config.timestamp_column,
        config.side_column, config.exact_net_column, config.exact_gross_column,
        config.admission_column, *score_columns.values(),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise StageIIIReportingError(f"report ledger lacks columns: {missing[:12]}")
    work = frame.copy()
    timestamp = pd.to_datetime(work[config.timestamp_column], utc=True, errors="coerce")
    if timestamp.isna().any():
        raise StageIIIReportingError("decision timestamps must be valid UTC values")
    work[config.timestamp_column] = timestamp
    candidate = work[config.candidate_id_column].astype("string")
    symbol = work[config.symbol_column].astype("string")
    side = work[config.side_column].astype("string").str.lower()
    if candidate.isna().any() or candidate.str.strip().eq("").any():
        raise StageIIIReportingError("candidate identity must be non-empty")
    if symbol.isna().any() or symbol.str.strip().eq("").any():
        raise StageIIIReportingError("symbol identity must be non-empty")
    if not set(side.unique()).issubset({"long", "short"}):
        raise StageIIIReportingError("side must use canonical long/short values")
    identity = pd.DataFrame({"candidate": candidate, "symbol": symbol, "timestamp": timestamp, "side": side})
    if identity.duplicated().any():
        raise StageIIIReportingError("candidate/symbol/time/side identity must be unique")
    net = _numeric(work, config.exact_net_column)
    gross = _numeric(work, config.exact_gross_column)
    if not np.allclose(gross - config.total_cost_bps, net, rtol=0.0, atol=1e-5):
        raise StageIIIReportingError("gross minus declared cost must equal exact net exactly once")
    _canonical_boolean(work[config.admission_column], name=config.admission_column)
    for layer, score in score_columns.items():
        if not str(layer).strip():
            raise StageIIIReportingError("layer names must be non-empty")
        _numeric(work, score)
    if config.require_hit_surprise:
        for horizon, (value, support) in config.surprise_columns.items():
            if value not in work or support not in work:
                raise StageIIIReportingError(f"missing signed hit-surprise fields for {horizon}")
            _numeric(work, value)
            support_values = _numeric(work, support)
            if (support_values < 0).any():
                raise StageIIIReportingError("hit-surprise support must be non-negative")
    work["__side"] = side
    work["__month"] = timestamp.dt.strftime("%Y-%m")
    work["__week"] = timestamp.dt.strftime("%G-W%V")
    work["__day"] = timestamp.dt.strftime("%Y-%m-%d")
    work["__identity"] = (
        candidate.astype(str) + "|" + symbol.astype(str) + "|"
        + timestamp.astype(str) + "|" + side.astype(str)
    )
    return work.sort_values([config.timestamp_column, "__identity"], kind="stable").reset_index(drop=True)


def build_stage_iii_report_tables(
    frame: pd.DataFrame,
    *,
    score_columns: Mapping[str, str],
    config: StageIIIReportingConfig = StageIIIReportingConfig(),
) -> StageIIIReportTables:
    """Build layer/admission reports with one pooled-global selection per tail."""
    work = _validate_ledger(frame, score_columns=score_columns, config=config)
    admission = _canonical_boolean(work[config.admission_column], name=config.admission_column)
    summary_rows: list[dict[str, object]] = []
    attribution_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    surprise_rows: list[dict[str, object]] = []
    net_column = config.exact_net_column
    gross_column = config.exact_gross_column

    for layer, score_column in score_columns.items():
        for admission_scope, mask in (
            ("without_21d", np.ones(len(work), dtype=bool)),
            ("with_21d", admission),
        ):
            population = work.loc[mask].copy()
            if population.empty:
                raise StageIIIReportingError(f"{admission_scope} leaves no candidates")
            population = population.sort_values(
                [score_column, "__identity"], ascending=[False, True], kind="stable"
            )
            for fraction in config.top_fractions:
                selected_rows = max(1, int(np.ceil(fraction * len(population))))
                selected = population.head(selected_rows).copy()
                net = selected[net_column].to_numpy(float)
                gross = selected[gross_column].to_numpy(float)
                score = selected[score_column].to_numpy(float)
                common = {
                    "layer": str(layer), "score_column": str(score_column),
                    "admission_scope": admission_scope, "top_fraction": float(fraction),
                    "ranking_basis": "pooled_global_after_common_bps_mapping",
                    "candidate_rows": int(len(population)), "selected_rows": int(len(selected)),
                }
                summary_rows.append({
                    **common, "gross_bps_per_trade": float(gross.mean()),
                    "net_bps_per_trade": float(net.mean()),
                    "cost_bps_per_trade": float((gross - net).mean()),
                    "gross_bps_sum": float(gross.sum()),
                    "net_bps_sum": float(net.sum()), "positive_net_rate": float((net > 0).mean()),
                    "score_net_spearman": _spearman(score, net),
                })
                for dimensions, columns in (
                    ("month", ["__month"]), ("week", ["__week"]),
                    ("side", ["__side"]), ("month_side", ["__month", "__side"]),
                    ("week_side", ["__week", "__side"]),
                ):
                    for keys, local in selected.groupby(columns, sort=True, observed=True):
                        key_values = keys if isinstance(keys, tuple) else (keys,)
                        record = {
                            **common, "scope": dimensions, "month": "__all__",
                            "week": "__all__", "side": "__all__",
                            "selected_rows": int(len(local)),
                            "gross_bps_per_trade": float(local[gross_column].mean()),
                            "net_bps_per_trade": float(local[net_column].mean()),
                            "cost_bps_per_trade": float(
                                (local[gross_column] - local[net_column]).mean()
                            ),
                            "gross_bps_sum": float(local[gross_column].sum()),
                            "net_bps_sum": float(local[net_column].sum()),
                        }
                        for name, value in zip(columns, key_values, strict=True):
                            record[{"__month": "month", "__week": "week", "__side": "side"}[name]] = str(value)
                        attribution_rows.append(record)
                # ``selected`` is ordered by model score for pooled-global tail
                # selection.  Serial diagnostics must instead use event time;
                # score order creates a fictitious autocorrelation statistic.
                for side_name, local in [("__all__", selected), *list(selected.groupby("__side", sort=True, observed=True))]:
                    local_time_ordered = local.sort_values(
                        [config.timestamp_column, "__identity"], kind="stable"
                    )
                    local_residual = (
                        local_time_ordered[net_column] - local_time_ordered[score_column]
                    ).to_numpy(float)
                    residual_rows.append({
                        **common, "side": str(side_name), "rows": int(len(local)),
                        "signed_residual_mean_bps": float(local_residual.mean()),
                        "signed_residual_std_bps": float(local_residual.std(ddof=0)),
                        "signed_residual_lag1_autocorrelation": _lag_one_autocorrelation(local_residual),
                    })
                daily = selected.groupby("__day", sort=True, observed=True).size().to_numpy(float)
                weekly = selected.groupby("__week", sort=True, observed=True).size().to_numpy(float)
                concentration_rows.append({
                    **common, "active_days": int(len(daily)),
                    "trades_per_active_day": float(len(selected) / max(len(daily), 1)),
                    "max_day_share": float(daily.max() / daily.sum()),
                    "day_hhi": float(np.square(daily / daily.sum()).sum()),
                    "max_week_share": float(weekly.max() / weekly.sum()),
                    "week_hhi": float(np.square(weekly / weekly.sum()).sum()),
                    "unique_symbols": int(selected[config.symbol_column].nunique()),
                    "max_symbol_share": float(selected[config.symbol_column].value_counts(normalize=True).max()),
                })
                if config.require_hit_surprise:
                    for horizon, (value_column, support_column) in config.surprise_columns.items():
                        values = selected[value_column].to_numpy(float)
                        support = selected[support_column].to_numpy(float)
                        weighted = float(np.dot(values, support) / support.sum()) if support.sum() > 0 else np.nan
                        surprise_rows.append({
                            **common, "horizon": horizon,
                            "signed_surprise_mean": float(values.mean()),
                            "signed_surprise_support_weighted_mean": weighted,
                            "effective_support_sum": float(support.sum()),
                            "positive_surprise_rate": float((values > 0).mean()),
                        })

    return StageIIIReportTables(
        schema=SCHEMA,
        tail_summary=pd.DataFrame(summary_rows),
        selected_attribution=pd.DataFrame(attribution_rows),
        residual_diagnostics=pd.DataFrame(residual_rows),
        time_concentration=pd.DataFrame(concentration_rows),
        hit_surprise=pd.DataFrame(surprise_rows),
    )


__all__ = [
    "SCHEMA", "StageIIIReportTables", "StageIIIReportingConfig",
    "StageIIIReportingError", "build_stage_iii_report_tables",
]
