#!/usr/bin/env python3
"""Compare OOS model and policy layers on strict canonical trade keys.

The inputs in the research pipeline are intentionally heterogeneous: a base
ledger can be a full scored universe, a residual-meta artifact can be a
candidate stream, and a portfolio replay can contain only accepted orders.
This reporter makes those differences explicit.  Every layer must declare its
selection semantics and cost provenance; deltas are calculated only over the
intersection of canonical UTC ``timestamp/symbol/side`` keys.

Typical invocation::

    python scripts/report_oos_layer_metrics.py \
      --base-ledger base.parquet --base-score-col score_base --base-top-frac .10 \
      --meta-predictions meta.parquet --meta-selected-col selected_top10 \
      --optimized-exit-replay exits.parquet --optimized-exit-selected-col accepted \
      --portfolio-decisions portfolio.parquet --portfolio-selected-col accepted \
      --cost-provenance base=fee30bps_plus_p90spread \
      --cost-provenance meta=fee30bps_plus_p90spread \
      --cost-provenance optimized_exit=fee30bps_plus_p90spread \
      --cost-provenance portfolio=fee30bps_plus_p90spread \
      --out-dir reports/oos_layers

No model fitting, ranking calibration, or outcome reconstruction occurs here.
It is a reporting-only tool and will fail closed on duplicate keys, missing
selection semantics, invalid timestamps, or side/archetype identity conflicts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


KEY_COLUMNS = ("__ts__", "__symbol__", "__side__")
KEY_SOURCE_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "__ts__": ("__ts__", "timestamp", "ts", "signal_timestamp", "decision_ts"),
    "__symbol__": ("__symbol__", "symbol", "instrument", "market"),
    "__side__": ("side_name", "side", "direction", "trade_side"),
}
ARCHETYPE_COLUMNS = (
    "archetype_label_family",
    "archetype_policy_key",
    "policy_archetype",
    "local_side_archetype",
    "source_archetype",
    "archetype",
)
SELECTION_COLUMNS = (
    "portfolio_accepted",
    "accepted",
    "admitted",
    "is_selected",
    "selected",
    "final_selected",
    "selected_top10",
    "selected_top20",
    "selected_top30",
)
RETURN_COLUMNS = (
    "notional_net_return",
    "position_net_return",
    "execution_net_return",
    "net_return_after_cost",
    "net_return",
    "ret_net",
    "net_ev_return",
    "ev_net_return",
    "ev_after_cost",
    "ev_after_1pct",
)
PORTFOLIO_RETURN_COLUMNS = (
    "portfolio_net_return",
    "bankroll_net_return",
    "portfolio_return",
    "portfolio_pnl_return",
)
PORTFOLIO_PNL_COLUMNS = (
    "portfolio_net_pnl",
    "position_net_pnl",
    "net_pnl",
    "realized_pnl",
    "pnl",
)
OUTCOME_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "clean_rate": ("clean_exec", "clean_exec_label", "clean_positive", "clean_path"),
    "dirty_positive_rate": ("dirty_positive", "dirty_positive_label"),
    "first_touch_bad_mae_rate": (
        "first_touch_bad_mae_1r",
        "first_touch_bad_mae",
    ),
    "full_path_bad_mae_rate": ("full_path_bad_mae_1r", "bad_mae_1r", "bad_mae"),
    "timeout_rate": ("timeout", "timeout_label", "is_timeout"),
    "stop_rate": ("full_stop_loss", "stop_loss", "stop_hit", "is_stop"),
}
EXIT_REASON_COLUMNS = ("position_exit_reason", "simple_policy_exit_reason", "exit_reason")
DELTA_COLUMNS = (
    "notional_net_return_per_trade",
    "sum_notional_net_return",
    "compounded_notional_return",
    "clean_rate",
    "dirty_positive_rate",
    "first_touch_bad_mae_rate",
    "full_path_bad_mae_rate",
    "timeout_rate",
    "stop_rate",
    "sum_portfolio_pnl",
    "compounded_portfolio_return",
)
GROUP_SPECS: Mapping[str, tuple[str, ...]] = {
    "overall": (),
    "month": ("month",),
    "week": ("week_start",),
    "side": ("side",),
    "archetype": ("archetype",),
    "week_side_archetype": ("week_start", "side", "archetype"),
}


@dataclass(frozen=True)
class LayerSpec:
    """One comparable stage in the causal base-to-portfolio chain."""

    name: str
    path: Path
    selected_col: str | None = None
    score_col: str | None = None
    top_frac: float | None = None
    allow_all_rows: bool = False
    return_col: str | None = None
    portfolio_return_col: str | None = None
    portfolio_pnl_col: str | None = None
    cost_provenance: str | None = None


@dataclass
class PreparedLayer:
    spec: LayerSpec
    rows: pd.DataFrame
    selected: pd.DataFrame
    key_sources: dict[str, str]
    archetype_source: str | None
    selection_basis: str
    return_col: str | None
    portfolio_return_col: str | None
    portfolio_pnl_col: str | None
    outcome_columns: dict[str, str]
    cost_provenance: str
    warnings: list[str] = field(default_factory=list)


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().str.lower()
    truthy = text.isin({"1", "true", "t", "yes", "y", "accepted", "selected"})
    falsy = text.isin({"0", "false", "f", "no", "n", "", "nan", "none"})
    unknown = numeric.isna() & ~truthy & ~falsy
    if bool(unknown.any()):
        examples = values.loc[unknown].astype(str).head(5).tolist()
        raise ValueError(f"Selection column contains unparseable values: {examples}")
    return (truthy | (numeric.fillna(0.0) != 0.0)).astype(bool)


def _canonical_side(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().str.lower()
    result = pd.Series(np.where(numeric > 0.0, "long", np.where(numeric < 0.0, "short", "")), index=values.index)
    result.loc[text.isin({"long", "buy", "bull", "1", "+1"})] = "long"
    result.loc[text.isin({"short", "sell", "bear", "-1"})] = "short"
    invalid = ~result.isin({"long", "short"})
    if bool(invalid.any()):
        examples = values.loc[invalid].astype(str).head(8).tolist()
        raise ValueError(f"Invalid side values; expected long/short or signed numeric: {examples}")
    return result.astype(str)


def _canonicalize(frame: pd.DataFrame, layer_name: str) -> tuple[pd.DataFrame, dict[str, str], str | None]:
    out = frame.copy()
    sources: dict[str, str] = {}
    for canonical, candidates in KEY_SOURCE_CANDIDATES.items():
        source = _first_present(out.columns, candidates)
        if source is None:
            raise ValueError(f"Layer {layer_name!r} has no source for canonical key {canonical}")
        sources[canonical] = source
    ts = pd.to_datetime(out[sources["__ts__"]], utc=True, errors="coerce")
    if bool(ts.isna().any()):
        examples = out.loc[ts.isna(), sources["__ts__"]].astype(str).head(5).tolist()
        raise ValueError(f"Layer {layer_name!r} has invalid UTC timestamps: {examples}")
    symbols = out[sources["__symbol__"]].astype(str).str.strip().str.upper()
    if bool(symbols.isin({"", "NAN", "NONE"}).any()):
        raise ValueError(f"Layer {layer_name!r} has missing symbols")
    out["__ts__"] = ts
    out["__symbol__"] = symbols
    out["__side__"] = _canonical_side(out[sources["__side__"]])
    duplicate = out.duplicated(list(KEY_COLUMNS), keep=False)
    if bool(duplicate.any()):
        examples = out.loc[duplicate, list(KEY_COLUMNS)].head(8).to_dict(orient="records")
        raise ValueError(f"Layer {layer_name!r} has duplicate canonical keys: {examples}")

    archetype_source = _first_present(out.columns, ARCHETYPE_COLUMNS)
    if archetype_source is None:
        out["__archetype__"] = "__missing__"
    else:
        archetype = out[archetype_source].astype(str).str.strip()
        out["__archetype__"] = archetype.mask(archetype.isin({"", "nan", "None"}), "__missing__")
    out["month"] = out["__ts__"].dt.strftime("%Y-%m")
    week = out["__ts__"].dt.floor("D") - pd.to_timedelta(out["__ts__"].dt.weekday, unit="D")
    out["week_start"] = week.dt.strftime("%Y-%m-%dT00:00:00Z")
    out["side"] = out["__side__"]
    out["archetype"] = out["__archetype__"]
    return out, sources, archetype_source


def _resolve_cost_provenance(frame: pd.DataFrame, spec: LayerSpec) -> str:
    if spec.cost_provenance:
        return str(spec.cost_provenance)
    for column in ("cost_provenance", "cost_contract", "round_trip_cost_contract"):
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(str).str.strip().unique().tolist()
        if len(values) == 1 and values[0]:
            return values[0]
        if len(values) > 1:
            raise ValueError(
                f"Layer {spec.name!r} has multiple {column!r} values; pass --cost-provenance explicitly"
            )
    raise ValueError(
        f"Layer {spec.name!r} has no explicit cost provenance. Pass --cost-provenance {spec.name}=..."
    )


def _select_rows(frame: pd.DataFrame, spec: LayerSpec) -> tuple[pd.DataFrame, str]:
    if spec.selected_col is not None:
        if spec.selected_col not in frame.columns:
            raise ValueError(f"Layer {spec.name!r} lacks selection column {spec.selected_col!r}")
        return frame.loc[_as_bool(frame[spec.selected_col])].copy(), f"explicit_column:{spec.selected_col}"
    inferred = _first_present(frame.columns, SELECTION_COLUMNS)
    if inferred is not None:
        return frame.loc[_as_bool(frame[inferred])].copy(), f"inferred_column:{inferred}"
    if spec.score_col is not None or spec.top_frac is not None:
        if spec.score_col is None or spec.top_frac is None:
            raise ValueError(f"Layer {spec.name!r} must provide both score_col and top_frac")
        if spec.score_col not in frame.columns:
            raise ValueError(f"Layer {spec.name!r} lacks score column {spec.score_col!r}")
        if not 0.0 < float(spec.top_frac) <= 1.0:
            raise ValueError(f"Layer {spec.name!r} top_frac must be in (0, 1]")
        score = _finite_numeric(frame[spec.score_col])
        valid = frame.loc[score.notna()].copy()
        valid["__selection_score__"] = score.loc[valid.index]
        n_rows = int(math.ceil(len(valid) * float(spec.top_frac)))
        ordered = valid.sort_values(
            ["__selection_score__", *KEY_COLUMNS],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        return ordered.head(n_rows).drop(columns="__selection_score__"), f"global_top_frac:{spec.score_col}:{spec.top_frac:g}"
    if spec.allow_all_rows:
        return frame.copy(), "explicit_all_rows"
    raise ValueError(
        f"Layer {spec.name!r} has no selection semantics. Supply a selected column, score/top-frac, or --allow-all-rows {spec.name}"
    )


def prepare_layer(spec: LayerSpec) -> PreparedLayer:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    frame = pd.read_parquet(spec.path)
    canonical, sources, archetype_source = _canonicalize(frame, spec.name)
    selected, selection_basis = _select_rows(canonical, spec)
    return_col = spec.return_col or _first_present(canonical.columns, RETURN_COLUMNS)
    portfolio_return_col = spec.portfolio_return_col or _first_present(canonical.columns, PORTFOLIO_RETURN_COLUMNS)
    portfolio_pnl_col = spec.portfolio_pnl_col or _first_present(canonical.columns, PORTFOLIO_PNL_COLUMNS)
    for configured in (return_col, portfolio_return_col, portfolio_pnl_col):
        if configured is not None and configured not in canonical.columns:
            raise ValueError(f"Layer {spec.name!r} references absent metric column {configured!r}")
    outcomes = {
        name: column
        for name, candidates in OUTCOME_COLUMNS.items()
        if (column := _first_present(canonical.columns, candidates)) is not None
    }
    return PreparedLayer(
        spec=spec,
        rows=canonical,
        selected=selected,
        key_sources=sources,
        archetype_source=archetype_source,
        selection_basis=selection_basis,
        return_col=return_col,
        portfolio_return_col=portfolio_return_col,
        portfolio_pnl_col=portfolio_pnl_col,
        outcome_columns=outcomes,
        cost_provenance=_resolve_cost_provenance(canonical, spec),
    )


def _rate(frame: pd.DataFrame, column: str | None) -> tuple[float, int]:
    if column is None or column not in frame.columns:
        return float("nan"), 0
    values = _finite_numeric(frame[column])
    return (float(values.mean()), int(values.notna().sum())) if values.notna().any() else (float("nan"), 0)


def _stop_rate(frame: pd.DataFrame, direct_column: str | None) -> tuple[float, int]:
    direct, support = _rate(frame, direct_column)
    if support:
        return direct, support
    reason_column = _first_present(frame.columns, EXIT_REASON_COLUMNS)
    if reason_column is None:
        return float("nan"), 0
    values = frame[reason_column].astype(str).str.lower()
    valid = ~values.isin({"", "nan", "none"})
    return (float(values.loc[valid].str.contains(r"(?:full_)?sl|stop|adverse", regex=True).mean()), int(valid.sum())) if bool(valid.any()) else (float("nan"), 0)


def _compounded(values: pd.Series) -> float:
    numeric = _finite_numeric(values).dropna()
    if numeric.empty:
        return float("nan")
    return float(np.prod(1.0 + numeric.to_numpy(dtype=np.float64)) - 1.0)


def metric_row(layer: PreparedLayer, rows: pd.DataFrame, scope: str, group_values: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate selected-row metrics without manufacturing missing outcomes."""
    result: dict[str, Any] = {
        "layer": layer.spec.name,
        "scope": scope,
        **group_values,
        "cost_provenance": layer.cost_provenance,
        "selection_basis": layer.selection_basis,
        "selected_rows": int(len(rows)),
        "selected_symbols": int(rows["__symbol__"].nunique()),
    }
    days = int(rows["__ts__"].dt.floor("D").nunique()) if not rows.empty else 0
    result["selected_days"] = days
    result["trades_per_day"] = float(len(rows) / days) if days else float("nan")

    if layer.return_col is not None:
        net = _finite_numeric(rows[layer.return_col])
        result["notional_return_column"] = layer.return_col
        result["notional_net_return_rows"] = int(net.notna().sum())
        result["notional_net_return_per_trade"] = float(net.mean()) if net.notna().any() else float("nan")
        result["sum_notional_net_return"] = float(net.sum()) if net.notna().any() else float("nan")
        result["compounded_notional_return"] = _compounded(net)
    else:
        result.update({
            "notional_return_column": None,
            "notional_net_return_rows": 0,
            "notional_net_return_per_trade": float("nan"),
            "sum_notional_net_return": float("nan"),
            "compounded_notional_return": float("nan"),
        })

    if layer.portfolio_return_col is not None:
        portfolio_return = _finite_numeric(rows[layer.portfolio_return_col])
        result["portfolio_return_column"] = layer.portfolio_return_col
        result["portfolio_return_rows"] = int(portfolio_return.notna().sum())
        result["compounded_portfolio_return"] = _compounded(portfolio_return)
    else:
        result.update({"portfolio_return_column": None, "portfolio_return_rows": 0, "compounded_portfolio_return": float("nan")})
    if layer.portfolio_pnl_col is not None:
        pnl = _finite_numeric(rows[layer.portfolio_pnl_col])
        result["portfolio_pnl_column"] = layer.portfolio_pnl_col
        result["portfolio_pnl_rows"] = int(pnl.notna().sum())
        result["sum_portfolio_pnl"] = float(pnl.sum()) if pnl.notna().any() else float("nan")
    else:
        result.update({"portfolio_pnl_column": None, "portfolio_pnl_rows": 0, "sum_portfolio_pnl": float("nan")})

    for metric, column in layer.outcome_columns.items():
        value, support = _rate(rows, column)
        result[metric] = value
        result[f"{metric}_rows"] = support
    for metric in OUTCOME_COLUMNS:
        result.setdefault(metric, float("nan"))
        result.setdefault(f"{metric}_rows", 0)
    stop_col = layer.outcome_columns.get("stop_rate")
    stop, stop_support = _stop_rate(rows, stop_col)
    result["stop_rate"] = stop
    result["stop_rate_rows"] = stop_support
    return result


def _subset_for_group(rows: pd.DataFrame, group_columns: Sequence[str], key: Any) -> pd.DataFrame:
    if not group_columns:
        return rows
    values = key if isinstance(key, tuple) else (key,)
    mask = pd.Series(True, index=rows.index)
    for column, value in zip(group_columns, values, strict=True):
        mask &= rows[column].eq(value)
    return rows.loc[mask]


def build_metric_table(layer: PreparedLayer, scope: str) -> pd.DataFrame:
    group_columns = GROUP_SPECS[scope]
    rows = layer.selected
    if not group_columns:
        return pd.DataFrame([metric_row(layer, rows, scope, {})])
    report: list[dict[str, Any]] = []
    for key, group in rows.groupby(list(group_columns), observed=True, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        report.append(metric_row(layer, group, scope, dict(zip(group_columns, values, strict=True))))
    return pd.DataFrame(report)


def _validate_overlap_archetypes(previous: pd.DataFrame, current: pd.DataFrame, previous_name: str, current_name: str) -> None:
    merged = previous.loc[:, [*KEY_COLUMNS, "__archetype__"]].merge(
        current.loc[:, [*KEY_COLUMNS, "__archetype__"]],
        on=list(KEY_COLUMNS),
        suffixes=("_previous", "_current"),
        validate="one_to_one",
    )
    mismatch = (
        ~merged["__archetype___previous"].eq("__missing__")
        & ~merged["__archetype___current"].eq("__missing__")
        & ~merged["__archetype___previous"].eq(merged["__archetype___current"])
    )
    if bool(mismatch.any()):
        examples = merged.loc[mismatch, [*KEY_COLUMNS, "__archetype___previous", "__archetype___current"]].head(8).to_dict(orient="records")
        raise ValueError(f"Archetype identity differs on overlapping rows: {previous_name!r} vs {current_name!r}: {examples}")


def build_delta_table(previous: PreparedLayer, current: PreparedLayer, scope: str) -> pd.DataFrame:
    """Compare adjacent layers using only rows selected by both stages."""
    _validate_overlap_archetypes(previous.selected, current.selected, previous.spec.name, current.spec.name)
    group_columns = GROUP_SPECS[scope]
    overlap_keys = current.selected.loc[:, [*KEY_COLUMNS, *group_columns]].merge(
        previous.selected.loc[:, list(KEY_COLUMNS)], on=list(KEY_COLUMNS), how="inner", validate="one_to_one"
    )
    previous_counts = build_metric_table(previous, scope)
    current_counts = build_metric_table(current, scope)
    count_key = list(group_columns)
    if count_key:
        previous_count_lookup = previous_counts.set_index(count_key)["selected_rows"]
        current_count_lookup = current_counts.set_index(count_key)["selected_rows"]
    else:
        previous_count_lookup = pd.Series({"__overall__": int(len(previous.selected))})
        current_count_lookup = pd.Series({"__overall__": int(len(current.selected))})

    groups: Iterable[tuple[Any, pd.DataFrame]]
    if group_columns:
        groups = overlap_keys.groupby(list(group_columns), observed=True, dropna=False, sort=True)
    else:
        groups = [((), overlap_keys)]
    rows: list[dict[str, Any]] = []
    costs_match = previous.cost_provenance == current.cost_provenance
    for key, keys in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        key_frame = keys.loc[:, list(KEY_COLUMNS)]
        previous_overlap = previous.selected.merge(key_frame, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
        current_overlap = current.selected.merge(key_frame, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
        prev_metrics = metric_row(previous, previous_overlap, scope, group_values)
        current_metrics = metric_row(current, current_overlap, scope, group_values)
        lookup_key: Any = key_values if len(group_columns) > 1 else (key_values[0] if group_columns else "__overall__")
        if group_columns and len(group_columns) > 1:
            lookup_key = key_values
        previous_selected = int(previous_count_lookup.loc[lookup_key]) if lookup_key in previous_count_lookup.index else 0
        current_selected = int(current_count_lookup.loc[lookup_key]) if lookup_key in current_count_lookup.index else 0
        row: dict[str, Any] = {
            "previous_layer": previous.spec.name,
            "layer": current.spec.name,
            "scope": scope,
            **group_values,
            "previous_cost_provenance": previous.cost_provenance,
            "cost_provenance": current.cost_provenance,
            "cost_comparable": bool(costs_match),
            "overlap_rows": int(len(key_frame)),
            "previous_selected_rows": previous_selected,
            "selected_rows": current_selected,
            "overlap_share_previous": float(len(key_frame) / previous_selected) if previous_selected else float("nan"),
            "overlap_share_current": float(len(key_frame) / current_selected) if current_selected else float("nan"),
        }
        for metric in DELTA_COLUMNS:
            if metric in {"notional_net_return_per_trade", "sum_notional_net_return", "compounded_notional_return", "sum_portfolio_pnl", "compounded_portfolio_return"} and not costs_match:
                row[f"delta_{metric}"] = float("nan")
            else:
                left = prev_metrics.get(metric, float("nan"))
                right = current_metrics.get(metric, float("nan"))
                row[f"delta_{metric}"] = float(right - left) if pd.notna(left) and pd.notna(right) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def generate_report(layers: Sequence[LayerSpec], out_dir: Path) -> dict[str, Any]:
    if len(layers) < 1:
        raise ValueError("At least one layer is required")
    names = [layer.name for layer in layers]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate layer names: {names}")
    prepared = [prepare_layer(spec) for spec in layers]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_outputs: dict[str, str] = {}
    delta_outputs: dict[str, str] = {}
    for scope in GROUP_SPECS:
        metrics = pd.concat([build_metric_table(layer, scope) for layer in prepared], ignore_index=True, sort=False)
        metrics_path = out_dir / f"oos_layer_metrics_{scope}.csv"
        metrics.to_csv(metrics_path, index=False)
        metrics_outputs[scope] = str(metrics_path)
        if len(prepared) > 1:
            deltas = pd.concat(
                [build_delta_table(previous, current, scope) for previous, current in zip(prepared, prepared[1:])],
                ignore_index=True,
                sort=False,
            )
            delta_path = out_dir / f"oos_layer_deltas_{scope}.csv"
            deltas.to_csv(delta_path, index=False)
            delta_outputs[scope] = str(delta_path)
    manifest = {
        "schema": "oos_layer_metrics_v1",
        "canonical_key": ["UTC timestamp", "symbol", "side"],
        "metric_tables": metrics_outputs,
        "delta_tables": delta_outputs,
        "layers": [
            {
                "name": layer.spec.name,
                "path": str(layer.spec.path),
                "rows": int(len(layer.rows)),
                "selected_rows": int(len(layer.selected)),
                "key_sources": layer.key_sources,
                "archetype_source": layer.archetype_source,
                "selection_basis": layer.selection_basis,
                "cost_provenance": layer.cost_provenance,
                "notional_return_column": layer.return_col,
                "portfolio_return_column": layer.portfolio_return_col,
                "portfolio_pnl_column": layer.portfolio_pnl_col,
                "outcome_columns": layer.outcome_columns,
                "warnings": layer.warnings,
            }
            for layer in prepared
        ],
        "delta_basis": "adjacent layers, selected-row canonical-key intersection only",
        "financial_delta_rule": "financial deltas are null when adjacent cost provenance differs",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_assignments(values: Sequence[str] | None, argument: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"{argument} expects LAYER=VALUE, got {value!r}")
        name, assigned = value.split("=", 1)
        if not name or not assigned:
            raise ValueError(f"{argument} expects LAYER=VALUE, got {value!r}")
        if name in parsed:
            raise ValueError(f"Duplicate {argument} layer assignment: {name!r}")
        parsed[name] = assigned
    return parsed


def _parse_float_assignments(values: Sequence[str] | None, argument: str) -> dict[str, float]:
    raw = _parse_assignments(values, argument)
    return {name: float(value) for name, value in raw.items()}


def _layers_from_args(args: argparse.Namespace) -> list[LayerSpec]:
    paths: list[tuple[str, Path]] = []
    fixed = (
        ("base", args.base_ledger),
        ("meta", args.meta_predictions),
        ("ev_admission", args.ev_admission),
        ("optimized_exit", args.optimized_exit_replay),
        ("portfolio", args.portfolio_decisions),
    )
    paths.extend((name, path) for name, path in fixed if path is not None)
    generic = _parse_assignments(args.layer, "--layer")
    paths.extend((name, Path(value)) for name, value in generic.items())
    if not paths:
        raise ValueError("Supply at least --base-ledger or --layer NAME=PATH")
    names = [name for name, _ in paths]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate input layer names: {names}")
    selected = _parse_assignments(args.selected_col, "--selected-col")
    score = _parse_assignments(args.score_col, "--score-col")
    top_frac = _parse_float_assignments(args.top_frac, "--top-frac")
    return_col = _parse_assignments(args.return_col, "--return-col")
    portfolio_return_col = _parse_assignments(args.portfolio_return_col, "--portfolio-return-col")
    portfolio_pnl_col = _parse_assignments(args.portfolio_pnl_col, "--portfolio-pnl-col")
    cost = _parse_assignments(args.cost_provenance, "--cost-provenance")
    known = set(names)
    for label, assignments in {
        "selected-col": selected,
        "score-col": score,
        "top-frac": top_frac,
        "return-col": return_col,
        "portfolio-return-col": portfolio_return_col,
        "portfolio-pnl-col": portfolio_pnl_col,
        "cost-provenance": cost,
    }.items():
        unknown = sorted(set(assignments) - known)
        if unknown:
            raise ValueError(f"--{label} names layers not supplied as inputs: {unknown}")
    allow_all = set(args.allow_all_rows or ())
    unknown_all = sorted(allow_all - known)
    if unknown_all:
        raise ValueError(f"--allow-all-rows names layers not supplied as inputs: {unknown_all}")
    return [
        LayerSpec(
            name=name,
            path=path,
            selected_col=selected.get(name),
            score_col=score.get(name),
            top_frac=top_frac.get(name),
            allow_all_rows=name in allow_all,
            return_col=return_col.get(name),
            portfolio_return_col=portfolio_return_col.get(name),
            portfolio_pnl_col=portfolio_pnl_col.get(name),
            cost_provenance=cost.get(name),
        )
        for name, path in paths
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ledger", type=Path)
    parser.add_argument("--meta-predictions", type=Path)
    parser.add_argument("--ev-admission", type=Path)
    parser.add_argument("--optimized-exit-replay", type=Path)
    parser.add_argument("--portfolio-decisions", type=Path)
    parser.add_argument("--layer", action="append", help="Additional ordered input: NAME=PATH")
    parser.add_argument("--selected-col", action="append", help="Selection flag: LAYER=COLUMN")
    parser.add_argument("--score-col", action="append", help="Score for deterministic global top fraction: LAYER=COLUMN")
    parser.add_argument("--top-frac", action="append", help="Top fraction with --score-col: LAYER=0.10")
    parser.add_argument("--allow-all-rows", action="append", help="Explicitly report every row for this LAYER")
    parser.add_argument("--return-col", action="append", help="Notional net return source: LAYER=COLUMN")
    parser.add_argument("--portfolio-return-col", action="append", help="Portfolio return source: LAYER=COLUMN")
    parser.add_argument("--portfolio-pnl-col", action="append", help="Portfolio PnL source: LAYER=COLUMN")
    parser.add_argument("--cost-provenance", action="append", help="Required cost provenance: LAYER=DESCRIPTION")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = generate_report(_layers_from_args(args), args.out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
