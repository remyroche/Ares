#!/usr/bin/env python3
"""Report the four weighted-PackB July replay layers from CSV or Parquet inputs.

The raw-meta and EV-mapped candidate tables are independently selected by one
global OOS top-fraction cutoff. Policy execution outcomes are
already post-admission. Portfolio decisions are restricted to an explicit
accepted/traded field. Outcome and exit fields are optional: unavailable values
remain NaN instead of being converted to zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_LAYER = "raw_meta_top10_without_ev_map"
EV_LAYER = "ev_mapped_top10"
POLICY_LAYER = "policy_before_portfolio"
PORTFOLIO_LAYER = "portfolio_after_constraints"
LAYER_ORDER = (RAW_LAYER, EV_LAYER, POLICY_LAYER, PORTFOLIO_LAYER)

ALIASES = {
    "timestamp": ("timestamp", "__ts__", "decision_timestamp"),
    "side": ("side_name", "side"),
    "archetype": (
        "archetype_label_family",
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
        "archetype_policy_key",
        "archetype",
    ),
    "net_return": (
        "position_net_return",
        "net_return",
        "ret_net_notional",
        "ev_after_1pct",
        "outcome_net_return",
    ),
    "notional_pnl": ("portfolio_net_pnl", "notional_pnl", "position_pnl", "net_pnl"),
    "exit_reason": ("position_exit_reason", "simple_policy_exit_reason", "exit_reason"),
    "stop": ("full_sl", "is_stop", "stop", "__first_touch_stop__"),
    "trailing": ("is_trailing", "trailing", "trailing_exit"),
    "timeout": ("is_timeout", "timeout", "__first_touch_timeout__"),
}
SCORE_ALIASES = {
    RAW_LAYER: (
        "raw_meta_score_rank",
        "raw_meta_score",
        "rank_meta_direct",
        "meta_score",
        "score_meta",
        "policy_parent_rank",
        "rank_pct",
    ),
    EV_LAYER: (
        "mapped_expected_ev_rank",
        "mapped_expected_ev",
        "expected_ev_rank_score",
        "rank_mlp_direct",
        "score_base_ev_residual_expert_hier_mapped",
        "ev_mapped_score",
        "calibrated_ev",
    ),
}
PORTFOLIO_INCLUDE_ALIASES = ("accepted", "portfolio_accepted", "was_traded", "portfolio_decision")
POLICY_INCLUDE_ALIASES = ("policy_admitted_before_portfolio", "threshold_basis_selected")
RAW_INCLUDE_ALIASES = ("raw_global_top10_selected",)
EV_INCLUDE_ALIASES = ("ev_mapped_global_top10_selected",)


@dataclass(frozen=True)
class Mapping:
    layer: str
    timestamp: str
    side: str
    archetype: str
    score: str | None
    net_return: str | None
    notional_pnl: str | None
    exit_reason: str | None
    stop: str | None
    trailing: str | None
    timeout: str | None
    include: str | None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type for {path}; use CSV or Parquet.")


def _resolve(
    frame: pd.DataFrame,
    *,
    layer: str,
    field: str,
    override: str | None,
    aliases: Iterable[str],
    required: bool,
) -> str | None:
    if override is not None:
        if override not in frame.columns:
            raise ValueError(f"{layer}: {field} override maps to missing column {override!r}")
        return override
    column = next((alias for alias in aliases if alias in frame.columns), None)
    if column is None and required:
        raise ValueError(
            f"{layer}: no exact mapping for required {field}; tried {list(aliases)!r}. "
            f"Pass --{layer.replace('_', '-')}-{field.replace('_', '-')}-col."
        )
    return column


def _mapping(frame: pd.DataFrame, layer: str, overrides: dict[str, str | None]) -> Mapping:
    fields = {
        field: _resolve(
            frame,
            layer=layer,
            field=field,
            override=overrides[field],
            aliases=aliases,
            required=field in {"timestamp", "side", "archetype"},
        )
        for field, aliases in ALIASES.items()
    }
    fields["score"] = _resolve(
        frame,
        layer=layer,
        field="score",
        override=overrides["score"],
        aliases=SCORE_ALIASES.get(layer, ()),
        required=layer in {RAW_LAYER, EV_LAYER},
    )
    fields["include"] = _resolve(
        frame,
        layer=layer,
        field="include",
        override=overrides["include"],
        aliases=(
            PORTFOLIO_INCLUDE_ALIASES
            if layer == PORTFOLIO_LAYER
            else POLICY_INCLUDE_ALIASES
            if layer == POLICY_LAYER
            else RAW_INCLUDE_ALIASES
            if layer == RAW_LAYER
            else EV_INCLUDE_ALIASES
            if layer == EV_LAYER
            else ()
        ),
        # A policy input may already be pre-filtered by the caller.  When the
        # admission flag is present we use it automatically; otherwise retain
        # the supplied policy rows.  Portfolio acceptance remains mandatory.
        required=layer == PORTFOLIO_LAYER,
    )
    return Mapping(layer=layer, **fields)  # type: ignore[arg-type]


def _side(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip().str.lower()
    return result.replace({"1": "long", "+1": "long", "1.0": "long", "-1": "short", "-1.0": "short"})


def _flag(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    numeric = pd.to_numeric(values, errors="coerce")
    result.loc[numeric.eq(1.0)] = 1.0
    result.loc[numeric.eq(0.0)] = 0.0
    text = values.astype("string").str.strip().str.lower()
    result.loc[text.isin({"true", "t", "yes", "y", "accepted", "traded", "would_trade", "shadow_traded"})] = 1.0
    result.loc[text.isin({"false", "f", "no", "n", "rejected", "rank_rejected", "blocked"})] = 0.0
    return result


def _reason_flag(reason: pd.Series, category: str) -> pd.Series:
    text = reason.astype("string").str.strip().str.lower()
    known = text.notna() & text.ne("")
    result = pd.Series(np.nan, index=reason.index, dtype="float64")
    if category == "stop":
        matched = text.str.contains(r"stop|full_sl|(?:^|_)sl(?:_|$)", regex=True, na=False)
    elif category == "trailing":
        matched = text.str.contains("trail", regex=False, na=False)
    elif category == "timeout":
        matched = text.str.contains("timeout", regex=False, na=False)
    else:  # pragma: no cover - fixed callers below
        raise ValueError(category)
    result.loc[known] = matched.loc[known].astype(float)
    return result


def _top_fraction(frame: pd.DataFrame, score: pd.Series, fraction: float) -> pd.Series:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"--top-fraction must be in (0, 1], got {fraction}")
    selected = pd.Series(False, index=frame.index)
    numeric = pd.to_numeric(score, errors="coerce")
    valid = numeric.dropna()
    if not valid.empty:
        count = max(1, int(np.ceil(len(valid) * fraction)))
        selected.loc[valid.sort_values(ascending=False, kind="mergesort").index[:count]] = True
    return selected


def build_layer(frame: pd.DataFrame, mapping: Mapping, top_fraction: float | None) -> pd.DataFrame:
    timestamp = pd.to_datetime(frame[mapping.timestamp], utc=True, errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"{mapping.layer}: {int(timestamp.isna().sum())} invalid UTC timestamps")
    archetype = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in ALIASES["archetype"]:
        if column not in frame:
            continue
        candidate = frame[column].astype("string").str.strip()
        valid = candidate.notna() & ~candidate.str.lower().isin(("", "nan", "none", "null"))
        archetype = archetype.mask(archetype.isna() & valid, candidate)
    archetype = archetype.fillna("missing")
    rows = pd.DataFrame(
        {
            "timestamp": timestamp,
            "side": _side(frame[mapping.side]),
            "archetype": archetype,
            "net_return": pd.to_numeric(frame[mapping.net_return], errors="coerce") if mapping.net_return else np.nan,
            "notional_pnl": pd.to_numeric(frame[mapping.notional_pnl], errors="coerce") if mapping.notional_pnl else np.nan,
        },
        index=frame.index,
    )
    reason = frame[mapping.exit_reason] if mapping.exit_reason else pd.Series(pd.NA, index=frame.index, dtype="string")
    for category in ("stop", "trailing", "timeout"):
        source = getattr(mapping, category)
        rows[f"{category}_flag"] = _flag(frame[source]) if source else _reason_flag(reason, category)
    if mapping.include:
        rows = rows.loc[_flag(frame[mapping.include]).eq(1.0)].copy()
    if top_fraction is not None:
        assert mapping.score is not None
        rows = rows.loc[_top_fraction(rows, frame.loc[rows.index, mapping.score], top_fraction)].copy()
    rows.insert(0, "layer", mapping.layer)
    return rows.reset_index(drop=True)


def _rate(values: pd.Series) -> float:
    known = values.dropna()
    return float(known.mean()) if not known.empty else float("nan")


def _count(values: pd.Series) -> float:
    known = values.dropna()
    return float(known.sum()) if not known.empty else float("nan")


def _worst_day(rows: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(rows[column], errors="coerce")
    daily = pd.DataFrame({"day": rows["timestamp"].dt.floor("D"), "value": values}).dropna(subset=["value"])
    return float(daily.groupby("day", sort=False)["value"].sum().min()) if not daily.empty else float("nan")


def _metrics(rows: pd.DataFrame) -> dict[str, float | int | str]:
    net = pd.to_numeric(rows["net_return"], errors="coerce")
    pnl = pd.to_numeric(rows["notional_pnl"], errors="coerce")
    outcome = net if net.notna().any() else pnl
    basis = "net_return" if net.notna().any() else "notional_pnl" if pnl.notna().any() else "missing"
    valid_outcome = outcome.dropna()
    days = int(rows["timestamp"].dt.floor("D").nunique())
    sides = rows.loc[rows["side"].isin({"long", "short"}), "side"]
    return {
        "rows": int(len(rows)),
        "trades": int(len(rows)),
        "days": days,
        "trades_per_day": float(len(rows) / days) if days else float("nan"),
        "outcome_basis": basis,
        "mean_net_return": float(net.mean()) if net.notna().any() else float("nan"),
        "sum_net_return": float(net.sum(min_count=1)),
        "mean_notional_pnl": float(pnl.mean()) if pnl.notna().any() else float("nan"),
        "sum_notional_pnl": float(pnl.sum(min_count=1)),
        "hit_rate": float((valid_outcome > 0.0).mean()) if not valid_outcome.empty else float("nan"),
        "worst_day_net_return": _worst_day(rows, "net_return"),
        "worst_day_notional_pnl": _worst_day(rows, "notional_pnl"),
        "long_share": float(sides.eq("long").mean()) if not sides.empty else float("nan"),
        "short_share": float(sides.eq("short").mean()) if not sides.empty else float("nan"),
        "stop_rate": _rate(rows["stop_flag"]),
        "stop_trades": _count(rows["stop_flag"]),
        "trailing_rate": _rate(rows["trailing_flag"]),
        "trailing_trades": _count(rows["trailing_flag"]),
        "timeout_rate": _rate(rows["timeout_flag"]),
        "timeout_trades": _count(rows["timeout_flag"]),
    }


def summarise(rows: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    records = []
    for key, group in rows.groupby(groups, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        records.append(dict(zip(groups, values, strict=True)) | _metrics(group))
    return pd.DataFrame(records)


def _add_mapping_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    for field in (*ALIASES, "score", "include"):
        parser.add_argument(f"--{prefix}-{field.replace('_', '-')}-col")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-meta", type=Path, required=True)
    parser.add_argument("--ev-mapped", type=Path, required=True)
    parser.add_argument("--policy-execution", type=Path, required=True)
    parser.add_argument("--portfolio-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    for prefix in ("raw-meta", "ev-mapped", "policy", "portfolio"):
        _add_mapping_args(parser, prefix)
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace, prefix: str) -> dict[str, str | None]:
    return {field: getattr(args, f"{prefix}_{field}_col") for field in (*ALIASES, "score", "include")}


def build_report(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    specs = (
        (RAW_LAYER, args.raw_meta, "raw_meta", args.top_fraction),
        (EV_LAYER, args.ev_mapped, "ev_mapped", args.top_fraction),
        (POLICY_LAYER, args.policy_execution, "policy", None),
        (PORTFOLIO_LAYER, args.portfolio_decisions, "portfolio", None),
    )
    mappings: list[Mapping] = []
    layers = []
    for layer, path, prefix, fraction in specs:
        mapping = _mapping(_read_table(path), layer, _overrides(args, prefix))
        frame = _read_table(path)
        mappings.append(mapping)
        # A persisted selection mask is authoritative. Applying top-fraction
        # again after filtering a replay union would select only 10% of the
        # already selected rows and change the experiment denominator.
        layers.append(build_layer(frame, mapping, None if mapping.include else fraction))
    rows = pd.concat(layers, ignore_index=True, copy=False)
    rows["utc_week_start"] = rows["timestamp"].dt.normalize() - pd.to_timedelta(rows["timestamp"].dt.dayofweek, unit="D")
    rows["utc_day"] = rows["timestamp"].dt.floor("D")
    overall = pd.DataFrame({"layer": LAYER_ORDER}).merge(summarise(rows, ["layer"]), on="layer", how="left")
    overall[["rows", "trades", "days"]] = overall[["rows", "trades", "days"]].fillna(0).astype(int)
    return {
        "overall": overall,
        "by_utc_day": summarise(rows, ["layer", "utc_day"]),
        "by_utc_week": summarise(rows, ["layer", "utc_week_start"]),
        "by_archetype": summarise(rows, ["layer", "archetype"]),
        "by_utc_week_archetype": summarise(rows, ["layer", "utc_week_start", "archetype"]),
        "by_side": summarise(rows, ["layer", "side"]),
        "field_mappings": pd.DataFrame([{field: getattr(mapping, field) for field in Mapping.__dataclass_fields__} for mapping in mappings]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in build_report(args).items():
        table.to_csv(args.output_dir / f"weighted_packb_july_replay_layers_{name}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
