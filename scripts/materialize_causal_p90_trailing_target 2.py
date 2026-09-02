#!/usr/bin/env python3
"""Materialize a small keyed sidecar for the p90-spread trailing target.

The sidecar avoids rewriting the large causal label store.  It is keyed by the
same UTC timestamp, symbol and side used by the base train/replay contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.trailing_cost_target import (  # noqa: E402
    CausalSpreadP90Spec,
KEY_COLUMNS,
    build_trailing_cost_targets,
    causal_p90_spread_cost,
    pooled_asset_p90_spread_cost,
    target_contract_manifest,
)


DEFAULT_LABELS = Path("data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels")
DEFAULT_SPREAD_HISTORY = Path("data_perp/exchanges/krakenfutures/spread_snapshots/history.parquet")
DEFAULT_OUTPUT = Path("data_perp/artifacts/20260721_causal_p90_trailing_target_v1/target_sidecar.parquet")


def _label_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No label parquet files under {path}")
    return files


def materialize(
    *,
    labels_path: Path,
    spread_history_path: Path,
    output_path: Path,
    spec: CausalSpreadP90Spec,
    cost_estimator: str = "causal_rolling_p90",
    blend_long: float | None = None,
    blend_short: float | None = None,
    write_diagnostics: bool = False,
) -> dict:
    needed = list(KEY_COLUMNS) + [
        "__first_touch_target_soft__",
        "__first_touch_capture_net__",
        "__first_touch_round_trip_cost__",
        "__trailing_profit_activated__",
        "__trailing_profit_activation_bar__",
        "__first_touch_timeout__",
        "__is_timeout__",
        "__first_touch_full_path_mae_norm__",
        "__archetype_label_family__",
        "archetype_label_family",
        "policy_archetype",
    ]
    chunks: list[pd.DataFrame] = []
    for path in _label_files(labels_path):
        # Inspect only parquet metadata. Loading a full label shard here would
        # defeat the sidecar's purpose and can consume multiple GB per shard.
        available = set(pq.ParquetFile(path).schema.names)
        required = [column for column in needed if column in available]
        missing = set(KEY_COLUMNS).difference(required)
        if missing:
            raise ValueError(f"{path} missing keys={sorted(missing)}")
        chunk = pd.read_parquet(path, columns=required)
        for column in needed:
            if column not in chunk.columns:
                chunk[column] = 0.0
        chunks.append(chunk.loc[:, needed])
    rows = pd.concat(chunks, ignore_index=True, copy=False)
    archetype_candidates = (
        "__archetype_label_family__",
        "archetype_label_family",
        "policy_archetype",
    )
    archetype = pd.Series("unknown", index=rows.index, dtype="string")
    for column in archetype_candidates:
        values = rows[column].astype("string")
        present = values.notna() & values.ne("0") & values.ne("")
        archetype = archetype.where(~present, values)
    history = pd.read_parquet(spread_history_path, columns=["observed_ts", "symbol", "spread_bps"])
    estimator = str(cost_estimator).strip().lower()
    if estimator == "causal_rolling_p90":
        cost = causal_p90_spread_cost(rows, history, spec=spec)
    elif estimator == "pooled_asset_p90":
        cost = pooled_asset_p90_spread_cost(rows, history, spec=spec)
    else:
        raise ValueError(f"Unsupported cost_estimator={cost_estimator!r}")
    from extreme_price_movements.trailing_cost_target import (
        DEFAULT_TARGET_SPECS,
        TrailingCostTargetSpec,
    )

    target_specs = dict(DEFAULT_TARGET_SPECS)
    for side, blend in (("long", blend_long), ("short", blend_short)):
        if blend is not None:
            if not 0.0 <= float(blend) <= 1.0:
                raise ValueError("target blend must lie in [0, 1]")
            current = target_specs[side]
            target_specs[side] = TrailingCostTargetSpec(
                margin=current.margin,
                temperature=current.temperature,
                blend=float(blend),
                activation_bonus=current.activation_bonus,
                slow_timeout_penalty=current.slow_timeout_penalty,
                adverse_path_penalty=current.adverse_path_penalty,
            )
    target = build_trailing_cost_targets(rows, cost, specs=target_specs)
    result = pd.concat(
        [
            rows.loc[:, list(KEY_COLUMNS)],
            archetype.rename("archetype_label_family"),
            cost,
            target,
        ],
        axis=1,
        copy=False,
    )
    result["side"] = (
        result["side_name"].astype(str).str.lower().eq("short").map({True: -1, False: 1})
        .astype("int8")
    )
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Target sidecar has duplicate UTC timestamp/symbol/side keys")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="zstd", compression_level=9)
    diagnostics_path = output_path.with_suffix(".side_archetype_diagnostics.csv")
    if write_diagnostics:
        # Diagnostics are useful but need not be materialized during normal
        # target construction. Avoid a second wide 4.5M-row copy by grouping
        # only the compact report columns.
        report = result.loc[:, [
            "__ts__", "side_name", "archetype_label_family", "p90_cost_observed",
            "target_soft_incumbent", "target_soft_p90_trailing_blend",
            "capture_net_p90_spread_fee30bps",
        ]].copy()
        report["month"] = pd.to_datetime(report.pop("__ts__"), utc=True).dt.strftime("%Y-%m")
        report["side_name"] = report["side_name"].astype("category")
        report["archetype_label_family"] = report["archetype_label_family"].astype("category")
        diagnostics = report.groupby(
            ["month", "side_name", "archetype_label_family"], dropna=False, observed=True
        ).agg(
            rows=("p90_cost_observed", "size"),
            cost_observed_rows=("p90_cost_observed", "sum"),
            cost_observed_fraction=("p90_cost_observed", "mean"),
            incumbent_target_mean=("target_soft_incumbent", "mean"),
            challenger_target_mean=("target_soft_p90_trailing_blend", "mean"),
            cost_aware_net_mean=("capture_net_p90_spread_fee30bps", "mean"),
        ).reset_index()
        diagnostics.to_csv(diagnostics_path, index=False)
    manifest = target_contract_manifest(
        cost_spec=spec,
        target_specs=target_specs,
        rows=len(result),
        observed_rows=int(result["p90_cost_observed"].sum()),
        cost_estimator=estimator,
    )
    manifest.update({
        "labels_path": str(labels_path),
        "spread_history_path": str(spread_history_path),
        "output_path": str(output_path),
        "side_archetype_diagnostics": str(diagnostics_path) if write_diagnostics else None,
    })
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--spread-history-path", type=Path, default=DEFAULT_SPREAD_HISTORY)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lookback-days", type=int, default=28)
    parser.add_argument("--min-observations", type=int, default=48)
    parser.add_argument("--min-distinct-days", type=int, default=7)
    parser.add_argument("--write-diagnostics", action="store_true")
    parser.add_argument("--blend-long", type=float, default=None)
    parser.add_argument("--blend-short", type=float, default=None)
    parser.add_argument(
        "--cost-estimator",
        choices=("causal_rolling_p90", "pooled_asset_p90"),
        default="causal_rolling_p90",
        help="Use pooled_asset_p90 for static liquidity-proxy target research only.",
    )
    args = parser.parse_args()
    manifest = materialize(
        labels_path=args.labels_path,
        spread_history_path=args.spread_history_path,
        output_path=args.output_path,
        spec=CausalSpreadP90Spec(
            lookback_days=args.lookback_days,
            min_observations=args.min_observations,
            min_distinct_days=args.min_distinct_days,
        ),
        cost_estimator=str(args.cost_estimator),
        blend_long=args.blend_long,
        blend_short=args.blend_short,
        write_diagnostics=bool(args.write_diagnostics),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
