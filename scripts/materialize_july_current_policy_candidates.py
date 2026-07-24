#!/usr/bin/env python3
"""Convert canonical-chain or inference-ledger rows to replay candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Optional monthly label directory used for a causal same-symbol/side "
            "barrier fallback when a shadow inference ledger did not reach the "
            "execution stage."
        ),
    )
    return parser.parse_args()


def _coalesce(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=object)
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        result = result.where(result.notna(), values)
    return result


def _load_label_barriers(labels_dir: Path, months: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for side in ("long", "short"):
        for month in sorted(months):
            year, month_number = month.split("-")
            path = labels_dir / f"train_global_{side}_5_{year}_{month_number}.parquet"
            if not path.exists():
                continue
            part = pd.read_parquet(
                path,
                columns=["__ts__", "__symbol__", "__barrier_pct__"],
            ).rename(
                columns={
                    "__ts__": "timestamp",
                    "__symbol__": "symbol",
                    "__barrier_pct__": "label_barrier_pct",
                }
            )
            part["side_name"] = side
            part["timestamp"] = pd.to_datetime(
                part["timestamp"], utc=True, errors="coerce"
            )
            part["symbol"] = part["symbol"].astype(str)
            part["label_barrier_pct"] = pd.to_numeric(
                part["label_barrier_pct"], errors="coerce"
            )
            parts.append(part.dropna(subset=["timestamp", "label_barrier_pct"]))
    if not parts:
        raise FileNotFoundError(
            f"No monthly label barriers found under {labels_dir} for {sorted(months)}"
        )
    barriers = pd.concat(parts, ignore_index=True, copy=False)
    return barriers.sort_values(["timestamp", "symbol", "side_name"], kind="stable")


def _attach_causal_label_barrier(
    selected: pd.DataFrame,
    labels_dir: Path,
) -> pd.DataFrame:
    missing = ~np.isfinite(
        pd.to_numeric(selected["barrier_pct"], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
    )
    if not bool(missing.any()):
        selected["barrier_source"] = "source_row"
        return selected
    months = set(selected.loc[missing, "timestamp"].dt.strftime("%Y-%m").dropna())
    barriers = _load_label_barriers(labels_dir, months)
    work = selected.reset_index(drop=False).rename(columns={"index": "__source_index"})
    output: list[pd.DataFrame] = []
    for side, group in work.groupby("side_name", sort=False):
        right = barriers.loc[barriers["side_name"].eq(side)].drop(
            columns="side_name"
        )
        group = group.sort_values(["timestamp", "symbol"], kind="stable")
        right = right.sort_values(["timestamp", "symbol"], kind="stable")
        output.append(
            pd.merge_asof(
                group,
                right,
                on="timestamp",
                by="symbol",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    merged = pd.concat(output, ignore_index=True, copy=False).sort_values(
        "__source_index", kind="stable"
    )
    source_barrier = pd.to_numeric(merged["barrier_pct"], errors="coerce")
    label_barrier = pd.to_numeric(merged.pop("label_barrier_pct"), errors="coerce")
    merged["barrier_pct"] = source_barrier.fillna(label_barrier)
    merged["barrier_source"] = np.where(
        source_barrier.notna(), "source_row", "causal_label_asof"
    )
    return merged.drop(columns="__source_index").reset_index(drop=True)


def _normalise_source(source: pd.DataFrame) -> pd.DataFrame:
    selected = source.loc[source["threshold_basis_selected"].fillna(False)].copy()
    selected["timestamp"] = pd.to_datetime(
        _coalesce(selected, ("__ts__", "signal_bar_ts")),
        utc=True,
        errors="coerce",
    )
    selected["symbol"] = _coalesce(selected, ("__symbol__", "symbol")).astype(str)
    side = _coalesce(selected, ("side_name", "side")).astype(str).str.lower()
    side = side.replace({"1": "long", "1.0": "long", "-1": "short", "-1.0": "short"})
    selected["side_name"] = side
    selected["side"] = np.where(side.eq("short"), -1.0, 1.0).astype(np.float32)
    selected["strategy_id"] = _coalesce(selected, ("strategy_id",)).where(
        lambda value: value.notna(), side + "_s52_meta_threshold_handoff"
    )
    selected["rank_pct"] = pd.to_numeric(
        _coalesce(
            selected,
            (
                "threshold_basis_rank_score",
                "threshold_basis_corrected_expected_ev_rank",
                "policy_rank_pct",
            ),
        ),
        errors="coerce",
    )
    selected["calibrated_score"] = pd.to_numeric(
        _coalesce(
            selected,
            (
                "threshold_basis_corrected_expected_ev",
                "expected_net_ev_after_1pct_mlp_direct",
                "calibrated_score",
            ),
        ),
        errors="coerce",
    )
    selected["barrier_pct"] = pd.to_numeric(
        _coalesce(
            selected,
            ("__barrier_pct__", "policy_effective_barrier_pct", "barrier_pct"),
        ),
        errors="coerce",
    )

    policy_archetype = _coalesce(
        selected,
        ("policy_archetype", "local_side_archetype"),
    ).astype(str)
    archetype_key = _coalesce(
        selected,
        ("__archetype_policy_key__", "archetype_policy_key"),
    )
    missing_key = archetype_key.isna()
    archetype_key = archetype_key.astype(str)
    for prefix in ("long__", "short__"):
        archetype_key = archetype_key.str.removeprefix(prefix)
    archetype_key = archetype_key.where(~missing_key, policy_archetype)
    for prefix in ("long__", "short__"):
        archetype_key = archetype_key.str.removeprefix(prefix)
    selected["archetype_policy_key"] = archetype_key
    selected["policy_archetype"] = np.where(
        policy_archetype.str.startswith(("long__", "short__")),
        policy_archetype,
        side + "__" + archetype_key,
    )
    selected["local_side_archetype"] = selected["policy_archetype"]
    selected["base_strategy_threshold"] = np.float32(0.0)

    spread = pd.to_numeric(
        _coalesce(
            selected,
            (
                "median_spread_bps",
                "expected_spread_bps",
                "policy_spread_bps",
                "spread_bps",
            ),
        ),
        errors="coerce",
    )
    historical_cost = pd.to_numeric(
        _coalesce(selected, ("estimated_ev_historical_cost_bps",)), errors="coerce"
    )
    # Historical model EV embeds the 100 bps round-trip fee plus its spread
    # estimate. Recover only the spread component; the replay applies the 1%
    # fee exactly once.
    spread = spread.fillna((historical_cost - 100.0).clip(lower=0.0))
    selected["expected_spread_bps"] = spread
    selected["policy_spread_bps"] = spread
    selected["expected_half_spread_bps"] = spread / 2.0
    selected["spread_cost_bps"] = spread / 2.0
    selected["exit_quote_half_spread_bps"] = spread / 2.0
    selected["exit_spread_cost_bps"] = spread / 2.0
    return selected


def main() -> int:
    args = parse_args()
    source = pd.read_parquet(args.input)
    selected = _normalise_source(source)
    selected["barrier_source"] = np.where(
        selected["barrier_pct"].notna(), "source_row", "missing"
    )
    if args.labels_dir is not None:
        selected = _attach_causal_label_barrier(selected, args.labels_dir)

    required = ["timestamp", "symbol", "side_name", "rank_pct", "barrier_pct"]
    invalid = selected[required].isna().any(axis=1)
    if bool(invalid.any()):
        sample = selected.loc[invalid, required + ["policy_archetype"]].head(10)
        raise ValueError(
            f"{int(invalid.sum())}/{len(selected)} selected rows lack a replay "
            f"contract; pass --labels-dir for missing barriers. Sample:\n{sample}"
        )

    columns = [
        "timestamp",
        "symbol",
        "side",
        "side_name",
        "strategy_id",
        "rank_pct",
        "calibrated_score",
        "barrier_pct",
        "barrier_source",
        "base_strategy_threshold",
        "policy_archetype",
        "local_side_archetype",
        "archetype_policy_key",
        "expected_spread_bps",
        "policy_spread_bps",
        "expected_half_spread_bps",
        "spread_cost_bps",
        "exit_quote_half_spread_bps",
        "exit_spread_cost_bps",
        "threshold_basis_corrected_expected_ev",
        "threshold_basis_corrected_expected_ev_rank",
        "threshold_basis_reason",
    ]
    columns = [column for column in columns if column in selected.columns]
    output = selected[columns].sort_values(
        ["timestamp", "symbol", "strategy_id"], kind="stable"
    )
    duplicate_keys = ["timestamp", "symbol", "strategy_id"]
    if int(output.duplicated(duplicate_keys).sum()):
        raise ValueError("Replay candidates contain duplicate timestamp/symbol/strategy rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    manifest = {
        "schema": "july_current_policy_candidates_v2",
        "source": str(args.input),
        "output": str(args.output),
        "selected_rows": int(len(output)),
        "min_timestamp": output["timestamp"].min().isoformat() if len(output) else None,
        "max_timestamp": output["timestamp"].max().isoformat() if len(output) else None,
        "barrier_sources": output["barrier_source"].value_counts().to_dict(),
        "fee_contract": "No fee deducted here; execution replay applies 1% once.",
        "spread_contract": "Full spread split into entry/exit half-spreads.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
