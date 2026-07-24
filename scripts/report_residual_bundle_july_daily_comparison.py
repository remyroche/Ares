#!/usr/bin/env python3
"""Compare residual meta bundles on exact OOS plus forward July rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.inference.side_residual_expert import SideResidualExpertBundle
from scripts.backfill_complete_july_meta_predictions import (
    _capture_for_policy_keys,
    _load_optional_label_period,
)


SCORE = "score_base_ev_residual_expert_hier_mapped"
KEYS = ["__ts__", "__symbol__", "side_name"]


def _utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _top_fraction(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    score = pd.to_numeric(frame[SCORE], errors="coerce")
    eligible = frame.loc[np.isfinite(score)].copy()
    count = max(1, int(math.ceil(len(eligible) * float(fraction))))
    return eligible.sort_values(SCORE, ascending=False, kind="stable").head(count)


def _load_oos(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    wanted = [
        *KEYS,
        "archetype_policy_key",
        SCORE,
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    available = set(pq.read_schema(path).names)
    frame = pd.read_parquet(path, columns=[column for column in wanted if column in available])
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame = frame.loc[frame["__ts__"].ge(start) & frame["__ts__"].lt(end)].copy()
    if frame.duplicated(KEYS).any():
        raise ValueError(f"OOS artifact has duplicate candidate keys: {path}")
    frame["score_provenance"] = "expanding_window_oos"
    frame["outcome_provenance"] = "corrected_causal_label"
    frame["net_ev"] = pd.to_numeric(frame["ev_after_1pct"], errors="coerce")
    return frame


def _load_forward(
    path: Path,
    bundle: SideResidualExpertBundle,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    required = sorted(
        set(bundle.required_input_features("long"))
        | set(bundle.required_input_features("short"))
    )
    wanted = list(
        dict.fromkeys(
            [
                *KEYS,
                "archetype_policy_key",
                "score",
                *required,
            ]
        )
    )
    available = set(pq.read_schema(path).names)
    missing = sorted(set(wanted) - available)
    if missing:
        raise ValueError(f"Forward artifact is missing required fields: {missing[:20]}")
    frame = pd.read_parquet(path, columns=wanted)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame = frame.loc[frame["__ts__"].ge(start) & frame["__ts__"].lt(end)].copy()
    frame["score_base"] = pd.to_numeric(frame["score"], errors="coerce").astype(np.float32)
    scored = bundle.transform(frame)
    frame = pd.concat([frame.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    frame["score_provenance"] = "final_refit_forward"
    frame["outcome_provenance"] = "causal_15m_path_replay"
    return frame


def _evaluate_variant_side(
    *,
    variant: str,
    side: str,
    oos: pd.DataFrame,
    forward: pd.DataFrame,
    top_fraction: float,
    data_root: Path,
    path_len: int,
    exchange: str,
    labels: pd.DataFrame,
    policy_manifest: dict[str, object],
    default_barrier_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pd.concat(
        [
            oos.loc[oos["side_name"].eq(side)],
            forward.loc[forward["side_name"].eq(side)],
        ],
        ignore_index=True,
        sort=False,
    )
    if pool.duplicated(KEYS).any():
        raise ValueError(f"Combined {variant}/{side} pool contains duplicate keys")
    selected = _top_fraction(pool, top_fraction)
    old = selected.loc[selected["score_provenance"].eq("expanding_window_oos")].copy()
    new = selected.loc[selected["score_provenance"].eq("final_refit_forward")].copy()
    if not new.empty:
        barrier_history = labels.loc[
            labels["side_name"].astype(str).str.lower().eq(side),
            ["__ts__", "__symbol__", "__barrier_pct__"],
        ].copy()
        barrier_history["__ts__"] = pd.to_datetime(
            barrier_history["__ts__"], utc=True, errors="coerce"
        )
        new = new.drop(columns="__barrier_pct__", errors="ignore")
        if barrier_history.empty:
            new["__barrier_pct__"] = np.float32(default_barrier_pct)
        else:
            new = pd.merge_asof(
                new.sort_values(["__ts__", "__symbol__"], kind="stable"),
                barrier_history.sort_values(["__ts__", "__symbol__"], kind="stable"),
                on="__ts__",
                by="__symbol__",
                direction="backward",
                allow_exact_matches=True,
            )
            new["__barrier_pct__"] = pd.to_numeric(
                new["__barrier_pct__"], errors="coerce"
            ).fillna(default_barrier_pct)
        captured, _ = _capture_for_policy_keys(
            new,
            side=side,
            policy_keys=new["archetype_policy_key"],
            policy_manifest=policy_manifest,
            data_root=data_root,
            path_len=path_len,
            allow_partial_paths=False,
        )
        new = pd.concat([new.reset_index(drop=True), captured.reset_index(drop=True)], axis=1)
        new["net_ev"] = pd.to_numeric(new["capture_net"], errors="coerce")
        new["capture_valid_path"] = pd.to_numeric(
            new["capture_valid_path"], errors="coerce"
        ).fillna(0.0)
        new = new.loc[new["capture_valid_path"].eq(1.0)].copy()
    evaluated = pd.concat([old, new], ignore_index=True, sort=False)
    evaluated["variant"] = variant
    evaluated["side_name"] = side
    evaluated["utc_day"] = evaluated["__ts__"].dt.strftime("%Y-%m-%d")
    evaluated["positive_ev"] = pd.to_numeric(evaluated["net_ev"], errors="coerce").gt(0)
    daily = (
        evaluated.groupby("utc_day", sort=True, observed=True)
        .agg(
            selected_trades=("net_ev", "count"),
            net_ev_per_trade=("net_ev", "mean"),
            total_net_ev=("net_ev", "sum"),
            positive_trade_rate=("positive_ev", "mean"),
        )
        .reset_index()
    )
    daily.insert(0, "side_name", side)
    daily.insert(0, "variant", variant)
    return evaluated, daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summarize-evaluated",
        type=Path,
        help=(
            "Reuse an existing july_selected_outcomes.parquet and regenerate only "
            "the provenance-aware summaries. No models or paths are replayed."
        ),
    )
    parser.add_argument("--current-forward-candidates", type=Path)
    parser.add_argument("--hybrid-forward-candidates", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--policy-manifest", type=Path)
    parser.add_argument("--default-barrier-pct", type=float, default=0.02)
    parser.add_argument("--current-bundle", type=Path)
    parser.add_argument("--current-oos", type=Path)
    parser.add_argument("--hybrid-bundle", type=Path)
    parser.add_argument("--hybrid-oos", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--start", default="2026-07-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-07-21T00:00:00Z")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--sides", nargs="+", choices=("long", "short"), default=["long", "short"])
    args = parser.parse_args()

    start, end = _utc(args.start), _utc(args.end_exclusive)
    if args.summarize_evaluated is not None:
        evaluated = pd.read_parquet(args.summarize_evaluated)
        evaluated["__ts__"] = pd.to_datetime(
            evaluated["__ts__"], utc=True, errors="coerce"
        )
        evaluated = evaluated.loc[
            evaluated["__ts__"].ge(start) & evaluated["__ts__"].lt(end)
        ].copy()
        evaluated["utc_day"] = evaluated["__ts__"].dt.strftime("%Y-%m-%d")
        evaluated["positive_ev"] = pd.to_numeric(
            evaluated["net_ev"], errors="coerce"
        ).gt(0.0)
        group_keys = ["variant", "side_name", "score_provenance"]
        overall_scope = (
            evaluated.groupby(group_keys, sort=False, observed=True)
            .agg(
                selected_trades=("net_ev", "count"),
                net_ev_per_trade=("net_ev", "mean"),
                total_net_ev=("net_ev", "sum"),
                positive_trade_rate=("positive_ev", "mean"),
                active_days=("utc_day", "nunique"),
            )
            .reset_index()
        )
        overall_scope["trades_per_active_day"] = (
            overall_scope["selected_trades"]
            / overall_scope["active_days"].clip(lower=1)
        )
        daily_scope = (
            evaluated.groupby(
                [*group_keys, "utc_day"], sort=True, observed=True
            )
            .agg(
                selected_trades=("net_ev", "count"),
                net_ev_per_trade=("net_ev", "mean"),
                total_net_ev=("net_ev", "sum"),
                positive_trade_rate=("positive_ev", "mean"),
            )
            .reset_index()
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        overall_scope.to_csv(
            args.output_dir / "july_overall_by_prediction_scope.csv", index=False
        )
        daily_scope.to_csv(
            args.output_dir / "july_daily_by_prediction_scope.csv", index=False
        )
        scope_manifest = {
            "schema": "residual_bundle_july_scope_summary_v1",
            "source": str(args.summarize_evaluated),
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
            "selection_contract": (
                "retrospective pooled-July global top fraction within side; "
                "not a causal admission threshold and not portfolio constrained"
            ),
            "prediction_scopes": {
                "expanding_window_oos": (
                    "monthly expanding-window model prediction; model-selection "
                    "exceptions must be read from the source model manifest"
                ),
                "final_refit_forward": (
                    "final residual bundle fit through 2026-07-10 21:00 UTC; "
                    "forward diagnostic only and never OOS"
                ),
            },
        }
        (args.output_dir / "scope_manifest.json").write_text(
            json.dumps(scope_manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(overall_scope.to_string(index=False))
        return 0

    required_paths = {
        "--current-forward-candidates": args.current_forward_candidates,
        "--labels-dir": args.labels_dir,
        "--policy-manifest": args.policy_manifest,
        "--current-bundle": args.current_bundle,
        "--current-oos": args.current_oos,
    }
    missing_paths = [name for name, value in required_paths.items() if value is None]
    if missing_paths:
        parser.error("missing required arguments: " + ", ".join(missing_paths))
    variants: dict[str, tuple[Path, Path, Path]] = {
        "current_residual_only": (
            args.current_bundle,
            args.current_oos,
            args.current_forward_candidates,
        ),
    }
    if args.hybrid_bundle is not None or args.hybrid_oos is not None:
        if (
            args.hybrid_bundle is None
            or args.hybrid_oos is None
            or args.hybrid_forward_candidates is None
        ):
            raise ValueError(
                "--hybrid-bundle, --hybrid-oos, and --hybrid-forward-candidates are required together"
            )
        variants["hybrid_oldlong_canonicalshort"] = (
            args.hybrid_bundle,
            args.hybrid_oos,
            args.hybrid_forward_candidates,
        )
    all_daily: list[pd.DataFrame] = []
    all_evaluated: list[pd.DataFrame] = []
    coverage: dict[str, object] = {}
    labels = _load_optional_label_period(args.labels_dir, start=start, end=end)
    policy_manifest = json.loads(args.policy_manifest.read_text(encoding="utf-8"))
    for variant, (bundle_path, oos_path, forward_path) in variants.items():
        bundle = SideResidualExpertBundle.load(bundle_path)
        oos = _load_oos(oos_path, start, end)
        oos_max = oos["__ts__"].max()
        if pd.isna(oos_max):
            raise ValueError(f"No exact OOS rows for {variant}")
        forward_start = oos_max + pd.Timedelta(hours=1)
        forward = _load_forward(
            forward_path,
            bundle,
            start=forward_start,
            end=end,
        )
        coverage[variant] = {
            "oos_rows": int(len(oos)),
            "oos_max_ts": oos_max.isoformat(),
            "forward_rows": int(len(forward)),
            "forward_min_ts": forward["__ts__"].min().isoformat() if not forward.empty else None,
            "forward_max_ts": forward["__ts__"].max().isoformat() if not forward.empty else None,
        }
        for side in args.sides:
            evaluated, daily = _evaluate_variant_side(
                variant=variant,
                side=side,
                oos=oos,
                forward=forward,
                top_fraction=args.top_fraction,
                data_root=args.data_root,
                path_len=args.path_len,
                exchange=args.exchange,
                labels=labels,
                policy_manifest=policy_manifest,
                default_barrier_pct=args.default_barrier_pct,
            )
            all_evaluated.append(evaluated)
            all_daily.append(daily)

    evaluated = pd.concat(all_evaluated, ignore_index=True, sort=False)
    daily = pd.concat(all_daily, ignore_index=True, sort=False)
    calendar = pd.MultiIndex.from_product(
        [
            list(variants),
            args.sides,
            pd.date_range(start.normalize(), end.normalize(), inclusive="left").strftime("%Y-%m-%d"),
        ],
        names=["variant", "side_name", "utc_day"],
    ).to_frame(index=False)
    daily = calendar.merge(daily, on=["variant", "side_name", "utc_day"], how="left")
    daily["selected_trades"] = daily["selected_trades"].fillna(0).astype(int)
    overall = (
        evaluated.groupby(["variant", "side_name"], sort=False, observed=True)
        .agg(
            selected_trades=("net_ev", "count"),
            net_ev_per_trade=("net_ev", "mean"),
            total_net_ev=("net_ev", "sum"),
            positive_trade_rate=("positive_ev", "mean"),
            active_days=("utc_day", "nunique"),
        )
        .reset_index()
    )
    overall["trades_per_calendar_day"] = overall["selected_trades"] / 20.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / "july_daily_by_bundle_side.csv", index=False)
    overall.to_csv(args.output_dir / "july_overall_by_bundle_side.csv", index=False)
    evaluated.to_parquet(args.output_dir / "july_selected_outcomes.parquet", index=False)
    manifest = {
        "schema": "residual_bundle_july_daily_comparison_v1",
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "selection": "global_top_fraction_within_each_bundle_and_side",
        "top_fraction": args.top_fraction,
        "round_trip_cost": 0.01,
        "oos_outcome": "ev_after_1pct from corrected causal label artifact",
        "forward_outcome": f"causal {args.path_len}x15m path replay with row geometry",
        "coverage": coverage,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print(daily.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
