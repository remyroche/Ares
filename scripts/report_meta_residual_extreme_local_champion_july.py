#!/usr/bin/env python3
"""Apply the frozen strict local-state overlay to the July champion contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meta_residual_extreme_local_champion_overlay import (  # noqa: E402
    FEATURES,
    KEYS,
    PARENT,
    _adjust_rank,
    _autocorrelation_report,
    _breakdown,
    _composite,
    _fit_references,
    _metric_row,
)


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
        "--state-artifact",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_"
            "compactlocal_market_20260712_v3/oos_residual_event_states.parquet"
        ),
    )
    parser.add_argument(
        "--july-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "champion_frozen_single_source_202501_20260710/prediction_shards/"
            "predictions_2026-07.parquet"
        ),
    )
    parser.add_argument(
        "--frozen-overlay-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_20260712_v5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_july_20260712_v1"
        ),
    )
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--train-end", default="2026-04-01")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_start = pd.Timestamp(args.train_start, tz="UTC")
    train_end = pd.Timestamp(args.train_end, tz="UTC")

    frozen_manifest = json.loads(
        (args.frozen_overlay_dir / "manifest.json").read_text(encoding="utf-8")
    )
    params = dict(frozen_manifest["strict_best"])
    catalog = pd.read_csv(args.frozen_overlay_dir / "selected_local_features_strict.csv")

    train_columns = [
        *KEYS,
        "historical_rank",
        "hit_probability",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    train = pd.read_parquet(args.champion_ledger, columns=train_columns)
    train["__ts__"] = pd.to_datetime(train["__ts__"], utc=True)
    train = train.loc[train["__ts__"].ge(train_start) & train["__ts__"].lt(train_end)]

    state_columns = [*KEYS, *FEATURES]
    state = pd.read_parquet(args.state_artifact, columns=state_columns)
    state["__ts__"] = pd.to_datetime(state["__ts__"], utc=True)
    state = state.drop_duplicates(KEYS, keep="last")
    train = train.merge(
        state.loc[state["__ts__"].lt(train_end)],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    references = _fit_references(train, catalog, int(params["top_feature_count"]))

    july_columns = [
        *KEYS,
        "threshold_alternative_rank",
        "threshold_alternative_selected",
        "hit_probability",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
        "outcomes_available",
    ]
    july = pd.read_parquet(args.july_predictions, columns=july_columns)
    july["__ts__"] = pd.to_datetime(july["__ts__"], utc=True)
    july = july.loc[july["outcomes_available"].eq(True)].drop(
        columns="outcomes_available"
    )
    july = july.merge(state, on=KEYS, how="inner", validate="one_to_one")

    parent_rank = pd.to_numeric(
        july["threshold_alternative_rank"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float32)
    parent_selected = july["threshold_alternative_selected"].fillna(False).to_numpy(
        dtype=bool
    )
    if not np.array_equal(parent_selected, parent_rank >= 0.90):
        raise RuntimeError("July parent selection does not match its saved policy rank")

    adverse = _composite(july, references, "adverse")
    positive = _composite(july, references, "positive")
    adjusted = _adjust_rank(
        parent_rank,
        adverse,
        positive,
        threshold=float(params["threshold"]),
        alpha_down=float(params["alpha_down"]),
        alpha_up=float(params["alpha_up"]),
    )
    selected = adjusted >= 0.90
    july["resid_strict_extreme_adverse_composite"] = adverse
    july["resid_strict_extreme_positive_composite"] = positive
    july["threshold_alternative_rank_strict_extreme_local"] = adjusted
    july["selected_parent"] = parent_selected
    july["selected_strict_extreme_local"] = selected

    parent_selector = f"{PARENT}_july_policy"
    strict_selector = f"{PARENT}_strict_extreme_local_july_policy"
    summary = pd.DataFrame(
        [
            _metric_row(july, parent_selected, parent_selector),
            _metric_row(july, selected, strict_selector),
        ]
    )
    calendar, autocorrelation = _autocorrelation_report(
        july, {parent_selector: parent_selected, strict_selector: selected}
    )
    mean_abs_ac = (
        autocorrelation.groupby("arm", observed=True)[
            "signed_surprise_autocorr_lag1"
        ]
        .apply(lambda values: float(values.abs().mean()))
        .rename("mean_abs_signed_surprise_autocorr_lag1")
    )
    summary = summary.merge(mean_abs_ac, left_on="selector", right_index=True, how="left")

    july.to_parquet(args.output_dir / "july_predictions.parquet", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat(
        [
            _breakdown(july, parent_selected, parent_selector),
            _breakdown(july, selected, strict_selector),
        ],
        ignore_index=True,
    ).to_csv(args.output_dir / "breakdowns.csv", index=False)
    calendar.to_csv(args.output_dir / "hit_surprise_calendar.csv", index=False)
    autocorrelation.to_csv(args.output_dir / "hit_surprise_autocorrelation.csv", index=False)

    train_search = pd.read_csv(args.frozen_overlay_dir / "train_search.csv")
    tail_metric_rows: list[dict[str, Any]] = []
    tail_breakdowns: list[pd.DataFrame] = []
    tail_masks: dict[str, np.ndarray] = {}
    for tail_threshold in (0.95, 0.96, 0.975, 0.99, 0.995):
        candidates = train_search.loc[
            train_search["threshold"].eq(tail_threshold)
            & train_search["alpha_down"].gt(0.0)
            & train_search["alpha_up"].eq(0.0)
        ].sort_values(
            ["objective", "activity_ratio", "alpha_down"],
            ascending=[False, False, True],
            kind="stable",
        )
        if candidates.empty:
            continue
        tail_params = candidates.iloc[0].to_dict()
        tail_adjusted = _adjust_rank(
            parent_rank,
            adverse,
            positive,
            threshold=float(tail_params["threshold"]),
            alpha_down=float(tail_params["alpha_down"]),
            alpha_up=0.0,
        )
        tail_mask = tail_adjusted >= 0.90
        tail_selector = f"{PARENT}_july_forced_local_tail_{tail_threshold:.3f}"
        tail_metric = _metric_row(july, tail_mask, tail_selector)
        tail_metric.update(
            {
                "train_threshold": float(tail_params["threshold"]),
                "train_alpha_down": float(tail_params["alpha_down"]),
                "train_activity_ratio": float(tail_params["activity_ratio"]),
                "train_objective": float(tail_params["objective"]),
                "dropped_rows_vs_parent": int(np.sum(parent_selected & ~tail_mask)),
                "added_rows_vs_parent": int(np.sum(~parent_selected & tail_mask)),
            }
        )
        tail_metric_rows.append(tail_metric)
        tail_masks[tail_selector] = tail_mask
        tail_breakdowns.append(_breakdown(july, tail_mask, tail_selector))
    if tail_metric_rows:
        _, tail_autocorrelation = _autocorrelation_report(july, tail_masks)
        tail_mean_abs_ac = (
            tail_autocorrelation.groupby("arm", observed=True)[
                "signed_surprise_autocorr_lag1"
            ]
            .apply(lambda values: float(values.abs().mean()))
            .rename("mean_abs_signed_surprise_autocorr_lag1")
        )
        tail_summary = pd.DataFrame(tail_metric_rows).merge(
            tail_mean_abs_ac, left_on="selector", right_index=True, how="left"
        )
        tail_summary.to_csv(args.output_dir / "tail_comparison_summary.csv", index=False)
        pd.concat(tail_breakdowns, ignore_index=True).to_csv(
            args.output_dir / "tail_comparison_breakdowns.csv", index=False
        )
        tail_autocorrelation.to_csv(
            args.output_dir / "tail_comparison_autocorrelation.csv", index=False
        )

    dropped = parent_selected & ~selected
    _write_json(
        args.output_dir / "manifest.json",
        {
            "schema": "meta_residual_extreme_local_champion_july_v1",
            "parent": PARENT,
            "frozen_overlay_manifest": str(args.frozen_overlay_dir / "manifest.json"),
            "july_predictions": str(args.july_predictions),
            "state_artifact": str(args.state_artifact),
            "params": params,
            "train_rows": int(len(train)),
            "july_rows_with_outcomes_and_state": int(len(july)),
            "parent_rows": int(parent_selected.sum()),
            "strict_rows": int(selected.sum()),
            "dropped_rows": int(dropped.sum()),
            "added_rows": int((~parent_selected & selected).sum()),
            "dropped_mean_ev_after_1pct": float(july.loc[dropped, "ev_after_1pct"].mean()),
            "leakage_contract": (
                "The local feature catalog, empirical references, tail threshold, and "
                "rank alpha are frozen from rows before 2026-04-01. July outcomes are "
                "used only after frozen selection for reporting."
            ),
            "policy_contract": (
                "The parent is the saved July 8-day reachable-EV policy rank. The local "
                "overlay can only demote its selected rows; it cannot add trades."
            ),
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
