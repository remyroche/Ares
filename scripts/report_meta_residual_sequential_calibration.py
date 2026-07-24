#!/usr/bin/env python3
"""Leakage-safe sequential calibration diagnostics for residual meta ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    _experiment_score,
    _parse_months,
    metrics_by_scope,
    surprise_calendar,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fit_iso(scores: pd.Series, target: pd.Series) -> IsotonicRegression | None:
    x = pd.to_numeric(scores, errors="coerce").clip(0.0, 1.0)
    y = pd.to_numeric(target, errors="coerce").clip(0.0, 1.0)
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 200 or int(x.loc[valid].nunique()) < 8:
        return None
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        x.loc[valid], y.loc[valid]
    )


def sequential_calibrate(
    predictions: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    month_col: str = "calendar_month",
    min_local_rows: int = 600,
    min_side_rows: int = 1_000,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit prior-OOS calibrators and assign each later month with frozen state.

    The first month remains raw because no prior alternative OOS predictions are
    available inside the artifact. Later months use local side x archetype
    isotonic calibrators when support permits, then side-level, then global.
    """

    frame = predictions.copy(deep=False)
    frame[month_col] = frame[month_col].astype(str)
    frame["side_name"] = frame.get("side_name", "missing").astype(str).str.lower()
    frame["archetype_policy_key"] = frame.get(
        "archetype_policy_key", "missing"
    ).astype(str)
    out = np.full(len(frame), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for month in sorted(frame[month_col].dropna().unique().tolist()):
        valid_idx = frame.index[frame[month_col].eq(month)]
        train = frame.loc[frame[month_col].lt(month)].copy()
        valid = frame.loc[valid_idx].copy()
        raw_valid = (
            pd.to_numeric(valid[source_col], errors="coerce")
            .clip(0.0, 1.0)
            .fillna(0.5)
            .to_numpy(dtype=np.float32)
        )
        if train.empty:
            out[valid_idx] = raw_valid
            rows.append(
                {
                    "month": month,
                    "rows": int(len(valid)),
                    "train_rows": 0,
                    "fallback": "raw_no_prior",
                    "local_models": 0,
                    "side_models": 0,
                    "sources": {"raw_no_prior": int(len(valid))},
                }
            )
            continue

        global_model = _fit_iso(train[source_col], train[target_col])
        side_models: dict[str, IsotonicRegression] = {}
        local_models: dict[tuple[str, str], IsotonicRegression] = {}
        for side, group in train.groupby("side_name", sort=False):
            if len(group) >= int(min_side_rows):
                model = _fit_iso(group[source_col], group[target_col])
                if model is not None:
                    side_models[str(side)] = model
        for key, group in train.groupby(
            ["side_name", "archetype_policy_key"], sort=False
        ):
            if len(group) >= int(min_local_rows):
                model = _fit_iso(group[source_col], group[target_col])
                if model is not None:
                    local_models[(str(key[0]), str(key[1]))] = model

        calibrated: list[float] = []
        sources: list[str] = []
        for _, row in valid.iterrows():
            raw_value = pd.to_numeric(pd.Series([row[source_col]]), errors="coerce").iloc[
                0
            ]
            x = float(np.clip(raw_value, 0.0, 1.0)) if pd.notna(raw_value) else 0.5
            local_key = (str(row["side_name"]), str(row["archetype_policy_key"]))
            if local_key in local_models:
                calibrated.append(float(local_models[local_key].predict([x])[0]))
                sources.append("local")
            elif str(row["side_name"]) in side_models:
                calibrated.append(float(side_models[str(row["side_name"])].predict([x])[0]))
                sources.append("side")
            elif global_model is not None:
                calibrated.append(float(global_model.predict([x])[0]))
                sources.append("global")
            else:
                calibrated.append(x)
                sources.append("raw_no_model")
        out[valid_idx] = np.asarray(calibrated, dtype=np.float32)
        rows.append(
            {
                "month": month,
                "rows": int(len(valid)),
                "train_rows": int(len(train)),
                "fallback": None,
                "local_models": int(len(local_models)),
                "side_models": int(len(side_models)),
                "sources": dict(pd.Series(sources).value_counts()),
            }
        )
    return out, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--mode",
        choices=("rank_score", "hit_probability"),
        default="rank_score",
        help=(
            "rank_score calibrates score_alternative and can change selected rows; "
            "hit_probability calibrates hit_prob_alternative and leaves ranking unchanged."
        ),
    )
    parser.add_argument("--target-col", default="clean_exec")
    parser.add_argument("--min-local-rows", type=int, default=600)
    parser.add_argument("--min-side-rows", type=int, default=1000)
    parser.add_argument(
        "--report-months",
        default=None,
        help=(
            "Optional comma-separated YYYY-MM months used for metrics only. "
            "Calibration still uses every earlier month present in the input "
            "predictions, so a burn-in month can be present without being scored."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_dir = args.run_dir / args.arm
    predictions_path = arm_dir / "oos_predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    predictions = pd.read_parquet(predictions_path)
    if args.mode == "rank_score":
        source_col = "score_alternative"
        target_col = args.target_col
        out_arm = f"{args.arm}_seq_rank_calibrated"
    else:
        source_col = "hit_prob_alternative"
        target_col = args.target_col
        out_arm = f"{args.arm}_seq_hitprob_calibrated"
    calibrated, contract = sequential_calibrate(
        predictions,
        source_col=source_col,
        target_col=target_col,
        min_local_rows=int(args.min_local_rows),
        min_side_rows=int(args.min_side_rows),
    )
    out = predictions.copy()
    out[f"{source_col}_raw"] = out[source_col]
    out[source_col] = calibrated
    out_dir = args.run_dir / out_arm
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "oos_predictions.parquet", index=False)
    report_months = (
        _parse_months(args.report_months) if args.report_months is not None else None
    )
    report_frame = out
    if report_months is not None:
        report_frame = out.loc[
            out["calendar_month"].astype(str).isin(report_months)
        ].copy()
        if report_frame.empty:
            raise ValueError(
                f"No rows remain after --report-months filter: {list(report_months)}"
            )
    metrics = metrics_by_scope(report_frame, out_arm)
    calendar, autocorr, period_cmp = surprise_calendar(report_frame, out_arm)
    metrics.to_csv(out_dir / "metrics_by_scope.csv", index=False)
    calendar.to_csv(out_dir / "hit_surprise_calendar.csv", index=False)
    autocorr.to_csv(out_dir / "hit_surprise_autocorrelation.csv", index=False)
    period_cmp.to_csv(out_dir / "high_surprise_period_comparison.csv", index=False)
    scorecard = _experiment_score(metrics, autocorr, period_cmp, out_arm)
    pd.DataFrame([scorecard]).to_csv(out_dir / "experiment_scorecard.csv", index=False)
    contract.to_csv(out_dir / "calibration_contract.csv", index=False)
    manifest = {
        "generated_by": "report_meta_residual_sequential_calibration",
        "source_arm": args.arm,
        "output_arm": out_arm,
        "mode": args.mode,
        "source_col": source_col,
        "target_col": target_col,
        "calibration_months": sorted(out["calendar_month"].astype(str).unique().tolist()),
        "report_months": list(report_months) if report_months is not None else None,
        "contract": (
            "Each month uses only prior alternative OOS predictions. The first "
            "month remains raw when no prior month exists in the artifact. "
            "When report_months is set, excluded months are calibration burn-in "
            "only and are not included in scorecard metrics."
        ),
        "scorecard": scorecard,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"status": "complete", **manifest}), sort_keys=True))


if __name__ == "__main__":
    main()
