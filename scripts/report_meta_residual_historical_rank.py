#!/usr/bin/env python3
"""Evaluate the residual meta alternative with causal historical score ranks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_historical_rank import (
    HistoricalScoreRankReference,  # noqa: E402
)
from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    OUTCOME_COLUMNS,
    REFERENCE_DERIVED_COLUMNS,
)
from extreme_price_movements.meta_residual_overlay import (
    ResidualOverlayState,  # noqa: E402
)
from scripts.run_meta_residual_ae_representation_ablation import ARM  # noqa: E402
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    _calibrate,
    _fit_platt,
    _merge_residual_features,
)

FRACTIONS = (0.05, 0.10, 0.20, 0.30)
SCOPES = {
    "overall": [],
    "month": ["calendar_month"],
    "week": ["week_start"],
    "side": ["side_name"],
    "archetype": ["archetype_policy_key"],
    "month_side_archetype": ["calendar_month", "side_name", "archetype_policy_key"],
    "week_side_archetype": ["week_start", "side_name", "archetype_policy_key"],
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if value is pd.NaT:
            return None
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _cache_suffix(arm: str) -> str:
    return arm.removeprefix(ARM)


def _burnin(
    root: Path,
    arm: str = ARM,
    generated_cache: str | None = None,
) -> pd.DataFrame:
    burnin = pd.read_parquet(
        root / "lifecycle_only_burnin" / "oos_predictions_march_burnin.parquet"
    )
    cache_name = (
        generated_cache
        or f"residual_walkforward_ae_gmm_eval_mar_jun{_cache_suffix(arm)}.parquet"
    )
    generated = pd.read_parquet(root / "cache" / cache_name)
    burnin = _merge_residual_features(burnin, generated)
    burnin["score_lifecycle_only"] = pd.to_numeric(
        burnin["score_alternative"], errors="coerce"
    ).astype(np.float32)
    arm_dir = root / arm
    state_path = arm_dir / "residual_overlay_state.joblib"
    if state_path.exists():
        state = joblib.load(state_path)
    else:
        # The first raw-overlay artifact predates persisted inference state.
        # Reconstruct its exact March-frozen global coefficients from the
        # manifest; local normalization remains disabled for this arm.
        manifest = json.loads((arm_dir / "manifest.json").read_text(encoding="utf-8"))
        state = ResidualOverlayState(
            hit_alpha=float(manifest.get("hit_alpha", 0.0)),
            dirty_lambda=float(manifest.get("dirty_lambda", 0.0)),
            local_hit_alpha=float(manifest.get("local_hit_alpha", 0.0)),
            local_dirty_lambda=float(manifest.get("local_dirty_lambda", 0.0)),
        )
    safe = burnin.drop(
        columns=[
            name
            for name in OUTCOME_COLUMNS | REFERENCE_DERIVED_COLUMNS
            if name in burnin.columns
        ],
        errors="ignore",
    )
    burnin["score_alternative"] = state.transform(
        safe,
        burnin["score_lifecycle_only"].fillna(0.5).to_numpy(dtype=np.float32),
    )
    calibrator_path = arm_dir / "hit_calibrator.joblib"
    calibrator = (
        joblib.load(calibrator_path)
        if calibrator_path.exists()
        else _fit_platt(burnin["score_alternative"], burnin["clean_exec"])
    )
    burnin["hit_prob_alternative"] = _calibrate(calibrator, burnin["score_alternative"])
    burnin["calendar_month"] = "2026-03"
    burnin["week_start"] = _true_monday_week_start(burnin["__ts__"])
    return burnin


def _true_monday_week_start(values: pd.Series) -> pd.Series:
    timestamp = pd.to_datetime(values, utc=True, errors="coerce").dt.floor("D")
    return timestamp - pd.to_timedelta(timestamp.dt.weekday.to_numpy(), unit="D")


def _walkforward_ranks(
    burnin: pd.DataFrame, oos: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    prior = burnin.copy()
    frames: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for month in ("2026-04", "2026-05", "2026-06"):
        valid = oos[oos["calendar_month"].astype(str).eq(month)].copy()
        current_state = HistoricalScoreRankReference(
            score_col="score_current_reference"
        ).fit(prior)
        alternative_state = HistoricalScoreRankReference(
            score_col="score_alternative"
        ).fit(prior)
        valid["historical_rank_current_reference"] = current_state.transform(
            valid, "score_current_reference"
        )
        valid["historical_rank_alternative"] = alternative_state.transform(
            valid, "score_alternative"
        )
        folds.append(
            {
                "month": month,
                "prior_rows": int(len(prior)),
                "valid_rows": int(len(valid)),
                "prior_end": str(pd.to_datetime(prior["__ts__"], utc=True).max()),
                "valid_start": str(pd.to_datetime(valid["__ts__"], utc=True).min()),
                "current_reference": current_state.manifest(),
                "alternative_reference": alternative_state.manifest(),
            }
        )
        frames.append(valid)
        prior = pd.concat([prior, valid], ignore_index=True)
    return pd.concat(frames, ignore_index=True), folds


def _record(
    frame: pd.DataFrame, mask: pd.Series, selector: str, fraction: float
) -> dict[str, Any]:
    selected = frame.loc[mask]
    return {
        "selector": selector,
        "fraction": fraction,
        "candidate_rows": int(len(frame)),
        "selected_rows": int(mask.sum()),
        "trades_per_observed_day": float(
            mask.sum()
            / max(pd.to_datetime(frame["__ts__"], utc=True).dt.floor("D").nunique(), 1)
        ),
        "mean_ev_after_1pct": float(
            pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean()
        ),
        "clean_exec_precision": float(
            pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
        ),
        "dirty_positive_rate": float(
            pd.to_numeric(selected["dirty_positive"], errors="coerce").mean()
        ),
        "first_touch_bad_mae_rate": float(
            pd.to_numeric(selected["first_touch_bad_mae_1r"], errors="coerce").mean()
        ),
        "full_path_bad_mae_rate": float(
            pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean()
        ),
        "timeout_rate": float(
            pd.to_numeric(selected["timeout"], errors="coerce").mean()
        ),
    }


def _metrics(frame: pd.DataFrame, alternative_selector: str = ARM) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, group_cols in SCOPES.items():
        grouped: Iterable[tuple[Any, pd.DataFrame]] = (
            [((), frame)]
            if not group_cols
            else frame.groupby(group_cols, dropna=False, sort=True)
        )
        for key, group in grouped:
            values = key if isinstance(key, tuple) else (key,)
            for selector, rank_col in (
                ("current_reference", "historical_rank_current_reference"),
                (alternative_selector, "historical_rank_alternative"),
            ):
                for fraction in FRACTIONS:
                    mask = pd.to_numeric(group[rank_col], errors="coerce").ge(
                        1.0 - fraction
                    )
                    row = _record(group, mask, selector, fraction)
                    row["scope"] = scope
                    for name, value in zip(group_cols, values, strict=False):
                        row[name] = value
                    rows.append(row)
    return pd.DataFrame(rows)


def _calendar(
    frame: pd.DataFrame,
    alternative_selector: str = ARM,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for selector, rank_col, probability_col in (
        (
            "current_reference",
            "historical_rank_current_reference",
            "hit_prob_current_reference",
        ),
        (alternative_selector, "historical_rank_alternative", "hit_prob_alternative"),
    ):
        selected = frame[
            pd.to_numeric(frame[rank_col], errors="coerce").ge(0.90)
        ].copy()
        selected["date"] = pd.to_datetime(selected["__ts__"], utc=True).dt.floor("D")
        selected["hit_surprise"] = pd.to_numeric(
            selected["clean_exec"], errors="coerce"
        ) - pd.to_numeric(selected[probability_col], errors="coerce")
        daily = (
            selected.groupby(
                ["date", "side_name", "archetype_policy_key"], dropna=False
            )
            .agg(
                rows=("clean_exec", "size"),
                hit_rate=("clean_exec", "mean"),
                mean_hit_surprise=("hit_surprise", "mean"),
                mean_ev_after_1pct=("ev_after_1pct", "mean"),
            )
            .reset_index()
        )
        daily["selector"] = selector
        parts.append(daily)
    calendar = pd.concat(parts, ignore_index=True)
    ac_rows: list[dict[str, Any]] = []
    for (selector, side, archetype), group in calendar.groupby(
        ["selector", "side_name", "archetype_policy_key"], dropna=False
    ):
        series = group.sort_values("date")["mean_hit_surprise"]
        ac_rows.append(
            {
                "selector": selector,
                "side_name": side,
                "archetype_policy_key": archetype,
                "days": len(series),
                "surprise_autocorr_lag1": float(series.autocorr(1))
                if len(series) >= 3
                else np.nan,
            }
        )
    autocorr = pd.DataFrame(ac_rows)
    keys = ["date", "side_name", "archetype_policy_key"]
    base = calendar[calendar["selector"].eq("current_reference")]
    alt = calendar[calendar["selector"].eq(alternative_selector)]
    comparison = base.merge(alt, on=keys, suffixes=("_base", "_alt"), how="inner")
    comparison["surprise_abs_improvement"] = (
        comparison["mean_hit_surprise_base"].abs()
        - comparison["mean_hit_surprise_alt"].abs()
    )
    comparison["ev_delta"] = (
        comparison["mean_ev_after_1pct_alt"] - comparison["mean_ev_after_1pct_base"]
    )
    comparison["baseline_tail_threshold"] = comparison.groupby(
        ["side_name", "archetype_policy_key"]
    )["mean_hit_surprise_base"].transform(lambda values: values.abs().quantile(0.90))
    comparison["baseline_high_surprise"] = (
        comparison["mean_hit_surprise_base"]
        .abs()
        .ge(comparison["baseline_tail_threshold"])
    )
    comparison["high_surprise_significantly_improved"] = comparison[
        "surprise_abs_improvement"
    ].ge(0.20 * comparison["mean_hit_surprise_base"].abs())
    return calendar, autocorr, comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=ARM)
    parser.add_argument(
        "--generated-cache",
        default=None,
        help="Optional residual-feature cache filename under the experiment cache directory.",
    )
    parser.add_argument(
        "--package-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Embed the final historical rank in the inference bundle; defaults on only for the champion arm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm = str(args.arm)
    package_inference = bool(
        arm == ARM if args.package_inference is None else args.package_inference
    )
    root = DEFAULT_OUT_DIR
    out_dir = root / (
        "historical_rank_oos" if arm == ARM else f"historical_rank_oos_{arm}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    burnin = _burnin(root, arm, args.generated_cache)
    oos = pd.read_parquet(root / arm / "oos_predictions.parquet")
    ranked, folds = _walkforward_ranks(burnin, oos)
    ranked["week_start"] = _true_monday_week_start(ranked["__ts__"])
    metrics = _metrics(ranked, arm)
    calendar, autocorr, comparison = _calendar(ranked, arm)
    ranked.to_parquet(
        out_dir / "oos_predictions_historical_rank.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(out_dir / "metrics_by_scope.csv", index=False)
    calendar.to_csv(out_dir / "hit_surprise_calendar.csv", index=False)
    autocorr.to_csv(out_dir / "hit_surprise_autocorrelation.csv", index=False)
    comparison.to_csv(out_dir / "high_surprise_period_comparison.csv", index=False)

    full_history = pd.concat([burnin, oos], ignore_index=True)
    final_reference = HistoricalScoreRankReference(score_col="score_alternative").fit(
        full_history
    )
    bundle_dir = root / "inference_bundle_residual_ae_gmm"
    reference_path = (
        bundle_dir / "alternative_meta_historical_rank_reference.joblib"
        if package_inference
        else out_dir / "historical_rank_reference.joblib"
    )
    joblib.dump(final_reference, reference_path, compress=3)
    restored = joblib.load(reference_path)
    july_rows = 0
    max_diff = np.nan
    bundle_rank_max_diff = np.nan
    if package_inference:
        bundle_path = bundle_dir / "alternative_meta_residual_ae_gmm_bundle.joblib"
        bundle = joblib.load(bundle_path)
        bundle.historical_rank_reference = final_reference
        joblib.dump(bundle, bundle_path, compress=3)
        restored_bundle = joblib.load(bundle_path)
        july = pd.read_parquet(bundle_dir / "july_oos_inference_preview.parquet")
        july["historical_rank_alternative"] = final_reference.transform(
            july, "score_residual_overlay"
        )
        restored_rank = restored.transform(july, "score_residual_overlay")
        max_diff = float(
            np.nanmax(
                np.abs(
                    july["historical_rank_alternative"].to_numpy(dtype=np.float32)
                    - restored_rank.to_numpy(dtype=np.float32)
                )
            )
        )
        bundle_rank_frame = pd.DataFrame(
            {
                final_reference.side_col: july[final_reference.side_col],
                final_reference.score_col: july["score_residual_overlay"],
            },
            index=july.index,
        )
        bundle_rank = restored_bundle.historical_rank_reference.transform(
            bundle_rank_frame
        )
        bundle_rank_max_diff = float(
            np.nanmax(
                np.abs(
                    july["historical_rank_alternative"].to_numpy(dtype=np.float32)
                    - bundle_rank.to_numpy(dtype=np.float32)
                )
            )
        )
        july_rows = int(len(july))
        july.to_parquet(
            bundle_dir / "july_oos_inference_preview_with_historical_rank.parquet",
            index=False,
        )
    overall = metrics[(metrics["scope"].eq("overall")) & metrics["fraction"].eq(0.10)]
    manifest = {
        "schema": "meta_residual_historical_rank_oos_v1",
        "arm": arm,
        "rank_contract": "expanding_prior_score_cdf_by_side",
        "folds": folds,
        "top10": overall.to_dict(orient="records"),
        "final_reference": final_reference.manifest(),
        "reference_path": str(reference_path),
        "july_rows": july_rows,
        "july_rank_roundtrip_max_abs_diff": max_diff,
        "bundle_rank_roundtrip_max_abs_diff": bundle_rank_max_diff,
        "historical_rank_embedded_in_bundle": package_inference,
        "inference_rank_parity_pass": bool(
            not package_inference
            or (july_rows > 0 and max_diff <= 1e-7 and bundle_rank_max_diff <= 1e-7)
        ),
        "current_model_overwritten": False,
    }
    _write_json(out_dir / "manifest.json", manifest)
    if package_inference:
        _write_json(bundle_dir / "historical_rank_manifest.json", manifest)
        bundle_manifest_path = bundle_dir / "manifest.json"
        bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        bundle_manifest["historical_rank"] = final_reference.manifest()
        bundle_manifest["historical_rank_reference_path"] = str(reference_path)
        bundle_manifest["historical_rank_embedded"] = True
        bundle_manifest["historical_rank_parity_pass"] = manifest[
            "inference_rank_parity_pass"
        ]
        bundle_manifest_path.write_text(
            json.dumps(bundle_manifest, indent=2, default=str),
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
