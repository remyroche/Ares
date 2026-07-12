#!/usr/bin/env python3
"""March-select a causal rank blend of PCA overlay and retrained surprise meta."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_meta_residual_historical_rank import (  # noqa: E402
    _burnin,
    _calendar,
    _metrics,
    _true_monday_week_start,
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)

PCA_ARM = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay"
FORCED_ARM = "lifecycle_residual_surprise_head_forced_retrained"
BLEND_ARM = "lifecycle_residual_pca8_forced_retrained_rank_blend"
PCA_CACHE = "residual_walkforward_ae_gmm_eval_mar_jun_pca8_clip8_baseline.parquet"
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _side_rank(frame: pd.DataFrame, score_col: str) -> pd.Series:
    return (
        frame.groupby("side_name", dropna=False)[score_col]
        .rank(method="average", pct=True)
        .astype(np.float32)
    )


def _merge_components(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    right_cols = KEYS + [
        "score_alternative",
        "hit_prob_alternative",
        "historical_rank_alternative",
    ]
    right_cols = [name for name in right_cols if name in right.columns]
    renamed = right[right_cols].rename(
        columns={
            "score_alternative": "score_forced",
            "hit_prob_alternative": "hit_prob_forced",
            "historical_rank_alternative": "rank_forced",
        }
    )
    out = left.merge(renamed, on=KEYS, how="inner", validate="one_to_one")
    out = out.rename(
        columns={
            "score_alternative": "score_pca",
            "hit_prob_alternative": "hit_prob_pca",
            "historical_rank_alternative": "rank_pca",
        }
    )
    return out


def _calendar_abs_ac(frame: pd.DataFrame, rank_col: str, prob_col: str) -> float:
    selected = frame[pd.to_numeric(frame[rank_col], errors="coerce").ge(0.90)].copy()
    selected["date"] = pd.to_datetime(selected["__ts__"], utc=True).dt.floor("D")
    selected["surprise"] = pd.to_numeric(
        selected["clean_exec"], errors="coerce"
    ) - pd.to_numeric(selected[prob_col], errors="coerce")
    daily = (
        selected.groupby(["date", "side_name", "archetype_policy_key"], dropna=False)[
            "surprise"
        ]
        .mean()
        .reset_index()
    )
    values = []
    for _, group in daily.groupby(["side_name", "archetype_policy_key"], dropna=False):
        series = group.sort_values("date")["surprise"]
        values.append(series.autocorr(1) if len(series) >= 3 else np.nan)
    return float(pd.Series(values, dtype=float).abs().mean())


def _burnin_search(root: Path) -> tuple[pd.DataFrame, pd.Series]:
    pca = _burnin(root, PCA_ARM, PCA_CACHE)
    forced = pd.read_parquet(root / FORCED_ARM / "burnin_predictions_march.parquet")
    merged = _merge_components(pca, forced)
    merged["rank_pca"] = _side_rank(merged, "score_pca")
    merged["rank_forced"] = _side_rank(merged, "score_forced")
    rows: list[dict[str, float]] = []
    for weight in np.linspace(0.0, 1.0, 21):
        merged["score_blend"] = (
            (1.0 - weight) * merged["rank_pca"] + weight * merged["rank_forced"]
        ).astype(np.float32)
        merged["rank_blend"] = _side_rank(merged, "score_blend")
        merged["hit_prob_blend"] = (
            (1.0 - weight) * pd.to_numeric(merged["hit_prob_pca"], errors="coerce")
            + weight * pd.to_numeric(merged["hit_prob_forced"], errors="coerce")
        ).astype(np.float32)
        selected = merged[merged["rank_blend"].ge(0.90)]
        ev = float(pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean())
        clean = float(pd.to_numeric(selected["clean_exec"], errors="coerce").mean())
        bad = float(
            pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean()
        )
        timeout = float(pd.to_numeric(selected["timeout"], errors="coerce").mean())
        ac = _calendar_abs_ac(merged, "rank_blend", "hit_prob_blend")
        objective = 100.0 * ev + 0.20 * clean - 0.10 * bad - 0.05 * timeout - 0.20 * ac
        rows.append(
            {
                "forced_weight": float(weight),
                "selected_rows": int(len(selected)),
                "mean_ev_after_1pct": ev,
                "clean_exec_precision": clean,
                "full_path_bad_mae_rate": bad,
                "timeout_rate": timeout,
                "mean_abs_surprise_autocorr_lag1": ac,
                "objective": objective,
            }
        )
    search = pd.DataFrame(rows).sort_values("objective", ascending=False, kind="stable")
    best = search.iloc[0]
    return search, best


def _oos_blend(
    root: Path, weight: float
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pca = pd.read_parquet(
        root
        / f"historical_rank_oos_{PCA_ARM}"
        / "oos_predictions_historical_rank.parquet"
    )
    forced = pd.read_parquet(
        root
        / f"historical_rank_oos_{FORCED_ARM}"
        / "oos_predictions_historical_rank.parquet"
    )
    merged = _merge_components(pca, forced)
    merged["score_blend"] = (
        (1.0 - weight) * pd.to_numeric(merged["rank_pca"], errors="coerce")
        + weight * pd.to_numeric(merged["rank_forced"], errors="coerce")
    ).astype(np.float32)
    merged["hit_prob_alternative"] = (
        (1.0 - weight) * pd.to_numeric(merged["hit_prob_pca"], errors="coerce")
        + weight * pd.to_numeric(merged["hit_prob_forced"], errors="coerce")
    ).astype(np.float32)
    frames: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for month in ("2026-04", "2026-05", "2026-06"):
        valid = merged[merged["calendar_month"].astype(str).eq(month)].copy()
        # Component ranks already use expanding prior score CDFs. Applying a
        # second CDF would break the required zero-weight comparator parity.
        valid["historical_rank_alternative"] = valid["score_blend"]
        valid["score_alternative"] = valid["score_blend"]
        folds.append(
            {
                "month": month,
                "valid_rows": int(len(valid)),
                "rank_contract": "weighted_existing_expanding_prior_component_ranks",
            }
        )
        frames.append(valid)
    output = pd.concat(frames, ignore_index=True)
    output["week_start"] = _true_monday_week_start(output["__ts__"])
    return output, folds


def main() -> None:
    root = DEFAULT_OUT_DIR
    arm_dir = root / BLEND_ARM
    hist_dir = root / f"historical_rank_oos_{BLEND_ARM}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)
    search, best = _burnin_search(root)
    search.to_csv(arm_dir / "burnin_blend_search.csv", index=False)
    weight = float(best["forced_weight"])
    output, folds = _oos_blend(root, weight)
    metrics = _metrics(output, BLEND_ARM)
    calendar, autocorr, comparison = _calendar(output, BLEND_ARM)
    output.to_parquet(
        hist_dir / "oos_predictions_historical_rank.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(hist_dir / "metrics_by_scope.csv", index=False)
    calendar.to_csv(hist_dir / "hit_surprise_calendar.csv", index=False)
    autocorr.to_csv(hist_dir / "hit_surprise_autocorrelation.csv", index=False)
    comparison.to_csv(hist_dir / "high_surprise_period_comparison.csv", index=False)
    overall = metrics[
        metrics["scope"].eq("overall")
        & metrics["fraction"].eq(0.10)
        & metrics["selector"].eq(BLEND_ARM)
    ].iloc[0]
    ac = float(
        pd.to_numeric(
            autocorr.loc[autocorr["selector"].eq(BLEND_ARM), "surprise_autocorr_lag1"],
            errors="coerce",
        )
        .abs()
        .mean()
    )
    manifest = {
        "schema": "meta_residual_retrained_rank_blend_v1",
        "arm": BLEND_ARM,
        "pca_arm": PCA_ARM,
        "forced_retrained_arm": FORCED_ARM,
        "selection_period": "2026-03",
        "selected_forced_weight": weight,
        "burnin_best": best.to_dict(),
        "historical_top10": overall.to_dict(),
        "historical_mean_abs_surprise_autocorr_lag1": ac,
        "folds": folds,
        "current_meta_model_overwritten": False,
        "leakage_contract": "Blend weight selected on March only; April-June ranks use prior component and blend score distributions only.",
    }
    (arm_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    (hist_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
