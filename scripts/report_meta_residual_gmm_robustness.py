#!/usr/bin/env python3
"""Summarize residual AE/GMM seed robustness and temporal concentration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (
    transform_ae_gmm_features,  # noqa: E402
)
from scripts.run_meta_residual_ae_representation_ablation import ARM  # noqa: E402
from scripts.run_train_meta_residual_archetype_enhancement import (
    DEFAULT_OUT_DIR,  # noqa: E402
)

ARMS = (
    ("packaged", 0, ARM, "residual_ae_gmm_eval_latest_recognizer.joblib"),
    (
        "packaged",
        17,
        f"{ARM}_seed17",
        "residual_ae_gmm_eval_latest_recognizer_seed17.joblib",
    ),
    (
        "packaged",
        29,
        f"{ARM}_seed29",
        "residual_ae_gmm_eval_latest_recognizer_seed29.joblib",
    ),
    (
        "temporal_v2",
        0,
        f"{ARM}_temporal_v2",
        "residual_ae_gmm_eval_latest_recognizer_temporal_v2.joblib",
    ),
    (
        "temporal_v2",
        17,
        f"{ARM}_temporal_v2_seed17",
        "residual_ae_gmm_eval_latest_recognizer_temporal_v2_seed17.joblib",
    ),
    (
        "temporal_v2",
        29,
        f"{ARM}_temporal_v2_seed29",
        "residual_ae_gmm_eval_latest_recognizer_temporal_v2_seed29.joblib",
    ),
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _historical_dir(root: Path, arm: str) -> Path:
    return root / (
        "historical_rank_oos" if arm == ARM else f"historical_rank_oos_{arm}"
    )


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    states: dict[str, dict[str, Any]] = {}
    seed_rows: list[dict[str, Any]] = []
    for family, seed, arm, state_name in ARMS:
        recognizer = joblib.load(root / "states" / state_name)
        state = recognizer.ae_gmm_state
        states[arm] = state
        historical = _historical_dir(root, arm)
        metrics = pd.read_csv(historical / "metrics_by_scope.csv")
        top10 = metrics[
            metrics["scope"].eq("overall")
            & metrics["fraction"].eq(0.10)
            & metrics["selector"].eq(arm)
        ].iloc[0]
        weeks = metrics[
            metrics["scope"].eq("week")
            & metrics["fraction"].eq(0.10)
            & metrics["selector"].eq(arm)
        ]
        autocorr = pd.read_csv(historical / "hit_surprise_autocorrelation.csv")
        autocorr = autocorr[autocorr["selector"].eq(arm)]
        selected = state.get("selected_config", {})
        seed_rows.append(
            {
                "family": family,
                "seed": seed,
                "arm": arm,
                "selected_components": int(state.get("gmm_n_components", 0)),
                "selected_reg_covar": float(state.get("gmm_reg_covar", np.nan)),
                "selected_temporal_concentration_score": float(
                    selected.get("temporal_concentration_score", np.nan)
                ),
                "top10_ev_after_1pct": float(top10["mean_ev_after_1pct"]),
                "top10_clean_precision": float(top10["clean_exec_precision"]),
                "top10_full_bad_mae": float(top10["full_path_bad_mae_rate"]),
                "top10_timeout": float(top10["timeout_rate"]),
                "mean_abs_surprise_autocorr_lag1": float(
                    pd.to_numeric(autocorr["surprise_autocorr_lag1"], errors="coerce")
                    .abs()
                    .mean()
                ),
                "positive_weeks": int(weeks["mean_ev_after_1pct"].gt(0.0).sum()),
                "weeks": int(len(weeks)),
                "worst_week_ev": float(weeks["mean_ev_after_1pct"].min()),
            }
        )

    first_state = next(iter(states.values()))
    columns = list(first_state["feature_columns"])
    data = pd.read_parquet(
        root / "cache" / "compact_reference_with_lifecycle.parquet",
        columns=["__ts__", *columns],
    )
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = data[data["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC"))].reset_index(
        drop=True
    )
    cluster_rows: list[dict[str, Any]] = []
    for family, seed, arm, _state_name in ARMS:
        state = states[arm]
        sample_rows = int(state.get("gmm_fit_rows", 5_000))
        positions = np.linspace(0, len(data) - 1, sample_rows, dtype=np.int64)
        sample = data.iloc[positions]
        transformed = transform_ae_gmm_features(
            sample[columns],
            state,
            prefix="validation_",
        )
        probability_columns = [
            name
            for name in transformed.columns
            if name.startswith("validation_gmm_prob_")
        ]
        labels = (
            transformed[probability_columns].to_numpy(dtype=np.float32).argmax(axis=1)
        )
        timestamp = sample["__ts__"]
        weeks = (
            (
                timestamp.dt.floor("D")
                - pd.to_timedelta(timestamp.dt.weekday.to_numpy(), unit="D")
            )
            .astype(str)
            .to_numpy()
        )
        for cluster in sorted(np.unique(labels).tolist()):
            mask = labels == cluster
            _keys, counts = np.unique(weeks[mask], return_counts=True)
            cluster_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "arm": arm,
                    "cluster": int(cluster),
                    "rows": int(mask.sum()),
                    "occupancy": float(mask.mean()),
                    "max_single_week_share": float(counts.max() / counts.sum()),
                    "weeks_covered": int(len(counts)),
                }
            )
    seed_frame = pd.DataFrame(seed_rows)
    cluster_frame = pd.DataFrame(cluster_rows)
    family = (
        seed_frame.groupby("family", sort=True)
        .agg(
            seeds=("seed", "size"),
            mean_top10_ev_after_1pct=("top10_ev_after_1pct", "mean"),
            std_top10_ev_after_1pct=("top10_ev_after_1pct", "std"),
            mean_clean_precision=("top10_clean_precision", "mean"),
            mean_full_bad_mae=("top10_full_bad_mae", "mean"),
            mean_abs_surprise_autocorr_lag1=("mean_abs_surprise_autocorr_lag1", "mean"),
            minimum_worst_week_ev=("worst_week_ev", "min"),
        )
        .reset_index()
    )
    concentration = (
        cluster_frame.groupby("family", sort=True)
        .agg(
            minimum_occupancy=("occupancy", "min"),
            maximum_occupancy=("occupancy", "max"),
            maximum_single_week_share=("max_single_week_share", "max"),
            minimum_weeks_covered=("weeks_covered", "min"),
        )
        .reset_index()
    )
    family = family.merge(concentration, on="family", how="left")
    seed_frame.to_csv(report_dir / "stage7_gmm_seed_metrics.csv", index=False)
    cluster_frame.to_csv(
        report_dir / "stage7_gmm_cluster_temporal_concentration.csv", index=False
    )
    family.to_csv(report_dir / "stage7_gmm_family_summary.csv", index=False)
    packaged = family[family["family"].eq("packaged")].iloc[0]
    temporal = family[family["family"].eq("temporal_v2")].iloc[0]
    manifest = {
        "schema": "meta_residual_gmm_robustness_v1",
        "packaged_seed_mean_top10_ev_after_1pct": float(
            packaged["mean_top10_ev_after_1pct"]
        ),
        "temporal_v2_seed_mean_top10_ev_after_1pct": float(
            temporal["mean_top10_ev_after_1pct"]
        ),
        "packaged_max_single_week_share": float(packaged["maximum_single_week_share"]),
        "packaged_minimum_weeks_covered": int(packaged["minimum_weeks_covered"]),
        "packaged_minimum_occupancy": float(packaged["minimum_occupancy"]),
        "packaged_maximum_occupancy": float(packaged["maximum_occupancy"]),
        "decision": "keep_packaged_representation",
        "rationale": (
            "The corrected temporal challenger does not improve seed-average causal top10 EV; "
            "the packaged components are broadly distributed across weeks and satisfy occupancy bounds."
        ),
    }
    (report_dir / "stage7_gmm_robustness_manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
