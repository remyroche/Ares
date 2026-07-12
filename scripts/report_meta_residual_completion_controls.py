#!/usr/bin/env python3
"""Complete Stage 11 and Stage 15 controls for the residual-meta candidate.

The script does not modify or retrain the current base/meta artifacts.  It
uses fold-local train-only representations for the outcome-free discovery
control and evaluates the frozen historical-rank policy under repeated asset
and timestamp subsamples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_residual_archetypes import (  # noqa: E402
    ResidualArchetypeConfig,
    _choose_gmm,
    _cluster_catalog,
    _descriptor_matrix,
    _time_spread_indices,
)
from scripts.run_meta_residual_pca_representation_ablation import (  # noqa: E402
    _fit_pca,
    _transform_pca,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    DEFAULT_OUT_DIR,
    EVAL_MONTHS,
)

CHAMPION = "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay"
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOME_COLUMNS = [
    "score_meta_base_soft_label",
    "clean_exec",
    "ev_after_1pct",
    "dirty_positive",
    "full_path_bad_mae_1r",
    "timeout",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _reversed_descriptors(desc: pd.DataFrame) -> pd.DataFrame:
    out = desc.copy()
    signed = out["signed_surprise"].to_numpy(dtype=np.float32, copy=True)
    negative = out["negative_surprise"].to_numpy(dtype=np.float32, copy=True)
    positive = out["positive_surprise"].to_numpy(dtype=np.float32, copy=True)
    neg_tail = out["negative_tail"].to_numpy(dtype=np.float32, copy=True)
    pos_tail = out["positive_tail"].to_numpy(dtype=np.float32, copy=True)
    out["signed_surprise"] = -signed
    out["negative_surprise"] = positive
    out["positive_surprise"] = negative
    out["negative_tail"] = pos_tail
    out["positive_tail"] = neg_tail
    return out


def _matched_cluster_mapping(
    left: np.ndarray, right: np.ndarray
) -> tuple[dict[int, int], float]:
    left_ids = np.asarray(sorted(np.unique(left).tolist()), dtype=np.int32)
    right_ids = np.asarray(sorted(np.unique(right).tolist()), dtype=np.int32)
    counts = np.zeros((len(left_ids), len(right_ids)), dtype=np.int64)
    left_pos = {int(value): idx for idx, value in enumerate(left_ids)}
    right_pos = {int(value): idx for idx, value in enumerate(right_ids)}
    for a, b in zip(left, right):
        counts[left_pos[int(a)], right_pos[int(b)]] += 1
    row, col = linear_sum_assignment(-counts)
    mapping = {int(left_ids[a]): int(right_ids[b]) for a, b in zip(row, col)}
    matched = sum(counts[a, b] for a, b in zip(row, col))
    return mapping, float(matched / max(len(left), 1))


def _sign_reversal_control(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff = pd.Timestamp("2026-04-01", tz="UTC")
    train = data.loc[data["__ts__"].lt(cutoff)]
    idx = _time_spread_indices(len(train), 60_000)
    sample = train.iloc[idx].copy()
    config = ResidualArchetypeConfig(
        cluster_candidates=(4,),
        max_cluster_fit_rows=60_000,
        min_cluster_rows=100,
        random_state=20260711,
    )
    desc = _descriptor_matrix(sample, config)
    reversed_desc = _reversed_descriptors(desc)
    _, labels, _ = _choose_gmm(desc, config, 20260711)
    _, reversed_labels, _ = _choose_gmm(reversed_desc, config, 20260711)
    mapping, assignment_match = _matched_cluster_mapping(labels, reversed_labels)
    baseline_catalog, baseline_semantics, _ = _cluster_catalog(desc, labels, "baseline")
    reversed_catalog, reversed_semantics, _ = _cluster_catalog(
        reversed_desc, reversed_labels, "sign_reversed"
    )
    rows: list[dict[str, Any]] = []
    changed_rows = 0
    total_rows = 0
    for cluster, reversed_cluster in sorted(mapping.items()):
        support = int(np.sum(labels == cluster))
        baseline_semantic = baseline_semantics.get(cluster, "missing")
        reversed_semantic = reversed_semantics.get(reversed_cluster, "missing")
        changed = baseline_semantic != reversed_semantic
        changed_rows += support * int(changed)
        total_rows += support
        rows.append(
            {
                "baseline_cluster": cluster,
                "reversed_cluster": reversed_cluster,
                "support_rows": support,
                "baseline_semantic": baseline_semantic,
                "reversed_semantic": reversed_semantic,
                "semantic_changed": changed,
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "sample_rows": int(len(sample)),
        "cluster_assignment_match_after_hungarian": assignment_match,
        "support_weighted_semantic_change_rate": float(
            changed_rows / max(total_rows, 1)
        ),
        "baseline_catalog": baseline_catalog,
        "reversed_catalog": reversed_catalog,
        "pass": bool(changed_rows / max(total_rows, 1) >= 0.50),
    }
    return table, summary


def _variance_ratio(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    if len(values) < 2:
        return np.nan
    mean = float(values.mean())
    total = float(np.mean((values - mean) ** 2))
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for cluster in np.unique(labels):
        part = values[labels == cluster]
        between += (len(part) / len(values)) * float((part.mean() - mean) ** 2)
    return float(between / total)


def _outcome_free_discovery(
    data: pd.DataFrame,
    pca_inputs: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_outputs: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_idx, month in enumerate(EVAL_MONTHS):
        start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
        end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
        train = data.loc[data["__ts__"].lt(start)]
        valid = data.loc[data["__ts__"].ge(start) & data["__ts__"].lt(end)]
        sample_idx = _time_spread_indices(len(train), 60_000)
        train_sample = train.iloc[sample_idx]
        pca_state = _fit_pca(
            train_sample,
            pca_inputs,
            20260711 + fold_idx * 101,
            requested_components=8,
            scaled_clip=8.0,
        )
        train_latent = _transform_pca(train_sample, pca_state).to_numpy(
            dtype=np.float32, copy=False
        )
        valid_latent = _transform_pca(valid, pca_state).to_numpy(
            dtype=np.float32, copy=False
        )
        gmm = GaussianMixture(
            n_components=4,
            covariance_type="diag",
            reg_covar=1e-3,
            max_iter=200,
            random_state=20260711 + fold_idx * 101,
        ).fit(train_latent)
        cluster = gmm.predict(valid_latent).astype(np.int16, copy=False)
        score = pd.to_numeric(valid["score_meta_base_soft_label"], errors="coerce")
        rank = score.groupby(valid["side_name"].astype(str), sort=False).rank(
            method="average", pct=True
        )
        clean = pd.to_numeric(valid["clean_exec"], errors="coerce").to_numpy(
            dtype=np.float32
        )
        surprise = clean - score.to_numpy(dtype=np.float32)
        output = pd.DataFrame(
            {
                "calendar_month": month,
                "cluster": cluster,
                "rank_pct": rank.to_numpy(dtype=np.float32),
                "hit_surprise": surprise,
                "ev_after_1pct": pd.to_numeric(
                    valid["ev_after_1pct"], errors="coerce"
                ).to_numpy(dtype=np.float32),
                "clean_exec": clean,
            }
        )
        output = output.loc[output["rank_pct"].ge(0.80)].reset_index(drop=True)
        fold_outputs.append(output)
        ratio = _variance_ratio(
            output["hit_surprise"].to_numpy(), output["cluster"].to_numpy()
        )
        stats = output.groupby("cluster", sort=True)["hit_surprise"].mean()
        fold_rows.append(
            {
                "calendar_month": month,
                "train_rows": int(len(train)),
                "train_sample_rows": int(len(train_sample)),
                "oos_rows": int(len(valid)),
                "top20_rows": int(len(output)),
                "surprise_between_total_variance_ratio": ratio,
                "cluster_surprise_range": float(stats.max() - stats.min()),
                "pca_effective_rank": float(pca_state["effective_rank"]),
            }
        )
    output = pd.concat(fold_outputs, ignore_index=True)
    cluster_metrics = (
        output.groupby(["calendar_month", "cluster"], sort=True)
        .agg(
            rows=("hit_surprise", "size"),
            mean_hit_surprise=("hit_surprise", "mean"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            clean_rate=("clean_exec", "mean"),
        )
        .reset_index()
    )
    folds = pd.DataFrame(fold_rows)
    summary = {
        "folds": fold_rows,
        "months_with_variance_ratio_ge_1pct": int(
            folds["surprise_between_total_variance_ratio"].ge(0.01).sum()
        ),
        "minimum_cluster_surprise_range": float(folds["cluster_surprise_range"].min()),
        "median_variance_ratio": float(
            folds["surprise_between_total_variance_ratio"].median()
        ),
    }
    summary["pass"] = bool(
        summary["months_with_variance_ratio_ge_1pct"] >= 2
        and summary["minimum_cluster_surprise_range"] >= 0.05
    )
    return cluster_metrics, summary


def _policy_metrics(
    frame: pd.DataFrame, mask: np.ndarray, prefix: str
) -> dict[str, float | int]:
    selected = frame.loc[mask]
    return {
        f"{prefix}_rows": int(len(selected)),
        f"{prefix}_ev": float(
            pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean()
        ),
        f"{prefix}_clean": float(
            pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
        ),
        f"{prefix}_bad_mae": float(
            pd.to_numeric(selected["full_path_bad_mae_1r"], errors="coerce").mean()
        ),
    }


def _subsampling_controls(
    frame: pd.DataFrame, draws: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = frame.copy(deep=False)
    work["utc_day"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    symbols = np.asarray(sorted(work["__symbol__"].astype(str).unique()), dtype=object)
    days = np.asarray(sorted(work["utc_day"].dropna().unique()))
    current_selected = (
        work["historical_rank_current_reference"].to_numpy(dtype=np.float32) >= 0.90
    )
    champion_selected = (
        work["historical_rank_alternative"].to_numpy(dtype=np.float32) >= 0.90
    )
    symbol_values = work["__symbol__"].astype(str).to_numpy()
    day_values = work["utc_day"].to_numpy()
    rng = np.random.default_rng(20260711)
    rows: list[dict[str, Any]] = []
    for kind, population, values in (
        ("asset_universe", symbols, symbol_values),
        ("timestamp_day", days, day_values),
    ):
        for fraction in (0.60, 0.80):
            take = max(1, int(round(len(population) * fraction)))
            for draw in range(draws):
                chosen = rng.choice(population, size=take, replace=False)
                keep = np.isin(values, chosen)
                row: dict[str, Any] = {
                    "kind": kind,
                    "fraction": fraction,
                    "draw": draw,
                    "population_size": int(len(population)),
                    "retained_units": int(take),
                }
                row.update(_policy_metrics(work, keep & current_selected, "current"))
                row.update(_policy_metrics(work, keep & champion_selected, "champion"))
                row["delta_ev"] = float(row["champion_ev"] - row["current_ev"])
                row["delta_clean"] = float(row["champion_clean"] - row["current_clean"])
                rows.append(row)
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["kind", "fraction"], sort=True)
        .agg(
            draws=("draw", "size"),
            champion_ev_mean=("champion_ev", "mean"),
            champion_ev_q025=("champion_ev", lambda x: float(np.quantile(x, 0.025))),
            delta_ev_mean=("delta_ev", "mean"),
            delta_ev_q025=("delta_ev", lambda x: float(np.quantile(x, 0.025))),
            delta_ev_q975=("delta_ev", lambda x: float(np.quantile(x, 0.975))),
            delta_positive_rate=(
                "delta_ev",
                lambda x: float(np.mean(np.asarray(x) > 0.0)),
            ),
            champion_positive_rate=(
                "champion_ev",
                lambda x: float(np.mean(np.asarray(x) > 0.0)),
            ),
            delta_clean_mean=("delta_clean", "mean"),
        )
        .reset_index()
    )
    manifest = {
        "draws_per_configuration": draws,
        "asset_count": int(len(symbols)),
        "utc_day_count": int(len(days)),
        "all_configurations_positive_delta_ci": bool(
            summary["delta_ev_q025"].gt(0.0).all()
        ),
        "all_configurations_positive_champion_ci": bool(
            summary["champion_ev_q025"].gt(0.0).all()
        ),
    }
    manifest["pass"] = bool(
        manifest["all_configurations_positive_delta_ci"]
        and manifest["all_configurations_positive_champion_ci"]
    )
    return results, summary, manifest


def main() -> None:
    root = DEFAULT_OUT_DIR
    report_dir = root / "final_report"
    compact_path = root / "cache" / "compact_reference_with_lifecycle.parquet"
    state_path = root / "states" / "residual_pca_eval_latest_pca8_clip8_baseline.joblib"
    historical_path = (
        root
        / f"historical_rank_oos_{CHAMPION}"
        / "oos_predictions_historical_rank.parquet"
    )
    state = joblib.load(state_path)
    pca_inputs = list(state["pca_state"]["columns"])
    requested = list(dict.fromkeys([*KEYS, *OUTCOME_COLUMNS, *pca_inputs]))
    data = pd.read_parquet(compact_path, columns=requested)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data = data.sort_values("__ts__", kind="stable").reset_index(drop=True)

    sign_table, sign_manifest = _sign_reversal_control(data)
    outcome_free_table, outcome_free_manifest = _outcome_free_discovery(
        data, pca_inputs
    )
    historical = pd.read_parquet(historical_path)
    historical["__ts__"] = pd.to_datetime(
        historical["__ts__"], utc=True, errors="coerce"
    )
    subsample_rows, subsample_summary, subsample_manifest = _subsampling_controls(
        historical
    )

    sign_table.to_csv(report_dir / "stage11_surprise_sign_reversal.csv", index=False)
    outcome_free_table.to_csv(
        report_dir / "stage11_outcome_free_discovery.csv", index=False
    )
    subsample_rows.to_csv(
        report_dir / "stage15_policy_subsampling_draws.csv", index=False
    )
    subsample_summary.to_csv(
        report_dir / "stage15_policy_subsampling_summary.csv", index=False
    )
    manifest = {
        "schema": "meta_residual_completion_controls_v1",
        "stage11": {
            "sign_reversal": sign_manifest,
            "outcome_free_discovery": outcome_free_manifest,
            "pass": bool(sign_manifest["pass"] and outcome_free_manifest["pass"]),
        },
        "stage15": subsample_manifest,
        "current_meta_model_overwritten": False,
    }
    _write_json(report_dir / "completion_controls_manifest.json", manifest)
    print(json.dumps(_json_safe(manifest), indent=2), flush=True)


if __name__ == "__main__":
    main()
