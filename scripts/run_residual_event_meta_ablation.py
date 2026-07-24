#!/usr/bin/env python3
"""Fixed-parameter meta ablation for causal residual-event state features.

The base candidate stream, meta parameters, training rows, OOS rows, target,
costs, and top-k denominator are identical across arms.  Feature selection and
HPO are deliberately frozen: this test asks whether revision-stable residual
state semantics add incremental value before a canonical selector/HPO rerun.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.meta_historical_rank import (  # noqa: E402
    HistoricalScoreRankReference,
)
from extreme_price_movements.residual_event_archetypes import (  # noqa: E402
    ResidualEventArchetypeConfig,
    causal_eight_day_hit_rate_overlay,
    residual_event_distilled_feature_names,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (  # noqa: E402
    META_POST_SELECTION_OOD_FEATURE_NAMES,
    _base_soft_label_target,
    _fit_base_soft_label_model,
    _predict,
)
from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    _add_reference_fold_features,
    _apply_ood_state,
    _calibrate,
    _fit_ood_state,
    _fit_platt,
    _matrix_fit_transform,
)

DEFAULT_STATE_ARTIFACT = Path(
    "data_perp/reports/residual_event_archetype_true_base_oof_"
    "compactlocal_market_20260712_v1/oos_residual_event_states.parquet"
)
DEFAULT_COLUMNS = Path(
    "data_perp/artifacts/s59_s52_frozen_inference_bundle_20260709/"
    "models/meta/2026-07/columns.json"
)
DEFAULT_MODEL_MANIFEST = Path(
    "data_perp/artifacts/s59_s52_frozen_inference_bundle_20260709/"
    "models/meta/2026-07/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "data_perp/reports/residual_event_meta_ablation_true_base_oof_20260712_v1"
)
DEFAULT_EXTERNAL_COMPARATOR = Path(
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay_"
    "sparse_shock_composite/oos_predictions_historical_rank.parquet"
)
EXTERNAL_COMPARATOR_ARM = (
    "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_globaloverlay_"
    "sparse_shock_composite"
)
KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
ARMS = ("baseline", "residual_local", "residual_local_market")


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


def _feature_contract(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("feature_names") if isinstance(payload, dict) else payload
    if not isinstance(names, list) or not names:
        raise ValueError(f"Invalid meta feature contract: {path}")
    return list(dict.fromkeys(map(str, names)))


def _model_params(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("regressor_params", {})
    if not isinstance(params, dict) or not params:
        raise ValueError(f"No regressor_params in {path}")
    return dict(params)


def _arm_features(arm: str, base: Sequence[str], available: Sequence[str]) -> list[str]:
    available_set = set(map(str, available))
    local = [
        name
        for name in residual_event_distilled_feature_names(include_market=False)
        if name in available_set
    ]
    market = [
        name
        for name in residual_event_distilled_feature_names(include_market=True)
        if name.startswith("resid_event_market_aegmm_") and name in available_set
    ]
    if arm == "baseline":
        additions: list[str] = []
    elif arm == "residual_local":
        additions = local
    elif arm == "residual_local_market":
        additions = [*local, *market]
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return list(dict.fromkeys([*base, *additions]))


def _daily_autocorrelation(
    frame: pd.DataFrame, *, surprise_col: str, group_cols: Sequence[str]
) -> pd.DataFrame:
    work = frame.copy(deep=False)
    work["day"] = pd.to_datetime(work["__ts__"], utc=True).dt.floor("D")
    daily = (
        work.groupby([*group_cols, "day"], observed=True, dropna=False)[surprise_col]
        .mean()
        .rename("surprise")
        .reset_index()
    )
    groups = (
        daily.groupby(list(group_cols), observed=True, dropna=False)
        if group_cols
        else [("global", daily)]
    )
    rows: list[dict[str, Any]] = []
    for key, group in groups:
        ordered = group.sort_values("day", kind="stable")
        values = pd.to_numeric(ordered["surprise"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        days = pd.to_datetime(ordered["day"], utc=True)
        prior = np.roll(values, 1)
        valid = days.diff().dt.days.eq(1).to_numpy() & np.isfinite(values) & np.isfinite(prior)
        if len(valid):
            valid[0] = False
        current = values[valid]
        previous = prior[valid]
        corr = (
            float(np.corrcoef(current, previous)[0, 1])
            if len(current) >= 3
            and np.std(current) > 1e-10
            and np.std(previous) > 1e-10
            else np.nan
        )
        payload: dict[str, Any] = {}
        if group_cols:
            key_values = key if isinstance(key, tuple) else (key,)
            payload.update(dict(zip(group_cols, key_values, strict=True)))
        payload.update(
            {
                "surprise_col": surprise_col,
                "days": int(len(ordered)),
                "consecutive_pairs": int(len(current)),
                "signed_mean": float(np.nanmean(values)),
                "signed_lag1_autocorrelation": corr,
                "adverse_lag1_product_mean": float(
                    np.mean(np.maximum(-current, 0.0) * np.maximum(-previous, 0.0))
                )
                if len(current)
                else np.nan,
                "favorable_lag1_product_mean": float(
                    np.mean(np.maximum(current, 0.0) * np.maximum(previous, 0.0))
                )
                if len(current)
                else np.nan,
            }
        )
        rows.append(payload)
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    work = frame.copy(deep=False)
    if groups:
        grouped = work.groupby(list(groups), observed=True, dropna=False)
    else:
        work = work.assign(scope="overall")
        groups = ("scope",)
        grouped = work.groupby(list(groups), observed=True, dropna=False)
    return grouped.agg(
        selected_rows=("score_model", "size"),
        timestamps=("__ts__", "nunique"),
        symbols=("__symbol__", "nunique"),
        mean_ev_after_1pct=("ev_after_1pct", "mean"),
        sum_ev_after_1pct=("ev_after_1pct", "sum"),
        clean_exec_precision=("clean_exec", "mean"),
        dirty_positive_rate=("dirty_positive", "mean"),
        bad_mae_rate=("full_path_bad_mae_1r", "mean"),
        timeout_rate=("timeout", "mean"),
        long_share=("side_name", lambda values: float(values.astype(str).eq("long").mean())),
    ).reset_index()


def _causal_historical_rank(
    *,
    burnin: pd.DataFrame,
    valid: pd.DataFrame,
    score_col: str,
) -> pd.Series:
    """Map scores to side-aware percentiles using only earlier score samples."""

    prior = burnin.loc[:, ["__ts__", "side_name", score_col]].copy()
    output = pd.Series(np.nan, index=valid.index, dtype=np.float32)
    months = pd.to_datetime(valid["__ts__"], utc=True).dt.to_period("M").astype(str)
    for month in sorted(months.dropna().unique()):
        month_index = valid.index[months.eq(month)]
        reference = HistoricalScoreRankReference(score_col=score_col).fit(prior)
        output.loc[month_index] = reference.transform(
            valid.loc[month_index], score_col
        ).to_numpy(dtype=np.float32)
        prior = pd.concat(
            [
                prior,
                valid.loc[month_index, ["__ts__", "side_name", score_col]],
            ],
            ignore_index=True,
            copy=False,
        )
    return output


def _append_metrics(
    *,
    frame: pd.DataFrame,
    mask: np.ndarray | pd.Series,
    score: np.ndarray,
    hit_probability: np.ndarray,
    arm: str,
    fraction: float,
    selection_basis: str,
    metric_frames: list[pd.DataFrame],
    autocorrelation_frames: list[pd.DataFrame],
) -> None:
    selected_mask = np.asarray(mask, dtype=bool)
    chosen = frame.loc[selected_mask].copy()
    chosen["score_model"] = np.asarray(score, dtype=np.float32)[selected_mask]
    chosen["arm"] = arm
    chosen["top_fraction"] = fraction
    chosen["selection_basis"] = selection_basis
    chosen["month"] = chosen["__ts__"].dt.strftime("%Y-%m")
    chosen["week_start"] = chosen["__ts__"].dt.to_period("W-SUN").dt.start_time
    for groups in (
        [],
        ["month"],
        ["week_start"],
        ["side_name"],
        ["side_name", "archetype_policy_key"],
        ["month", "side_name", "archetype_policy_key"],
    ):
        report = _metrics(chosen, groups)
        report["arm"] = arm
        report["top_fraction"] = fraction
        report["selection_basis"] = selection_basis
        report["grouping"] = "overall" if not groups else "_x_".join(groups)
        metric_frames.append(report)

    assessment = chosen.copy()
    assessment["hit_probability"] = np.asarray(
        hit_probability, dtype=np.float32
    )[selected_mask]
    assessment["selected"] = 1
    cfg = ResidualEventArchetypeConfig(
        score_col="score_model", probability_col="hit_probability"
    )
    overlay = causal_eight_day_hit_rate_overlay(
        assessment,
        config=cfg,
        selected_col="selected",
    )
    assessment["model_hit_surprise"] = (
        pd.to_numeric(assessment["clean_exec"], errors="coerce")
        - assessment["hit_probability"]
    ).astype(np.float32)
    assessment["assessment_hr8_surprise"] = overlay[
        "assessment_hr8_surprise"
    ].to_numpy(dtype=np.float32)
    for surprise_col in ("model_hit_surprise", "assessment_hr8_surprise"):
        for groups in ([], ["side_name", "archetype_policy_key"]):
            report = _daily_autocorrelation(
                assessment,
                surprise_col=surprise_col,
                group_cols=groups,
            )
            report["arm"] = arm
            report["top_fraction"] = fraction
            report["selection_basis"] = selection_basis
            report["grouping"] = (
                "overall" if not groups else "side_x_archetype"
            )
            autocorrelation_frames.append(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-artifact", type=Path, default=DEFAULT_STATE_ARTIFACT)
    parser.add_argument("--columns-json", type=Path, default=DEFAULT_COLUMNS)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-end", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-11")
    parser.add_argument("--rank-reference-start", default="2026-03-01")
    parser.add_argument("--embargo-hours", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--external-comparator-predictions",
        type=Path,
        default=DEFAULT_EXTERNAL_COMPARATOR,
    )
    parser.add_argument("--external-comparator-score-col", default="score_adjusted")
    parser.add_argument(
        "--external-comparator-rank-col", default="historical_rank_adjusted"
    )
    parser.add_argument(
        "--external-comparator-hit-probability-col",
        default="hit_prob_adjusted",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.state_artifact)
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    if "archetype_policy_key" not in data:
        raise KeyError("Residual-state artifact lacks archetype_policy_key")
    train_cutoff = pd.Timestamp(args.train_end, tz="UTC")
    train_end = train_cutoff - pd.Timedelta(hours=float(args.embargo_hours))
    rank_reference_start = pd.Timestamp(args.rank_reference_start, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    train = data.loc[data["__ts__"].lt(train_end)].copy()
    valid = data.loc[data["__ts__"].ge(train_cutoff) & data["__ts__"].lt(eval_end)].copy()
    if len(train) < 20_000 or len(valid) < 1_000:
        raise ValueError(f"Insufficient split rows: train={len(train)} valid={len(valid)}")
    train, valid = _add_reference_fold_features(train, valid)
    target, target_col = _base_soft_label_target(train)
    keep = target.notna()
    train = train.loc[keep].copy()
    target = target.loc[keep]

    base_contract = _feature_contract(args.columns_json)
    ood_names = set(META_POST_SELECTION_OOD_FEATURE_NAMES)
    raw_base = [name for name in base_contract if name not in ood_names]
    missing_raw_base = [
        name for name in raw_base if name not in train.columns or name not in valid.columns
    ]
    if missing_raw_base:
        raise KeyError(
            "Residual-state artifact is missing frozen raw meta features: "
            f"{missing_raw_base}"
        )
    x_train_raw, x_valid_raw, _ = _matrix_fit_transform(train, valid, raw_base)
    ood_state = _fit_ood_state(x_train_raw, raw_base)
    x_train_base = _apply_ood_state(x_train_raw, ood_state)
    x_valid_base = _apply_ood_state(x_valid_raw, ood_state)
    params = _model_params(args.model_manifest)

    prediction_columns = [
        *KEYS,
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    predictions = valid.reindex(columns=prediction_columns).copy()
    arm_manifests: dict[str, Any] = {}
    metric_frames: list[pd.DataFrame] = []
    autocorrelation_frames: list[pd.DataFrame] = []
    arm_scores: dict[str, np.ndarray] = {}
    arm_hit_probabilities: dict[str, np.ndarray] = {}
    arm_historical_ranks: dict[str, np.ndarray] = {}
    burnin_mask = train["__ts__"].ge(rank_reference_start)
    if int(burnin_mask.sum()) < 1_000:
        raise ValueError(
            "Historical-rank burn-in is too small: "
            f"rows={int(burnin_mask.sum())} start={rank_reference_start}"
        )
    for arm_idx, arm in enumerate(ARMS):
        selected = _arm_features(arm, base_contract, data.columns)
        residual = [name for name in selected if name.startswith("resid_event_")]
        x_train = x_train_base.copy()
        x_valid = x_valid_base.copy()
        for name in residual:
            median = float(pd.to_numeric(train[name], errors="coerce").median())
            if not np.isfinite(median):
                median = 0.0
            x_train[name] = (
                pd.to_numeric(train[name], errors="coerce").fillna(median).astype(np.float32)
            )
            x_valid[name] = (
                pd.to_numeric(valid[name], errors="coerce").fillna(median).astype(np.float32)
            )
        x_train = x_train.reindex(columns=selected, fill_value=0.0)
        x_valid = x_valid.reindex(columns=selected, fill_value=0.0)
        model = _fit_base_soft_label_model(
            x_train,
            target,
            train,
            int(args.seed + arm_idx * 101),
            lgbm_params=params,
        )
        if model is None:
            raise RuntimeError(f"Meta model did not fit for arm={arm}")
        score = np.asarray(_predict(model, x_valid, classifier=False), dtype=np.float32)
        burnin_score = np.asarray(
            _predict(model, x_train.loc[burnin_mask], classifier=False),
            dtype=np.float32,
        )
        hit_calibrator = _fit_platt(
            pd.Series(burnin_score),
            train.loc[burnin_mask, "clean_exec"].reset_index(drop=True),
        )
        hit_probability = _calibrate(hit_calibrator, pd.Series(score)).astype(
            np.float32
        )
        predictions[f"score__{arm}"] = score
        burnin = train.loc[burnin_mask, ["__ts__", "side_name"]].copy()
        burnin["score_model"] = burnin_score
        rank_frame = valid.loc[:, ["__ts__", "side_name"]].copy()
        rank_frame["score_model"] = score
        historical_rank = _causal_historical_rank(
            burnin=burnin,
            valid=rank_frame,
            score_col="score_model",
        ).to_numpy(dtype=np.float32)
        predictions[f"historical_rank__{arm}"] = historical_rank
        arm_scores[arm] = score
        arm_hit_probabilities[arm] = hit_probability
        arm_historical_ranks[arm] = historical_rank
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"feature": selected}).to_csv(
            arm_dir / "selected_features.csv", index=False
        )
        arm_manifests[arm] = {
            "feature_count": int(len(selected)),
            "residual_feature_count": int(len(residual)),
            "residual_features": residual,
            "rank_reference_rows": int(len(burnin)),
            "rank_reference_start": rank_reference_start,
        }

    external_manifest: dict[str, Any] | None = None
    external_path = args.external_comparator_predictions
    if external_path and external_path.exists():
        external = pd.read_parquet(
            external_path,
            columns=[
                *KEYS,
                args.external_comparator_score_col,
                args.external_comparator_rank_col,
                args.external_comparator_hit_probability_col,
            ],
        )
        external["__ts__"] = pd.to_datetime(external["__ts__"], utc=True)
        external = external.drop_duplicates(list(KEYS), keep="last")
        aligned = valid.loc[:, list(KEYS)].merge(
            external,
            on=list(KEYS),
            how="left",
            validate="one_to_one",
            sort=False,
        )
        external_score = pd.to_numeric(
            aligned[args.external_comparator_score_col], errors="coerce"
        ).to_numpy(dtype=np.float32)
        external_rank = pd.to_numeric(
            aligned[args.external_comparator_rank_col], errors="coerce"
        ).to_numpy(dtype=np.float32)
        external_hit_probability = pd.to_numeric(
            aligned[args.external_comparator_hit_probability_col], errors="coerce"
        ).to_numpy(dtype=np.float32)
        predictions[f"score__{EXTERNAL_COMPARATOR_ARM}"] = external_score
        predictions[f"historical_rank__{EXTERNAL_COMPARATOR_ARM}"] = external_rank
        arm_scores[EXTERNAL_COMPARATOR_ARM] = external_score
        arm_hit_probabilities[EXTERNAL_COMPARATOR_ARM] = external_hit_probability
        arm_historical_ranks[EXTERNAL_COMPARATOR_ARM] = external_rank
        external_manifest = {
            "path": str(external_path),
            "score_col": args.external_comparator_score_col,
            "rank_col": args.external_comparator_rank_col,
            "hit_probability_col": args.external_comparator_hit_probability_col,
            "matched_rows": int(np.isfinite(external_score).sum()),
            "valid_rows": int(len(valid)),
        }

    evaluation = valid.reindex(columns=prediction_columns).copy()
    for arm, score in arm_scores.items():
        finite_score = np.isfinite(score)
        for fraction in (0.10, 0.20):
            threshold = float(np.quantile(score[finite_score], 1.0 - fraction))
            _append_metrics(
                frame=evaluation,
                mask=finite_score & (score >= threshold),
                score=score,
                hit_probability=arm_hit_probabilities[arm],
                arm=arm,
                fraction=fraction,
                selection_basis="global_oos_quantile_diagnostic",
                metric_frames=metric_frames,
                autocorrelation_frames=autocorrelation_frames,
            )
            historical_rank = arm_historical_ranks[arm]
            _append_metrics(
                frame=evaluation,
                mask=np.isfinite(historical_rank)
                & (historical_rank >= 1.0 - fraction),
                score=score,
                hit_probability=arm_hit_probabilities[arm],
                arm=arm,
                fraction=fraction,
                selection_basis="causal_side_historical_rank",
                metric_frames=metric_frames,
                autocorrelation_frames=autocorrelation_frames,
            )

    metrics = pd.concat(metric_frames, ignore_index=True)
    baseline = metrics.loc[metrics["arm"].eq("baseline")].drop(
        columns=["arm"], errors="ignore"
    )
    join_keys = [
        name
        for name in (
            "top_fraction",
            "selection_basis",
            "grouping",
            "scope",
            "month",
            "week_start",
            "side_name",
            "archetype_policy_key",
        )
        if name in metrics.columns
    ]
    delta = metrics.merge(
        baseline,
        on=join_keys,
        how="left",
        suffixes=("", "__baseline"),
    )
    for name in (
        "mean_ev_after_1pct",
        "clean_exec_precision",
        "dirty_positive_rate",
        "bad_mae_rate",
        "timeout_rate",
    ):
        delta[f"delta_{name}"] = delta[name] - delta[f"{name}__baseline"]
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    delta.to_csv(output_dir / "metrics_delta_vs_baseline.csv", index=False)
    pd.concat(autocorrelation_frames, ignore_index=True).to_csv(
        output_dir / "signed_surprise_autocorrelation.csv", index=False
    )
    predictions.to_parquet(
        output_dir / "oos_predictions.parquet", index=False, compression="zstd"
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema": "residual_event_meta_ablation_v1",
            "state_artifact": str(args.state_artifact),
            "columns_json": str(args.columns_json),
            "model_manifest": str(args.model_manifest),
            "target_col": target_col,
            "params": params,
            "train_start": train["__ts__"].min(),
            "train_end": train_end,
            "train_rows": len(train),
            "valid_start": train_cutoff,
            "valid_end": eval_end,
            "valid_rows": len(valid),
            "rank_reference_start": rank_reference_start,
            "arms": arm_manifests,
            "external_comparator": external_manifest,
            "base_feature_contract": {
                "feature_count": len(base_contract),
                "raw_feature_count": len(raw_base),
                "post_selection_ood_feature_count": len(ood_names & set(base_contract)),
                "missing_raw_features": missing_raw_base,
            },
            "leakage_contract": {
                "base_scores": "source artifact must contain chronological base OOF scores",
                "state_features": "only OOS state transforms and revision-stable semantic/uncertainty features",
                "fit": "one meta fit on rows before train_end; all April-July rows are scored by the frozen fit",
                "feature_selection_hpo": "frozen production feature contract and parameters; no OOS tuning",
                "assessment_hr8": "outcome-derived causal smoother used only after scoring",
            },
        },
    )


if __name__ == "__main__":
    main()
