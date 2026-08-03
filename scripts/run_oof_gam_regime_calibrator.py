#!/usr/bin/env python3
"""Additive-spline OOF calibrator for frozen residual-trust sidecars.

The model is a GAM in the practical regression sense: a sum of univariate
splines, fitted with ridge regularisation.  It deliberately has no learned
cross-terms.  Regime and transition inputs remain independently named and are
only combined in the explicitly combined arm.  The action layer is forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, RegimeOOFStackError, assert_outcome_free, validate_candidate_identity  # noqa: E402
from extreme_price_movements.regime_stack_evaluation import EvaluationColumns, evaluate_matched_arms, global_top_k_mask  # noqa: E402
from scripts.run_interaction_conditioned_residual_trust_oof import (  # noqa: E402
    ACTION_TOKENS,
    DEFAULT_RISK,
    DEFAULT_SCORES,
    DEFAULT_SOFT,
    LABEL_DELAY,
    TARGET,
    _reject_action_fields,
    build_panel,
)


SCHEMA = "oof_gam_regime_calibrator_v1"
DEFAULT_TRUST_ROOT = ROOT / "data_perp/artifacts/interaction_conditioned_residual_trust_oof_2023q4_2024_20260730_v1"

SOURCE_ARMS = {
    "baseline_spline": "baseline",
    "regime_gam": "regime_only",
    "transition_gam": "transition_only",
    "combined_gam": "regime_plus_transition",
    "combined_plus_adverse_gam": "regime_plus_transition_plus_adverse_risk",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_lists() -> dict[str, list[str]]:
    """Frozen additive terms. No products or action-layer fields are allowed."""

    raw = ["raw_trust_score"]
    regime = [
        "regime_state_p__0", "regime_state_p__1", "regime_state_p__2",
        "regime_state_entropy", "regime_state_margin", "regime_state_ood_score", "regime_state_uncertainty",
    ]
    transition = [
        "transition_state_p__stable", "transition_state_p__approach", "transition_state_p__immediate_lead",
        "transition_state_p__transition", "transition_state_p__acceleration", "transition_state_p__early_destination",
        "transition_state_p__settled_destination", "transition_active_probability", "transition_state_entropy",
        "transition_state_margin", "transition_state_ood_score", "transition_state_uncertainty",
    ]
    result = {
        "baseline_spline": raw,
        "regime_gam": [*raw, *regime],
        "transition_gam": [*raw, *transition],
        "combined_gam": [*raw, *regime, *transition],
        "combined_plus_adverse_gam": [*raw, *regime, *transition, "adverse_competing_risk_p__regime_plus_transition"],
    }
    for fields in result.values():
        _reject_action_fields(fields)
        assert_outcome_free(pd.DataFrame(columns=fields), extra_forbidden=ACTION_TOKENS)
    return result


def _matrix(train: pd.DataFrame, evaluation: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    _reject_action_fields(features)
    train_x = train.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    evaluation_x = evaluation.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = train_x.median().fillna(0.0)
    return train_x.fillna(median), evaluation_x.fillna(median)


def _fit_isotonic(x: np.ndarray, y: np.ndarray):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 8 or np.unique(x).size < 2:
        mean = float(np.mean(y)) if len(y) else 0.0
        return lambda values: np.full(len(values), mean, dtype=float)
    model = IsotonicRegression(out_of_bounds="clip", increasing="auto").fit(x, y)
    return lambda values: np.asarray(model.predict(np.asarray(values, dtype=float)), dtype=float)


def _fit_gam(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit additive B-splines only, then a prior-label causal EV mapping."""

    train_x, evaluation_x = _matrix(train, evaluation, features)
    y = pd.to_numeric(train[TARGET], errors="coerce").fillna(0.0).to_numpy(float)
    # SplineTransformer operates columnwise; Ridge combines those bases
    # additively, so this cannot learn unrestricted feature interactions.
    gam = Pipeline([
        ("splines", SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear", include_bias=False)),
        ("ridge", Ridge(alpha=2.0)),
    ])
    gam.fit(train_x, y)
    raw_train = np.asarray(gam.predict(train_x), dtype=float)
    raw_eval = np.asarray(gam.predict(evaluation_x), dtype=float)
    mapper = _fit_isotonic(raw_train, y)
    return raw_eval, mapper(raw_eval)


def _load_source_sidecars(root: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for gam_arm, source_arm in SOURCE_ARMS.items():
        path = root / "prediction_sidecars" / f"{source_arm}.parquet"
        if not path.exists():
            raise RegimeOOFStackError(f"required residual-trust sidecar is missing: {path}")
        sidecar = validate_candidate_identity(pd.read_parquet(path))
        required = ["trust_fold_id", "trust_train_end_utc", "raw_trust_score", "mapped_score"]
        missing = [column for column in required if column not in sidecar]
        if missing:
            raise RegimeOOFStackError(f"residual-trust sidecar {source_arm!r} lacks {missing}")
        result[gam_arm] = sidecar
    reference = next(iter(result.values())).loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
    for arm, frame in result.items():
        identity = frame.loc[:, list(IDENTITY_COLUMNS)].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
        if not reference.equals(identity):
            raise RegimeOOFStackError(f"source sidecar {arm!r} does not share exact candidate support")
    return result


def _rank_ic(left: pd.Series, right: pd.Series) -> float:
    x, y = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return float("nan")
    return float(x.loc[valid].rank().corr(y.loc[valid].rank()))


def _side_metrics(frame: pd.DataFrame, *, arm: str, columns: EvaluationColumns, top_fraction: float) -> pd.DataFrame:
    selected = frame.loc[global_top_k_mask(frame, score_col=columns.mapped_score, top_fraction=top_fraction)].copy()
    rows: list[dict[str, Any]] = []
    for side, local in frame.groupby("side_name", observed=True, sort=True):
        selected_side = selected.loc[selected["side_name"].eq(side)]
        rows.append({"arm": arm, "side_name": side, "candidate_rows": int(len(local)), "global_selected_rows": int(len(selected_side)), "alpha_rank_ic": _rank_ic(local[columns.mapped_score], local[columns.alpha_target]), "execution_net_rank_ic": _rank_ic(local[columns.mapped_score], local[columns.net_ev]), "global_topk_mean_net_ev": float(selected_side[columns.net_ev].mean()) if len(selected_side) else float("nan"), "global_topk_hit_rate": float(selected_side[columns.net_ev].gt(0).mean()) if len(selected_side) else float("nan")})
    return pd.DataFrame(rows)


def _calibration(frame: pd.DataFrame, *, arm: str, columns: EvaluationColumns) -> tuple[pd.DataFrame, dict[str, float]]:
    work = frame.loc[:, [columns.mapped_score, columns.net_ev]].copy()
    work["bin"] = pd.qcut(work[columns.mapped_score].rank(method="first"), q=10, labels=False, duplicates="drop")
    output = work.groupby("bin", observed=True).agg(candidate_rows=(columns.net_ev, "size"), mean_mapped_score=(columns.mapped_score, "mean"), mean_realized_net_ev=(columns.net_ev, "mean")).reset_index()
    output["arm"] = arm
    output["signed_calibration_error"] = output["mean_mapped_score"] - output["mean_realized_net_ev"]
    return output, {"calibration_mae_decile": float(output["signed_calibration_error"].abs().mean()), "calibration_bias_decile": float(output["signed_calibration_error"].mean())}


def run(*, output_dir: Path, trust_root: Path = DEFAULT_TRUST_ROOT, soft_path: Path = DEFAULT_SOFT, scores_path: Path = DEFAULT_SCORES, risk_path: Path = DEFAULT_RISK, min_train_rows: int = 5000, top_fraction: float = .10) -> Path:
    output, trust_root = Path(output_dir), Path(trust_root)
    if output.exists():
        raise RegimeOOFStackError(f"refusing to overwrite output: {output}")
    source = _load_source_sidecars(trust_root)
    panel = build_panel(soft_path=Path(soft_path), scores_path=Path(scores_path), risk_path=Path(risk_path))
    features_by_arm = feature_lists()
    gam_sidecars: dict[str, list[pd.DataFrame]] = {arm: [] for arm in SOURCE_ARMS}
    provenance: list[dict[str, Any]] = []
    for gam_arm, source_sidecar in source.items():
        current = panel.merge(source_sidecar, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
        for fold, evaluation in current.groupby("trust_fold_id", sort=True, observed=True):
            evaluation_start = pd.to_datetime(evaluation["trust_train_end_utc"].iloc[0], utc=True)
            # A calibrator can only learn from *previous OOF trust predictions*.
            train = current.loc[(current["__ts__"] + LABEL_DELAY).lt(evaluation_start)].copy()
            if len(train) < int(min_train_rows):
                continue
            raw, mapped = _fit_gam(train, evaluation, features_by_arm[gam_arm])
            gam_sidecars[gam_arm].append(evaluation.loc[:, list(IDENTITY_COLUMNS)].assign(gam_fold_id=str(fold), gam_train_end_utc=evaluation_start, gam_label_available_before_utc=evaluation_start, raw_gam_score=raw, mapped_score=mapped))
            provenance.append({"gam_arm": gam_arm, "source_residual_trust_arm": SOURCE_ARMS[gam_arm], "gam_fold_id": str(fold), "evaluation_start_utc": evaluation_start, "train_rows": int(len(train)), "evaluation_rows": int(len(evaluation)), "train_label_available_max_utc": train["__ts__"].max() + LABEL_DELAY, "basis": "prior OOF residual-trust scores with resolved labels only"})
    if not provenance:
        raise RegimeOOFStackError("no GAM fold has adequate prior OOF training rows")
    output.mkdir(parents=True)
    sidecar_dir = output / "prediction_sidecars"
    sidecar_dir.mkdir()
    columns = EvaluationColumns(mapped_score="mapped_score", alpha_target="__reconstructed_soft_alpha_12h__", net_ev=TARGET, gross_ev="execution_gross_ev_12h", cost="execution_cost_return")
    frames: dict[str, pd.DataFrame] = {}
    for gam_arm, parts in gam_sidecars.items():
        if not parts:
            raise RegimeOOFStackError(f"GAM arm {gam_arm!r} has no valid OOF predictions")
        sidecar = pd.concat(parts, ignore_index=True).sort_values(["__ts__", "candidate_id"], kind="stable")
        validate_candidate_identity(sidecar)
        sidecar.to_parquet(sidecar_dir / f"{gam_arm}.parquet", index=False)
        frames[gam_arm] = panel.merge(sidecar, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
        uncal = source[gam_arm].merge(sidecar.loc[:, list(IDENTITY_COLUMNS)], on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
        frames[f"uncalibrated__{SOURCE_ARMS[gam_arm]}"] = panel.merge(uncal, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    summary, period_metrics, category_stability = evaluate_matched_arms(frames, columns=columns, top_fraction=top_fraction, category_col="regime_state_id")
    calibration_parts, side_parts = [], []
    for arm, frame in frames.items():
        calibration, values = _calibration(frame, arm=arm, columns=columns)
        calibration_parts.append(calibration)
        side_parts.append(_side_metrics(frame, arm=arm, columns=columns, top_fraction=top_fraction))
        latest = period_metrics.loc[(period_metrics["arm"] == arm) & (period_metrics["period_type"] == "month")].sort_values("period", kind="stable").tail(1)
        summary.loc[summary["arm"] == arm, "latest_month"] = latest["period"].iloc[0]
        summary.loc[summary["arm"] == arm, "latest_month_net_ev"] = latest["mean_net_ev"].iloc[0]
        for name, value in values.items():
            summary.loc[summary["arm"] == arm, name] = value
    summary["arm_family"] = np.where(summary["arm"].str.startswith("uncalibrated__"), "uncalibrated_residual_trust", "additive_spline_gam")
    pd.DataFrame(provenance).drop_duplicates().sort_values(["gam_arm", "evaluation_start_utc"], kind="stable").to_parquet(output / "fold_provenance.parquet", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    period_metrics.to_parquet(output / "period_metrics.parquet", index=False)
    category_stability.to_parquet(output / "category_stability.parquet", index=False)
    pd.concat(side_parts, ignore_index=True).to_parquet(output / "side_metrics.parquet", index=False)
    pd.concat(calibration_parts, ignore_index=True).to_parquet(output / "calibration_deciles.parquet", index=False)
    (output / "feature_lists.json").write_text(json.dumps({"model": "additive univariate B-splines + Ridge; no learned interactions", "source_arms": SOURCE_ARMS, "arms": features_by_arm, "forbidden_action_tokens": list(ACTION_TOKENS)}, indent=2, sort_keys=True) + "\n")
    outputs = [output / name for name in ("fold_provenance.parquet", "metrics_summary.csv", "period_metrics.parquet", "category_stability.parquet", "side_metrics.parquet", "calibration_deciles.parquet", "feature_lists.json")] + sorted(sidecar_dir.glob("*.parquet"))
    manifest: dict[str, Any] = {"schema": SCHEMA, "status": "STRICT_PRIOR_OOF_GAM_COMPLETE", "calibrator": "additive spline GAM: SplineTransformer(n_knots=5, degree=3) + Ridge(alpha=2); no interaction bases", "training": "only prior frozen OOF raw residual-trust predictions whose 12h labels resolve before fold evaluation", "selection": {"basis": "pooled_global_post_causal_ev_mapping_top_k", "top_fraction": top_fraction, "per_timestamp_selection": False, "per_side_selection": False}, "source_residual_trust": {"path": str(trust_root.resolve()), "manifest_sha256": _sha(trust_root / "manifest.json")}, "inputs": {str(Path(path).resolve()): _sha(Path(path)) for path in (soft_path, scores_path, risk_path)}, "outputs": {str(path.relative_to(output)): _sha(path) for path in outputs}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output / "manifest.sha256").write_text(_sha(output / "manifest.json") + "  manifest.json\n")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trust-root", type=Path, default=DEFAULT_TRUST_ROOT)
    parser.add_argument("--soft", type=Path, default=DEFAULT_SOFT)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument("--top-fraction", type=float, default=.10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(run(output_dir=args.output_dir, trust_root=args.trust_root, soft_path=args.soft, scores_path=args.scores, risk_path=args.risk, min_train_rows=args.min_train_rows, top_fraction=args.top_fraction))
