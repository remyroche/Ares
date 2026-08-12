#!/usr/bin/env python3
"""Compare simpler additive GAMs with portability-aware feature selection.

This is a strict-prior-OOF ablation beside ``run_oof_gam_regime_calibrator``.
The existing GAM uses fixed feature lists and selects no terms from historical
transport behaviour.  Here the model remains additive, but each held fold
selects causal fields using only earlier rows and a portable tail-economics
score:

    median(validation top-k net EV)
      - 0.75 * MAD(validation top-k net EV)
      - max(0, -worst validation top-k net EV)

The selector never sees the held fold.  It is intentionally small and
univariate during selection so it does not turn portability into a second
high-capacity model search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import (  # noqa: E402
    IDENTITY_COLUMNS,
    RegimeOOFStackError,
    assert_outcome_free,
    validate_candidate_identity,
)
from extreme_price_movements.regime_stack_evaluation import (  # noqa: E402
    EvaluationColumns,
    evaluate_matched_arms,
)
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
from scripts.run_oof_gam_regime_calibrator import (  # noqa: E402
    DEFAULT_TRUST_ROOT,
    SOURCE_ARMS,
    _load_source_sidecars,
    feature_lists,
)


SCHEMA = "oof_gam_portability_simplification_ablation_v1"
DEFAULT_CONTROL = ROOT / "data_perp/artifacts/oof_gam_regime_calibrator_2024q2q4_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/oof_gam_portability_simplification_20260812_v1"
SEED = 20260812


VARIANTS: dict[str, dict[str, Any]] = {
    # Fewer basis functions and stronger coefficient shrinkage, but retain the
    # old fixed field contract.  This isolates GAM simplification.
    "simple_quadratic": {
        "n_knots": 3,
        "degree": 2,
        "alpha": 10.0,
        "portable_selection": False,
    },
    # Same simpler GAM, with portability as the primary term-selection score.
    "portable_simple": {
        "n_knots": 3,
        "degree": 2,
        "alpha": 10.0,
        "portable_selection": True,
    },
    # The strongest simplification: piecewise-linear additive terms and a
    # larger ridge penalty.  It is useful as a low-variance control.
    "portable_linear": {
        "n_knots": 2,
        "degree": 1,
        "alpha": 20.0,
        "portable_selection": True,
    },
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    x = pd.Series(np.asarray(a, dtype=float))
    y = pd.Series(np.asarray(b, dtype=float))
    return float(x.rank().corr(y.rank())) if len(x) >= 3 else float("nan")


def _top_mean(y: np.ndarray, pred: np.ndarray, fraction: float = 0.10) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return float("nan")
    y, pred = y[valid], pred[valid]
    n = max(1, int(np.ceil(len(y) * fraction)))
    return float(np.mean(y[np.argpartition(pred, -n)[-n:]]))


def portability_score(values: Sequence[float]) -> float:
    """Return the primary train-only cross-block portability score."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("-inf")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med - 0.75 * mad - max(0.0, -float(np.min(x)))


def _numeric(frame: pd.DataFrame, field: str) -> np.ndarray:
    x = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
    finite = np.isfinite(x)
    if finite.any():
        x = np.where(finite, x, float(np.nanmedian(x[finite])))
    else:
        x = np.zeros(len(frame), dtype=float)
    return x


def _binned_univariate_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, bins: int = 8) -> np.ndarray:
    """Causal, low-variance univariate proxy used only by the selector."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    if len(x_train) < 32 or np.unique(x_train).size < 2:
        return np.full(len(x_eval), float(np.mean(y_train)) if len(y_train) else 0.0)
    edges = np.unique(np.nanquantile(x_train, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return np.full(len(x_eval), float(np.mean(y_train)))
    codes = np.clip(np.digitize(x_train, edges[1:-1], right=False), 0, len(edges) - 2)
    means = np.full(len(edges) - 1, float(np.mean(y_train)), dtype=float)
    for idx in range(len(means)):
        mask = codes == idx
        if mask.any():
            means[idx] = float(np.mean(y_train[mask]))
    eval_codes = np.clip(np.digitize(x_eval, edges[1:-1], right=False), 0, len(edges) - 2)
    return means[eval_codes]


def _nested_blocks(train: pd.DataFrame, max_rows: int = 60000) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return chronological train/validation blocks inside the prior rows."""
    work = train.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if len(work) > max_rows:
        idx = np.linspace(0, len(work) - 1, max_rows, dtype=int)
        work = work.iloc[np.unique(idx)].reset_index(drop=True)
    n = len(work)
    blocks: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    # Three expanding-window validations.  The first quarter is only used as
    # an initial fit; each validation block is strictly later than its fit.
    for i in range(1, 4):
        tr_end = int(np.floor(n * (i / 4.0)))
        va_end = int(np.floor(n * ((i + 1) / 4.0)))
        if va_end <= tr_end or tr_end < 256:
            continue
        fit = work.iloc[:tr_end].copy()
        val = work.iloc[tr_end:va_end].copy()
        val_start = pd.to_datetime(val["__ts__"].iloc[0], utc=True)
        fit = fit.loc[(pd.to_datetime(fit["__ts__"], utc=True) + LABEL_DELAY).lt(val_start)]
        if len(fit) >= 256 and len(val) >= 128:
            blocks.append((fit, val))
    return blocks


def _feature_portability_audit(train: pd.DataFrame, candidates: Sequence[str]) -> pd.DataFrame:
    """Score fields on prior chronological validation blocks only."""
    rows: list[dict[str, Any]] = []
    blocks = _nested_blocks(train)
    for feature in candidates:
        raw_feature = pd.to_numeric(train[feature], errors="coerce").to_numpy(float)
        feature_coverage = float(np.isfinite(raw_feature).mean()) if len(raw_feature) else 0.0
        vals: list[float] = []
        ics: list[float] = []
        for fit, val in blocks:
            y_fit = pd.to_numeric(fit[TARGET], errors="coerce").fillna(0.0).to_numpy(float)
            y_val = pd.to_numeric(val[TARGET], errors="coerce").fillna(0.0).to_numpy(float)
            pred = _binned_univariate_predict(_numeric(fit, feature), y_fit, _numeric(val, feature))
            vals.append(_top_mean(y_val, pred))
            ics.append(_rank_ic(pred, y_val))
        finite_ics = np.asarray(ics, dtype=float)
        finite_ics = finite_ics[np.isfinite(finite_ics)]
        rows.append(
            {
                "feature": feature,
                "validation_blocks": len(vals),
                "top10_values": vals,
                "top10_median": float(np.median(vals)) if vals else float("nan"),
                "top10_mad": float(np.median(np.abs(np.asarray(vals) - np.median(vals)))) if vals else float("nan"),
                "top10_worst": float(np.min(vals)) if vals else float("nan"),
                "portable_score": portability_score(vals),
                "positive_block_fraction": float(np.mean(np.asarray(vals) > 0.0)) if vals else 0.0,
                "median_rank_ic": float(np.median(finite_ics)) if len(finite_ics) else float("nan"),
                "rank_ic_sign_fraction": float(np.mean(finite_ics > 0.0)) if len(finite_ics) else 0.0,
                "coverage": feature_coverage,
            }
        )
    return pd.DataFrame(rows).sort_values(["portable_score", "median_rank_ic", "feature"], ascending=[False, False, True], kind="stable").reset_index(drop=True)


def _max_features(arm: str) -> int:
    if arm == "baseline_spline":
        return 1
    if arm in {"regime_gam", "transition_gam"}:
        return 4
    return 6


def _select_portable_fields(train: pd.DataFrame, candidates: Sequence[str], arm: str) -> tuple[list[str], pd.DataFrame]:
    candidates = [str(x) for x in candidates]
    _reject_action_fields(candidates)
    assert_outcome_free(pd.DataFrame(columns=candidates), extra_forbidden=ACTION_TOKENS)
    audit = _feature_portability_audit(train, candidates)
    # raw_trust_score is the pre-existing trust anchor.  Keep it even if the
    # selector finds a context field with a slightly higher finite-sample score.
    anchor = ["raw_trust_score"] if "raw_trust_score" in candidates else [candidates[0]]
    rest = audit.loc[~audit.feature.isin(anchor)].copy()
    k = max(0, _max_features(arm) - len(anchor))
    # Portability, not pooled fit quality, is the primary ordering.  A mild
    # sign-consistency veto removes fields that reverse on every other block;
    # if that would empty the contract, retain the best portable fields as a
    # diagnostic rather than silently returning an empty model.
    stable = rest.loc[(rest.rank_ic_sign_fraction >= 0.50) | (rest.positive_block_fraction >= 0.50)]
    chosen = stable.head(k) if len(stable) >= k else rest.head(k)
    selected = list(dict.fromkeys(anchor + chosen.feature.astype(str).tolist()))
    return selected, audit


def _fit_gam(train: pd.DataFrame, evaluation: pd.DataFrame, features: Sequence[str], *, n_knots: int, degree: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    train_x = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    eval_x = evaluation.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = train_x.median().fillna(0.0)
    train_x, eval_x = train_x.fillna(median), eval_x.fillna(median)
    y = pd.to_numeric(train[TARGET], errors="coerce").fillna(0.0).to_numpy(float)
    gam = Pipeline(
        [
            ("splines", SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile", extrapolation="linear", include_bias=False)),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    gam.fit(train_x, y)
    raw_train = np.asarray(gam.predict(train_x), dtype=float)
    raw_eval = np.asarray(gam.predict(eval_x), dtype=float)
    valid = np.isfinite(raw_train) & np.isfinite(y)
    if valid.sum() >= 8 and np.unique(raw_train[valid]).size >= 2:
        mapper = IsotonicRegression(out_of_bounds="clip", increasing="auto").fit(raw_train[valid], y[valid])
        mapped = np.asarray(mapper.predict(raw_eval), dtype=float)
    else:
        mapped = np.full(len(raw_eval), float(np.mean(y)) if len(y) else 0.0)
    return raw_eval, mapped


def _load_control_frames(panel: pd.DataFrame, control_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for arm in SOURCE_ARMS:
        path = control_dir / "prediction_sidecars" / f"{arm}.parquet"
        if not path.exists():
            raise RegimeOOFStackError(f"control GAM sidecar missing: {path}")
        sidecar = validate_candidate_identity(pd.read_parquet(path))
        work = panel.merge(sidecar.loc[:, [*IDENTITY_COLUMNS, "mapped_score"]], on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
        work = work.rename(columns={"mapped_score": "__portable_mapped_score"})
        work["mapped_score"] = work["__portable_mapped_score"]
        frames[f"control__{arm}"] = work
    return frames


def run(*, output_dir: Path, trust_root: Path = DEFAULT_TRUST_ROOT, control_dir: Path = DEFAULT_CONTROL, soft_path: Path = DEFAULT_SOFT, scores_path: Path = DEFAULT_SCORES, risk_path: Path = DEFAULT_RISK, min_train_rows: int = 5000, top_fraction: float = 0.10) -> Path:
    output, trust_root, control_dir = Path(output_dir), Path(trust_root), Path(control_dir)
    if output.exists():
        raise RegimeOOFStackError(f"refusing to overwrite output: {output}")
    source = _load_source_sidecars(trust_root)
    panel = build_panel(soft_path=Path(soft_path), scores_path=Path(scores_path), risk_path=Path(risk_path))
    features_by_arm = feature_lists()
    frames: dict[str, pd.DataFrame] = _load_control_frames(panel, control_dir)
    fold_rows: list[dict[str, Any]] = []
    feature_rows: list[pd.DataFrame] = []
    prediction_parts: dict[tuple[str, str], list[pd.DataFrame]] = {
        (variant, arm): [] for variant in VARIANTS for arm in SOURCE_ARMS
    }
    for arm, source_arm in SOURCE_ARMS.items():
        current = panel.merge(source[arm], on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
        candidate_features = list(features_by_arm[arm])
        for fold, evaluation in current.groupby("trust_fold_id", sort=True, observed=True):
            evaluation_start = pd.to_datetime(evaluation["trust_train_end_utc"].iloc[0], utc=True)
            train = current.loc[(pd.to_datetime(current["__ts__"], utc=True) + LABEL_DELAY).lt(evaluation_start)].copy()
            if len(train) < int(min_train_rows):
                continue
            for variant, config in VARIANTS.items():
                if config["portable_selection"]:
                    selected, audit = _select_portable_fields(train, candidate_features, arm)
                    audit = audit.assign(gam_arm=arm, variant=variant, trust_fold_id=str(fold), evaluation_start_utc=evaluation_start, selected=audit.feature.isin(selected))
                    feature_rows.append(audit)
                else:
                    selected = candidate_features
                raw, mapped = _fit_gam(train, evaluation, selected, n_knots=int(config["n_knots"]), degree=int(config["degree"]), alpha=float(config["alpha"]))
                prediction_parts[(variant, arm)].append(
                    evaluation.loc[:, list(IDENTITY_COLUMNS)].assign(
                        gam_fold_id=str(fold),
                        gam_train_end_utc=evaluation_start,
                        raw_gam_score=raw,
                        mapped_score=mapped,
                    )
                )
                fold_rows.append({"variant": variant, "gam_arm": arm, "source_residual_trust_arm": source_arm, "trust_fold_id": str(fold), "evaluation_start_utc": evaluation_start, "train_rows": int(len(train)), "evaluation_rows": int(len(evaluation)), "selected_features": selected, "n_features": len(selected), "n_knots": int(config["n_knots"]), "degree": int(config["degree"]), "ridge_alpha": float(config["alpha"]), "portable_selection": bool(config["portable_selection"]), "label_available_before_evaluation": bool((pd.to_datetime(train["__ts__"], utc=True) + LABEL_DELAY < evaluation_start).all())})
    if not fold_rows:
        raise RegimeOOFStackError("no simplified GAM fold has adequate prior training rows")
    output.mkdir(parents=True)
    sidecar_dir = output / "prediction_sidecars"
    sidecar_dir.mkdir()
    for (variant, arm), parts in prediction_parts.items():
        if not parts:
            raise RegimeOOFStackError(f"no predictions for {variant}/{arm}")
        sidecar = pd.concat(parts, ignore_index=True).sort_values(
            ["__ts__", "candidate_id"], kind="stable"
        )
        validate_candidate_identity(sidecar)
        sidecar.to_parquet(sidecar_dir / f"{variant}__{arm}.parquet", index=False, compression="zstd")
        work = panel.merge(sidecar, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
        frames[f"{variant}__{arm}"] = work

    columns = EvaluationColumns(
        mapped_score="mapped_score",
        alpha_target="__reconstructed_soft_alpha_12h__",
        net_ev=TARGET,
        gross_ev="execution_gross_ev_12h",
        cost="execution_cost_return",
    )
    summary, period_metrics, category_stability = evaluate_matched_arms(
        frames, columns=columns, top_fraction=top_fraction, category_col="regime_state_id"
    )
    summary["arm_family"] = np.where(
        summary["arm"].str.startswith("control__"),
        "sealed_current_gam",
        "simplified_gam",
    )
    summary["variant"] = summary["arm"].str.extract(r"^(?:control__)?([^_]+(?:_[^_]+)*)__", expand=False)
    summary["selection_basis"] = np.where(
        summary["arm"].str.startswith("control__"),
        "sealed_current_control",
        "prior_only_portability" ,
    )
    pd.DataFrame(fold_rows).assign(selected_features=lambda x: x["selected_features"].map(json.dumps)).to_parquet(output / "fold_provenance.parquet", index=False, compression="zstd")
    if feature_rows:
        pd.concat(feature_rows, ignore_index=True).to_parquet(output / "feature_selection_audit.parquet", index=False, compression="zstd")
    else:
        pd.DataFrame(columns=["feature", "portable_score"]).to_parquet(output / "feature_selection_audit.parquet", index=False, compression="zstd")
    summary.to_csv(output / "metrics_summary.csv", index=False)
    period_metrics.to_parquet(output / "period_metrics.parquet", index=False, compression="zstd")
    category_stability.to_parquet(output / "category_stability.parquet", index=False, compression="zstd")
    feature_contract = {
        "schema": SCHEMA,
        "model": "univariate additive B-splines + Ridge; no interaction bases",
        "variants": VARIANTS,
        "source_arms": SOURCE_ARMS,
        "fixed_feature_lists": features_by_arm,
        "fold_selected_features": [
            {"variant": row["variant"], "gam_arm": row["gam_arm"], "fold": row["trust_fold_id"], "features": row["selected_features"]}
            for row in fold_rows
        ],
        "selection_score": "median(top10 validation net EV) - 0.75*MAD - max(0,-worst validation net EV)",
        "selection_is_prior_only": True,
        "ranking_scope": "pooled_global",
        "per_timestamp_selection": False,
        "per_side_selection": False,
    }
    (output / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2, sort_keys=True, default=str) + "\n")
    inputs = (Path(soft_path), Path(scores_path), Path(risk_path))
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "control_dir": str(control_dir.resolve()),
        "control_manifest_sha256": _sha(control_dir / "manifest.json"),
        "source_trust_root": str(trust_root.resolve()),
        "source_manifest_sha256": _sha(trust_root / "manifest.json"),
        "candidate_rows": int(len(panel)),
        "prediction_rows_by_arm": {arm: int(len(value)) for arm, value in frames.items()},
        "variants": VARIANTS,
        "strict_prior_oof": True,
        "held_outcomes_used_for_selection": False,
        "selection_primary": "portable_tail_economics",
        "inputs": {str(path.resolve()): _sha(path) for path in inputs},
        "outputs": [str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output / "manifest.sha256").write_text(_sha(output / "manifest.json") + "  manifest.json\n")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trust-root", type=Path, default=DEFAULT_TRUST_ROOT)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--soft", type=Path, default=DEFAULT_SOFT)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--min-train-rows", type=int, default=5000)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(run(output_dir=args.output_dir, trust_root=args.trust_root, control_dir=args.control_dir, soft_path=args.soft, scores_path=args.scores, risk_path=args.risk, min_train_rows=args.min_train_rows, top_fraction=args.top_fraction))
