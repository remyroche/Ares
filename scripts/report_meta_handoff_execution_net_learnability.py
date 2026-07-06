#!/usr/bin/env python3
"""Month-forward learnability of execution-net replay outcomes.

The regime/source proxy layer can look clean while costed replay remains
negative.  This report tests the next question: given pre-entry selected-row
context plus replay-computable friction proxies, can a meta/execution layer rank
rows by actual replay net return?

This script is diagnostic.  It does not tune deployment thresholds and it does
not claim a frozen replay pass.  With the current artifacts there are only May
and June rows, so the only strict month-forward fold is train May -> validate
June.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None
    _LIGHTGBM_AVAILABLE = False


DEFAULT_INPUT = Path(
    "data_perp/reports/ae_gmm_archetype_validation_status_20260704/"
    "meta_prefeature_regime_source_interaction_audit_v1/meta_regime_context_filter_oos_v1/"
    "meta_regime_handoff_candidates_v1/execution_replay_all_exec_keys_cost1pct_v1/"
    "regime_friction_attribution_v1/meta_handoff_replay_regime_friction_candidates.parquet"
)
DEFAULT_OUT_DIR = DEFAULT_INPUT.parent / "execution_net_learnability_v1"
TOP_FRACTIONS = (0.30, 0.20, 0.10, 0.05)

BASE_NUMERIC_FEATURES = (
    "rank_pct",
    "calibrated_score",
    "meta_regime_score",
    "score_rank_pct_by_month",
    "barrier_pct",
    "policy_sl_return",
    "horizon_hours",
    "barrier_multiplier",
)
FRICTION_NUMERIC_FEATURES = (
    "expected_friction_bps",
    "expected_spread_bps",
    "expected_half_spread_bps",
    "spread_cost_bps",
    "entry_reanchor_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "price_gap_bps",
    "delay_window_range_bps",
    "delay_max_adverse_bps",
    "delay_max_favorable_bps",
)
CONTEXT_CATEGORICAL_FEATURES = (
    "side_name",
    "source_family",
    "candidate_archetype_side_aegmm_entropy_bin",
    "candidate_archetype_side_liquidity_bin",
    "candidate_archetype_side_volatility_bin",
    "candidate_archetype_side_activity_liquidity_bin",
    "candidate_archetype_side_directional_vol_imbalance_bin",
    "candidate_archetype_side_market_dispersion_bin",
    "candidate_volatility_shape_bin",
    "policy_overlay",
)
FEATURE_SETS: dict[str, dict[str, tuple[str, ...]]] = {
    "score_only": {
        "numeric": ("rank_pct", "calibrated_score", "meta_regime_score", "score_rank_pct_by_month"),
        "categorical": (),
    },
    "score_plus_context": {
        "numeric": BASE_NUMERIC_FEATURES,
        "categorical": CONTEXT_CATEGORICAL_FEATURES,
    },
    "score_context_friction": {
        "numeric": BASE_NUMERIC_FEATURES + FRICTION_NUMERIC_FEATURES,
        "categorical": CONTEXT_CATEGORICAL_FEATURES,
    },
}
BASELINE_SCORES = (
    "rank_pct",
    "calibrated_score",
    "meta_regime_score",
)
ORACLE_SCORES = (
    "gross_minus_friction_proxy_bps",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _mean(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _rate(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(arr.clip(0.0, 1.0).mean()) if len(arr) else float("nan")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    return float(x[mask].rank(method="average").corr(y[mask].rank(method="average")))


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["month"] = out["timestamp"].dt.to_period("M").astype(str)
    if "side_name" not in out.columns:
        side = _num(out, "side", 1.0).fillna(1.0)
        out["side_name"] = np.where(side.lt(0.0), "short", "long")
    for col in CONTEXT_CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "missing"
        out[col] = out[col].fillna("missing").astype(str)
    out["net_return"] = _num(out, "net_return")
    out["gross_return"] = _num(out, "gross_return")
    out["expected_friction_bps"] = _num(out, "expected_friction_bps")
    out["gross_minus_friction_proxy_bps"] = out["gross_return"] * 10000.0 - out["expected_friction_bps"]
    exit_reason = out.get("simple_policy_exit_reason", pd.Series("", index=out.index)).astype(str)
    out["replay_full_sl"] = exit_reason.eq("full_sl").astype(float)
    out["replay_timeout"] = exit_reason.eq("timeout").astype(float)
    out["replay_positive_net"] = out["net_return"].gt(0.0).astype(float)
    return out.dropna(subset=["timestamp", "net_return"]).reset_index(drop=True)


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _model(seed: int, *, force_extra_trees: bool = False) -> Any:
    if _LIGHTGBM_AVAILABLE and not force_extra_trees:
        return LGBMRegressor(
            objective="regression",
            n_estimators=120,
            learning_rate=0.045,
            num_leaves=15,
            min_child_samples=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=-1,
        )
    return ExtraTreesRegressor(
        n_estimators=250,
        min_samples_leaf=4,
        max_features=0.8,
        random_state=seed,
        n_jobs=-1,
    )


def _fit_predict(train: pd.DataFrame, valid: pd.DataFrame, *, feature_set: str, target: str, seed: int) -> tuple[pd.Series, dict[str, Any]]:
    spec = FEATURE_SETS[feature_set]
    numeric = [col for col in spec["numeric"] if col in train.columns and col in valid.columns]
    categorical = [col for col in spec["categorical"] if col in train.columns and col in valid.columns]
    if not numeric and not categorical:
        raise ValueError(f"No usable features for feature_set={feature_set}")
    y = _num(train, target).fillna(0.0)
    backend = "lightgbm" if _LIGHTGBM_AVAILABLE else "extra_trees"
    pipe = Pipeline(
        steps=[
            ("prep", _preprocessor(numeric, categorical)),
            ("model", _model(seed)),
        ]
    )
    try:
        pipe.fit(train[numeric + categorical], y)
    except TypeError as exc:
        if "force_all_finite" not in str(exc):
            raise
        backend = "extra_trees_fallback_after_lgbm_sklearn_api_mismatch"
        pipe = Pipeline(
            steps=[
                ("prep", _preprocessor(numeric, categorical)),
                ("model", _model(seed, force_extra_trees=True)),
            ]
        )
        pipe.fit(train[numeric + categorical], y)
    pred = pd.Series(pipe.predict(valid[numeric + categorical]), index=valid.index, dtype="float64")
    info = {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_count": int(len(numeric) + len(categorical)),
        "backend": backend,
    }
    return pred, info


def _top_metrics(frame: pd.DataFrame, score_col: str, frac: float) -> dict[str, Any]:
    valid = frame[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
    tag = f"top{int(round(frac * 100)):02d}"
    if valid.empty:
        return {
            f"{tag}_rows": 0,
            f"{tag}_mean_net": float("nan"),
            f"{tag}_sum_net": float("nan"),
            f"{tag}_hit_net": float("nan"),
            f"{tag}_mean_gross_bps": float("nan"),
            f"{tag}_mean_friction_bps": float("nan"),
            f"{tag}_full_sl": float("nan"),
            f"{tag}_timeout": float("nan"),
            f"{tag}_long_share": float("nan"),
            f"{tag}_short_share": float("nan"),
        }
    top_n = max(1, int(math.ceil(float(frac) * len(valid))))
    selected = valid.sort_values(score_col, ascending=False, kind="mergesort").head(top_n)
    side = selected["side_name"].astype(str)
    return {
        f"{tag}_rows": int(len(selected)),
        f"{tag}_mean_net": _mean(selected["net_return"]),
        f"{tag}_sum_net": float(pd.to_numeric(selected["net_return"], errors="coerce").sum()),
        f"{tag}_hit_net": _rate(selected["net_return"].gt(0.0)),
        f"{tag}_mean_gross_bps": _mean(selected["gross_return"] * 10000.0),
        f"{tag}_mean_friction_bps": _mean(selected["expected_friction_bps"]),
        f"{tag}_full_sl": _rate(selected["replay_full_sl"]),
        f"{tag}_timeout": _rate(selected["replay_timeout"]),
        f"{tag}_long_share": float(side.eq("long").mean()) if len(side) else float("nan"),
        f"{tag}_short_share": float(side.eq("short").mean()) if len(side) else float("nan"),
    }


def _evaluate(
    valid: pd.DataFrame,
    score: pd.Series,
    *,
    score_name: str,
    fold: str,
    train_rows: int,
    leakage_role: str,
    feature_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_frame = valid.copy()
    eval_frame["__score__"] = score.reindex(valid.index)
    row: dict[str, Any] = {
        "fold": fold,
        "score_name": score_name,
        "leakage_role": leakage_role,
        "train_rows": int(train_rows),
        "valid_rows": int(len(valid)),
        "scorable_rows": int(pd.to_numeric(eval_frame["__score__"], errors="coerce").notna().sum()),
        "base_mean_net": _mean(eval_frame["net_return"]),
        "base_hit_net": _rate(eval_frame["net_return"].gt(0.0)),
        "score_net_spearman": _spearman(eval_frame["__score__"], eval_frame["net_return"]),
        "score_gross_minus_friction_spearman": _spearman(eval_frame["__score__"], eval_frame["gross_minus_friction_proxy_bps"]),
        "mae_net": float(mean_absolute_error(eval_frame["net_return"], eval_frame["__score__"]))
        if score_name.startswith("model_") and eval_frame["__score__"].notna().all()
        else float("nan"),
    }
    if feature_info:
        row.update(feature_info)
    for frac in TOP_FRACTIONS:
        row.update(_top_metrics(eval_frame, "__score__", frac))
    return row


def _month_forward_rows(frame: pd.DataFrame, *, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(frame["month"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    for test_month in months[1:]:
        train = frame[frame["month"].astype(str).lt(test_month)].copy()
        valid = frame[frame["month"].astype(str).eq(test_month)].copy()
        if len(train) < 20 or len(valid) < 10:
            continue
        fold = f"train_lt_{test_month}__valid_{test_month}"
        for score_col in BASELINE_SCORES:
            if score_col in valid.columns:
                rows.append(
                    _evaluate(
                        valid,
                        pd.to_numeric(valid[score_col], errors="coerce"),
                        score_name=f"baseline_{score_col}",
                        fold=fold,
                        train_rows=len(train),
                        leakage_role="deployable_pre_entry_score",
                    )
                )
        for score_col in ORACLE_SCORES:
            if score_col in valid.columns:
                rows.append(
                    _evaluate(
                        valid,
                        pd.to_numeric(valid[score_col], errors="coerce"),
                        score_name=f"oracle_{score_col}",
                        fold=fold,
                        train_rows=len(train),
                        leakage_role="oracle_replay_outcome_upper_bound_not_deployable",
                    )
                )
        for feature_set in FEATURE_SETS:
            pred, info = _fit_predict(
                train,
                valid,
                feature_set=feature_set,
                target=target,
                seed=911 + len(feature_set) + len(test_month),
            )
            score_name = f"model_{target}_{feature_set}"
            rows.append(
                _evaluate(
                    valid,
                    pred,
                    score_name=score_name,
                    fold=fold,
                    train_rows=len(train),
                    leakage_role="month_forward_model_score",
                    feature_info=info,
                )
            )
            scored = valid.loc[:, [col for col in ("timestamp", "symbol", "side_name", "month", "scenario", "source_family") if col in valid.columns]].copy()
            scored["score_name"] = score_name
            scored["execution_net_score"] = pred.values
            scored["net_return"] = valid["net_return"].values
            scored["gross_return"] = valid["gross_return"].values
            scored["expected_friction_bps"] = valid["expected_friction_bps"].values
            scored_frames.append(scored)
    return pd.DataFrame(rows), pd.concat(scored_frames, ignore_index=True, sort=False) if scored_frames else pd.DataFrame()


def run_report(*, input_path: Path, out_dir: Path, target: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _prepare_frame(pd.read_parquet(input_path))
    all_rows: list[pd.DataFrame] = []
    all_scored: list[pd.DataFrame] = []
    for scope, group_cols in (
        ("pooled", []),
        ("scenario", ["scenario"]),
    ):
        groups = [(("all",), frame)] if not group_cols else frame.groupby(group_cols, dropna=False, sort=True)
        for key, group in groups:
            if not isinstance(key, tuple):
                key = (key,)
            metrics, scored = _month_forward_rows(group.copy(), target=target)
            if metrics.empty:
                continue
            metrics["scope"] = scope
            for col, value in zip(group_cols or ["group"], key):
                metrics[col] = str(value)
                if not scored.empty:
                    scored[col] = str(value)
            all_rows.append(metrics)
            if not scored.empty:
                scored["scope"] = scope
                all_scored.append(scored)
    metrics = pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    scored = pd.concat(all_scored, ignore_index=True, sort=False) if all_scored else pd.DataFrame()
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["scope", "top10_mean_net", "top10_hit_net", "score_net_spearman"],
            ascending=[True, False, False, False],
            kind="mergesort",
        )
    paths = {
        "metrics": out_dir / "execution_net_learnability_metrics.csv",
        "scored_validation": out_dir / "execution_net_learnability_scored_validation.parquet",
        "manifest": out_dir / "manifest.json",
        "report": out_dir / "execution_net_learnability_report.md",
    }
    metrics.to_csv(paths["metrics"], index=False)
    scored.to_parquet(paths["scored_validation"], index=False)
    manifest = {
        "generated_by": "report_meta_handoff_execution_net_learnability",
        "input_path": str(input_path),
        "out_dir": str(out_dir),
        "target": target,
        "backend": "lightgbm_available" if _LIGHTGBM_AVAILABLE else "extra_trees",
        "effective_model_backends": sorted(metrics["backend"].dropna().astype(str).unique())
        if "backend" in metrics.columns
        else [],
        "rows": int(len(frame)),
        "months": sorted(frame["month"].astype(str).unique()),
        "scenarios": sorted(frame["scenario"].astype(str).unique()) if "scenario" in frame.columns else [],
        "metrics_rows": int(len(metrics)),
        "leakage_contract": (
            "For each validation month, models train only on earlier months. "
            "This diagnostic uses replay outcomes as targets and reports top-k validation replay net."
        ),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    lines = [
        "# Execution-Net Learnability",
        "",
        manifest["leakage_contract"],
        "",
        f"Backend: `{manifest['backend']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        "",
        "## Top Rows",
        "",
    ]
    if metrics.empty:
        lines.append("No fold metrics produced.")
    else:
        display = [
            "scope",
            "scenario",
            "score_name",
            "leakage_role",
            "valid_rows",
            "base_mean_net",
            "score_net_spearman",
            "top30_mean_net",
            "top20_mean_net",
            "top10_mean_net",
            "top10_hit_net",
            "top10_mean_gross_bps",
            "top10_mean_friction_bps",
            "top10_full_sl",
            "top10_timeout",
        ]
        existing = [col for col in display if col in metrics.columns]
        lines.append(metrics[existing].head(30).to_markdown(index=False))
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target", choices=("net_return", "gross_minus_friction_proxy_bps"), default="net_return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(input_path=args.input, out_dir=args.out_dir, target=str(args.target))
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
