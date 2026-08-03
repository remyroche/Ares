#!/usr/bin/env python3
"""Discover tree-SHAP and conditional-permutation regime interactions safely.

Discovery is October--December 2023 and evaluation is January--March 2024 by
default.  Regime and transition namespaces remain independent throughout; the
script never uses realized outcome fields as predictors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_oof_stack import IDENTITY_COLUMNS, RegimeOOFStackError, assert_outcome_free, validate_candidate_identity  # noqa: E402


SCHEMA = "oof_regime_transition_interaction_v1"
DEFAULT_SOFT = ROOT / "data_perp/artifacts/reconstructed_2023apr_2024_candidate_oof_regime_transition_20260730_v1/candidate_oof_regime_transition.parquet"
DEFAULT_MULTIVIEW = ROOT / "data_perp/artifacts/fold_local_multiview_selection_2022_2026_20260730_v3"
DEFAULT_SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024q1_20260730_v2/oof_scores.parquet"

OUTCOME_TOKENS = ("target", "label", "execution_", "gross", "cost", "ret_", "outcome", "future", "expost", "realized")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_predictors(columns: Sequence[str]) -> list[str]:
    result = []
    for column in columns:
        low = str(column).lower()
        if any(token in low for token in OUTCOME_TOKENS):
            continue
        if low in {"candidate_id", "__ts__", "__symbol__", "side_name"}:
            continue
        result.append(str(column))
    return result


def _deterministic_sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    value = pd.util.hash_pandas_object(frame["candidate_id"].astype(str), index=False).astype("uint64")
    return frame.assign(__sample_key__=value).nsmallest(int(maximum), "__sample_key__").drop(columns="__sample_key__").copy()


def _load_multiview(path: Path, *, prefix: str, max_features: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "source_utc" not in frame:
        raise RegimeOOFStackError(f"multiview source lacks source_utc: {path}")
    frame["source_utc"] = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
    metadata = {"calendar_segment_id", "fold_id", "evaluation_start_utc", "evaluation_end_exclusive_utc"}
    features = [column for column in _safe_predictors(frame.columns) if column not in metadata and pd.api.types.is_numeric_dtype(frame[column])]
    features = features[: int(max_features)]
    payload = frame.loc[:, ["source_utc", *features]].copy()
    assert_outcome_free(payload.drop(columns="source_utc"))
    return payload.rename(columns={column: f"{prefix}{column}" for column in features})


def _asof(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    joined = pd.merge_asof(
        left.sort_values("__ts__", kind="stable"),
        right.sort_values("source_utc", kind="stable"),
        left_on="__ts__", right_on="source_utc", direction="backward", tolerance=pd.Timedelta(hours=2), allow_exact_matches=True,
    ).sort_values("candidate_id", kind="stable")
    if len(joined) != len(left) or joined["source_utc"].isna().any():
        raise RegimeOOFStackError("multiview as-of join has missing coverage or changed candidate count")
    return joined.drop(columns="source_utc")


def build_panel(*, soft_path: Path, multiview_root: Path, scores_path: Path, max_multiview_features: int) -> pd.DataFrame:
    soft = validate_candidate_identity(pd.read_parquet(soft_path))
    score_columns = [*IDENTITY_COLUMNS, "execution_net_ev_12h", "score_residual_expected_ev"]
    scores = pd.read_parquet(scores_path, columns=score_columns)
    scores = validate_candidate_identity(scores)
    panel = soft.merge(scores, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    if len(panel) != len(scores.loc[scores["candidate_id"].isin(soft["candidate_id"])]):
        raise RegimeOOFStackError("soft-state/score identity overlap is not exact")
    regime_mv = _load_multiview(multiview_root / "regime_oof_features.parquet", prefix="mvreg__", max_features=max_multiview_features)
    transition_mv = _load_multiview(multiview_root / "transition_oof_features.parquet", prefix="mvtrans__", max_features=max_multiview_features)
    first_available = max(regime_mv["source_utc"].min(), transition_mv["source_utc"].min())
    panel = panel.loc[panel["__ts__"].ge(first_available)].copy()
    if panel.empty:
        raise RegimeOOFStackError("no candidate rows overlap causal multiview feature coverage")
    panel = _asof(panel, regime_mv)
    panel = _asof(panel, transition_mv)
    panel["__month__"] = panel["__ts__"].dt.strftime("%Y-%m")
    return panel


def _matrix(train: pd.DataFrame, evaluation: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train.loc[:, features].apply(pd.to_numeric, errors="coerce")
    eval_x = evaluation.loc[:, features].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median().fillna(0.0)
    return train_x.fillna(medians).astype("float32"), eval_x.fillna(medians).astype("float32")


def _interaction_rows(values: np.ndarray, features: list[str]) -> pd.DataFrame:
    absolute = np.abs(values).mean(axis=0)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(features):
        for j in range(i + 1, len(features)):
            right = features[j]
            state_left, state_right = left.startswith("regime_state_"), right.startswith("regime_state_")
            transition_left, transition_right = left.startswith("transition_state_"), right.startswith("transition_state_")
            if state_left ^ state_right:
                family = "regime_x_feature"
            elif transition_left ^ transition_right:
                family = "transition_x_feature"
            elif (state_left and transition_right) or (state_right and transition_left):
                family = "regime_x_transition"
            else:
                continue
            rows.append({"feature_left": left, "feature_right": right, "interaction_family": family, "mean_abs_shap_interaction": float(absolute[i, j])})
    return pd.DataFrame(rows).sort_values("mean_abs_shap_interaction", ascending=False, kind="stable")


def _conditional_permutation(
    model: lgb.LGBMRegressor,
    x: pd.DataFrame,
    y: pd.Series,
    context: pd.Series,
    feature: str,
    seed: int,
) -> tuple[float, float]:
    baseline = mean_squared_error(y, model.predict(x))
    shuffled = x.copy()
    rng = np.random.default_rng(seed)
    for _group, positions in context.groupby(context, dropna=False).groups.items():
        pos = np.asarray(list(positions), dtype=int)
        if len(pos) > 1:
            shuffled.iloc[pos, shuffled.columns.get_loc(feature)] = rng.permutation(shuffled.iloc[pos][feature].to_numpy())
    delta = mean_squared_error(y, model.predict(shuffled)) - baseline
    return float(delta), float(baseline)


def _permutation_report(model: lgb.LGBMRegressor, x: pd.DataFrame, evaluation: pd.DataFrame, features: list[str], max_features: int, seed: int) -> pd.DataFrame:
    y = pd.to_numeric(evaluation["execution_net_ev_12h"], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for layer, context_col in (("regime", "regime_state_id"), ("transition", "transition_state_id")):
        candidate_features = [
            feature
            for feature in features
            if feature == "score_residual_expected_ev" or feature.startswith(("mvreg__", "mvtrans__"))
        ][: int(max_features)]
        for number, feature in enumerate(candidate_features):
            delta, baseline = _conditional_permutation(model, x, y, evaluation[context_col].astype(str).reset_index(drop=True), feature, seed + number)
            monthly: list[float] = []
            for _month, positions in evaluation.groupby("__month__", observed=True).groups.items():
                if len(positions) >= 100:
                    pos = np.asarray(list(positions), dtype=int)
                    monthly.append(_conditional_permutation(model, x.iloc[pos].reset_index(drop=True), y.iloc[pos].reset_index(drop=True), evaluation.iloc[pos][context_col].astype(str).reset_index(drop=True), feature, seed + number)[0])
            rows.append({"context_layer": layer, "feature": feature, "conditional_permutation_delta_mse": delta, "baseline_mse": baseline, "evaluation_rows": int(len(evaluation)), "evaluation_months": int(evaluation["__month__"].nunique()), "positive_month_fraction": float(np.mean(np.asarray(monthly) > 0.0)) if monthly else float("nan"), "month_delta_mse_q10": float(np.quantile(monthly, .10)) if monthly else float("nan")})
    return pd.DataFrame(rows).sort_values(["context_layer", "conditional_permutation_delta_mse"], ascending=[True, False], kind="stable")


def run(*, output_dir: Path, soft_path: Path = DEFAULT_SOFT, multiview_root: Path = DEFAULT_MULTIVIEW, scores_path: Path = DEFAULT_SCORES, discovery_end: str = "2024-01-01", evaluation_end: str = "2024-04-01", max_train_rows: int = 30000, max_eval_rows: int = 12000, max_multiview_features: int = 16, shap_rows: int = 1500, permutation_features: int = 12, seed: int = 52) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise RegimeOOFStackError(f"refusing to overwrite output: {output}")
    panel = build_panel(soft_path=Path(soft_path), multiview_root=Path(multiview_root), scores_path=Path(scores_path), max_multiview_features=max_multiview_features)
    cutoff, end = pd.to_datetime(discovery_end, utc=True), pd.to_datetime(evaluation_end, utc=True)
    train = panel.loc[panel["__ts__"].lt(cutoff)].copy()
    evaluation = panel.loc[panel["__ts__"].ge(cutoff) & panel["__ts__"].lt(end)].copy()
    if train.empty or evaluation.empty or train["__month__"].nunique() < 2 or evaluation["__month__"].nunique() < 2:
        raise RegimeOOFStackError("insufficient distinct discovery/evaluation month support; fail closed")
    train, evaluation = _deterministic_sample(train, max_train_rows), _deterministic_sample(evaluation, max_eval_rows)
    numeric = [column for column in panel.columns if pd.api.types.is_numeric_dtype(panel[column])]
    features = _safe_predictors(numeric)
    # The OOF residual score is a pre-entry model score, permitted as a base
    # context; realized EV remains the response only.
    if "score_residual_expected_ev" not in features:
        features.append("score_residual_expected_ev")
    x_train, x_eval = _matrix(train, evaluation, features)
    model = lgb.LGBMRegressor(n_estimators=280, learning_rate=.035, num_leaves=23, min_child_samples=120, subsample=.85, colsample_bytree=.8, reg_lambda=2.0, random_state=seed, n_jobs=4).fit(x_train, train["execution_net_ev_12h"])
    shap_sample = _deterministic_sample(evaluation, shap_rows)
    _, x_shap = _matrix(train, shap_sample, features)
    interaction = shap.TreeExplainer(model).shap_interaction_values(x_shap)
    interactions = _interaction_rows(np.asarray(interaction), features)
    permutations = _permutation_report(model, x_eval.reset_index(drop=True), evaluation.reset_index(drop=True), features, permutation_features, seed)
    support = pd.concat([
        evaluation.groupby(["__month__", "regime_state_id"], observed=True).size().rename("rows").reset_index().assign(layer="regime"),
        evaluation.groupby(["__month__", "transition_state_id"], observed=True).size().rename("rows").reset_index().assign(layer="transition"),
    ], ignore_index=True)
    output.mkdir(parents=True)
    interactions.to_csv(output / "tree_shap_interactions.csv", index=False)
    permutations.to_csv(output / "conditional_permutation_importance.csv", index=False)
    support.to_csv(output / "state_month_support.csv", index=False)
    manifest = {"schema": SCHEMA, "status": "DISCOVERY_2023_EVALUATION_2024_COMPLETE", "inputs": {"soft": {"path": str(Path(soft_path).resolve()), "sha256": _sha(Path(soft_path))}, "scores": {"path": str(Path(scores_path).resolve()), "sha256": _sha(Path(scores_path))}, "multiview_regime": {"path": str((Path(multiview_root)/'regime_oof_features.parquet').resolve()), "sha256": _sha(Path(multiview_root)/'regime_oof_features.parquet')}, "multiview_transition": {"path": str((Path(multiview_root)/'transition_oof_features.parquet').resolve()), "sha256": _sha(Path(multiview_root)/'transition_oof_features.parquet')}}, "split": {"discovery_before": cutoff.isoformat(), "evaluation": [cutoff.isoformat(), end.isoformat()], "train_rows": int(len(train)), "evaluation_rows": int(len(evaluation)), "train_months": sorted(train['__month__'].unique().tolist()), "evaluation_months": sorted(evaluation['__month__'].unique().tolist())}, "predictor_contract": "candidate soft-state, multiview causal features and OOF residual score only; realized economic columns are response-only", "outputs": {name: _sha(output / name) for name in ("tree_shap_interactions.csv", "conditional_permutation_importance.csv", "state_month_support.csv")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "manifest.sha256").write_text(_sha(output / "manifest.json") + "  manifest.json\n")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--soft", type=Path, default=DEFAULT_SOFT)
    p.add_argument("--multiview-root", type=Path, default=DEFAULT_MULTIVIEW)
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    p.add_argument("--discovery-end", default="2024-01-01")
    p.add_argument("--evaluation-end", default="2024-04-01")
    p.add_argument("--max-train-rows", type=int, default=30000); p.add_argument("--max-eval-rows", type=int, default=12000)
    p.add_argument("--max-multiview-features", type=int, default=16); p.add_argument("--shap-rows", type=int, default=1500); p.add_argument("--permutation-features", type=int, default=12); p.add_argument("--seed", type=int, default=52)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    out = run(output_dir=a.output_dir, soft_path=a.soft, multiview_root=a.multiview_root, scores_path=a.scores, discovery_end=a.discovery_end, evaluation_end=a.evaluation_end, max_train_rows=a.max_train_rows, max_eval_rows=a.max_eval_rows, max_multiview_features=a.max_multiview_features, shap_rows=a.shap_rows, permutation_features=a.permutation_features, seed=a.seed)
    print(json.dumps({"status": "ok", "output_dir": str(out)}, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
