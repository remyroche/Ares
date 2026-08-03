#!/usr/bin/env python3
"""Materialise or fit the Round-1 sequential target-family screen.

Default mode writes the frozen target dictionary and nested-OOF plan.  ``--fit``
performs only Round 1 (T0--T4): no certainty, teacher, GAM, ranking, archetypes
or portfolio constraints.  It refuses G1/G2 because the supplied summary-path
surface cannot recover first-touch order for changed barriers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.sequential_target_family_screen import (
    DEFAULT_SPECS,
    QUANTILES,
    TARGET_ARMS,
    TargetFamilyScreenError,
    attach_triple_barrier_context,
    materialize_target_family_labels,
    nested_oof_fold_plan,
    quantile_expected_value,
    reconcile_quantiles,
    target_family_manifest,
)


SCHEMA = "sequential_target_family_screen_runner_v1"
CAPACITY = {
    "n_estimators": 250, "learning_rate": 0.035, "num_leaves": 15,
    "min_child_samples": 200, "subsample": 0.80, "colsample_bytree": 0.80,
    "reg_lambda": 5.0, "random_state": 20260801, "n_jobs": 1, "verbosity": -1,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _huber(x: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    model = lgb.LGBMRegressor(objective="huber", alpha=0.90, **CAPACITY)
    model.fit(x, y)
    return model.predict(test)


def _quantiles(x: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    result = []
    for alpha in QUANTILES:
        model = lgb.LGBMRegressor(objective="quantile", alpha=float(alpha), **CAPACITY)
        model.fit(x, y)
        result.append(model.predict(test))
    return reconcile_quantiles(np.column_stack(result))


def _conditional_mean(weights: np.ndarray, net: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.dot(weights, net) / total) if total > 1e-8 else float(np.mean(net))


def _base_predict(arm: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit one target-specific base model and emit common-bps expected net.

    The second return is the distributional vector passed (OOF-only) to meta.
    Conditional state means are estimated exclusively from the base fit rows.
    """
    x = train.loc[:, features].to_numpy(np.float32)
    z = test.loc[:, features].to_numpy(np.float32)
    net = train["execution_net_ev_12h"].to_numpy(float)
    if arm == "T0_reconstructed_control":
        p = np.clip(_huber(x, train["target_t0_control"].to_numpy(float), z), 0.0, 1.0)
        y = train["target_t0_control"].to_numpy(float)
        expected = p * _conditional_mean(y, net) + (1.0 - p) * _conditional_mean(1.0 - y, net)
        return expected * 10_000.0, p[:, None]
    if arm == "T1_exact_net_huber":
        score = _huber(x, train["target_t1_net_return"].to_numpy(float), z)
        return score * 10_000.0, score[:, None]
    if arm == "T2_soft_atr_triple_barrier":
        probabilities = np.column_stack([
            np.maximum(_huber(x, train[column].to_numpy(float), z), 0.0)
            for column in ("target_t2_upper_soft", "target_t2_lower_soft", "target_t2_timeout_soft")
        ])
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-8)
        means = np.asarray([
            _conditional_mean(train[column].to_numpy(float), net)
            for column in ("target_t2_upper_soft", "target_t2_lower_soft", "target_t2_timeout_soft")
        ])
        return probabilities @ means * 10_000.0, probabilities
    if arm == "T3_exact_net_multi_quantile":
        q = _quantiles(x, train["target_t3_net_return"].to_numpy(float), z)
        return quantile_expected_value(q) * 10_000.0, q
    if arm == "T4_atr_normalized_net_multi_quantile":
        q_atr = _quantiles(x, train["target_t4_net_atr"].to_numpy(float), z)
        atr = test["competing_risk_atr_fraction"].to_numpy(float)
        q = reconcile_quantiles(q_atr * atr[:, None])
        return quantile_expected_value(q) * 10_000.0, q
    raise TargetFamilyScreenError(f"unknown target arm: {arm}")


def _top_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["target_arm", "model_variant"], observed=True, sort=True):
        ordered = group.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort")
        for fraction in (0.01, 0.05, 0.10, 0.20):
            selected = ordered.head(max(1, int(np.ceil(len(ordered) * fraction))))
            rows.append({
                "target_arm": keys[0], "model_variant": keys[1], "selection": "pooled_global_common_bps",
                "top_fraction": fraction, "population_rows": int(len(ordered)), "selected_rows": int(len(selected)),
                "gross_bps_per_trade": float(selected.execution_gross_ev_12h.mean() * 10_000.0),
                "cost_bps_per_trade": float(selected.execution_cost_return.mean() * 10_000.0),
                "net_bps_per_trade": float(selected.execution_net_ev_12h.mean() * 10_000.0),
                "win_rate": float((selected.execution_net_ev_12h > 0.0).mean()),
            })
    return pd.DataFrame(rows)


def _target_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, group in frame.groupby("target_arm", observed=True, sort=True):
        components = [c for c in group if c.startswith("base_component_")]
        row = {"target_arm": arm, "oof_rows": int(len(group)), "base_score_net_rank_ic": float(group.score_bps.rank().corr(group.execution_net_ev_12h.rank()))}
        if arm == "T2_soft_atr_triple_barrier":
            actual = group[["target_t2_upper_soft", "target_t2_lower_soft", "target_t2_timeout_soft"]].to_numpy(float)
            # The fixed five-column base matrix is padded for the point and
            # quantile targets.  T2 has exactly three meaningful outputs.
            predicted = group[components[:3]].to_numpy(float)
            row.update({
                "soft_log_loss": float(-(actual * np.log(np.clip(predicted, 1e-8, 1.0))).sum(axis=1).mean()),
                "soft_brier": float(np.square(actual - predicted).sum(axis=1).mean()),
                "quantile_crossing_rate": np.nan,
            })
        elif arm in {"T3_exact_net_multi_quantile", "T4_atr_normalized_net_multi_quantile"}:
            q = group[components].to_numpy(float)
            target = group["execution_net_ev_12h"].to_numpy(float)
            row.update({
                "soft_log_loss": np.nan, "soft_brier": np.nan,
                "quantile_crossing_rate": float((np.diff(q, axis=1) < -1e-8).any(axis=1).mean()),
                "q10_coverage": float((target <= q[:, 0] / 10_000.0).mean()),
                "q90_coverage": float((target <= q[:, -1] / 10_000.0).mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _fit(labels: pd.DataFrame, features: list[str], fold_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = labels.copy()
    labels["__ts__"] = pd.to_datetime(labels["__ts__"], utc=True, errors="raise")
    labels["__label_available_at__"] = pd.to_datetime(labels["__label_available_at__"], utc=True, errors="raise")
    folds = labels.groupby(fold_column, observed=True)["__ts__"].min().sort_values(kind="mergesort")
    predictions = []
    lineage = []
    for arm in TARGET_ARMS:
        # A maximum of five distribution components covers both quantile arms.
        # Keep a fixed meta matrix across target families.  Components not
        # emitted by a point target are structural zero padding, not missing
        # upstream predictions; the expected-net column remains the strict
        # OOF dependency checked below.
        component_store = np.zeros((len(labels), 5), dtype=float)
        expected_store = np.full(len(labels), np.nan, dtype=float)
        for position, (fold, test_start) in enumerate(folds.items()):
            if position == 0:
                continue
            test_mask = labels[fold_column].eq(fold).to_numpy()
            train_mask = labels[fold_column].isin(folds.index[:position]).to_numpy()
            train_mask &= labels["__label_available_at__"].lt(test_start).to_numpy()
            if not train_mask.any():
                continue
            expected, components = _base_predict(arm, labels.loc[train_mask], labels.loc[test_mask], features)
            expected_store[test_mask] = expected
            component_store[test_mask, : components.shape[1]] = components
            lineage.append({"target_arm": arm, "layer": "base", "fold": str(fold), "fit_end_ts": labels.loc[train_mask, "__label_available_at__"].max(), "test_start_ts": test_start, "rows": int(test_mask.sum()), "strict_oof": True})
        # Meta has only earlier *emitted* base OOF vectors.  It cannot consume
        # an in-sample upstream base output.
        for position, (fold, test_start) in enumerate(folds.items()):
            if position < 2:
                continue
            test_mask = labels[fold_column].eq(fold).to_numpy()
            meta_mask = labels[fold_column].isin(folds.index[1:position]).to_numpy()
            meta_mask &= labels["__label_available_at__"].lt(test_start).to_numpy()
            meta_mask &= np.isfinite(expected_store)
            if not meta_mask.any():
                continue
            base_cols = np.column_stack((expected_store, component_store))
            x_train = np.column_stack((labels.loc[meta_mask, features].to_numpy(np.float32), base_cols[meta_mask]))
            x_test = np.column_stack((labels.loc[test_mask, features].to_numpy(np.float32), base_cols[test_mask]))
            if not np.isfinite(expected_store[test_mask]).all():
                raise TargetFamilyScreenError("missing base output on a meta test candidate")
            exact_net_bps = labels.loc[meta_mask, "execution_net_ev_12h"].to_numpy(float) * 10_000.0
            residual_bps = exact_net_bps - expected_store[meta_mask]
            correction = _huber(x_train, residual_bps, x_test)
            # A direct meta-only economic learner is deliberately separate
            # from the residual learner.  Reporting the residual correction
            # itself as a trade score would put it in the wrong economic
            # target space and make the requested base/meta comparison
            # meaningless.
            meta_only = _huber(x_train, exact_net_bps, x_test)
            local = labels.loc[test_mask, ["candidate_id", "__ts__", "__decision_ts__", "side_name", "__symbol__", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "target_t2_upper_soft", "target_t2_lower_soft", "target_t2_timeout_soft"]].copy()
            local["target_arm"] = arm
            local["base_expected_net_bps"] = expected_store[test_mask]
            local["meta_only_expected_net_bps"] = meta_only
            local["meta_residual_bps"] = correction
            for index in range(5):
                local[f"base_component_{index + 1}"] = component_store[test_mask, index]
            local["base_prediction_is_strict_oof_for_meta"] = True
            local["meta_prediction_is_strict_oof"] = True
            local["base_fit_end_ts"] = labels.loc[labels[fold_column].isin(folds.index[:position]) & labels["__label_available_at__"].lt(test_start), "__label_available_at__"].max()
            local["meta_fit_end_ts"] = labels.loc[meta_mask, "__label_available_at__"].max()
            local["prediction_fold_id"] = str(fold)
            for variant, score in (
                ("base_only", expected_store[test_mask]),
                ("meta_only", meta_only),
                ("base_plus_meta", expected_store[test_mask] + correction),
            ):
                scored = local.copy()
                scored["model_variant"] = variant
                scored["score_bps"] = score
                predictions.append(scored)
            lineage.append({"target_arm": arm, "layer": "meta", "fold": str(fold), "fit_end_ts": labels.loc[meta_mask, "__label_available_at__"].max(), "test_start_ts": test_start, "rows": int(test_mask.sum()), "strict_oof": True})
    if not predictions:
        raise TargetFamilyScreenError("no nested base-plus-meta OOF predictions were emitted; add chronological folds")
    output = pd.concat(predictions, ignore_index=True)
    return output, _top_metrics(output), pd.DataFrame(lineage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--supportive-labels", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-column", default="oof_fold")
    parser.add_argument("--triple-geometry", choices=("G0", "G1", "G2"), default="G0")
    parser.add_argument("--triple-temperature-atr", type=float, choices=(0.10, 0.25, 0.50), default=0.25)
    parser.add_argument("--fit", action="store_true", help="Run the bounded Round-1 base-plus-meta fits; otherwise materialise only.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output}")
    payload = json.loads(args.features_json.read_text())
    feature_names = payload.get("raw_feature_columns") if isinstance(payload, dict) else payload
    if not isinstance(feature_names, list):
        raise TargetFamilyScreenError("features JSON must contain raw_feature_columns")
    features = list(validate_feature_columns(feature_names))
    # This screen needs the frozen causal matrix plus a very small target
    # dictionary.  Reading the full 238-column supportive surface alongside
    # it creates an unnecessary high-water-memory spike during LightGBM fits.
    # Projecting the source columns is semantically identical and keeps the
    # actual model run practical on the research host.
    ledger_required = {
        "candidate_id", "__ts__", "__decision_ts__", "__label_available_at__", "__symbol__", "side_name",
        "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "__first_touch_target_soft__",
        args.fold_column,
    }
    ledger = pd.read_parquet(args.ledger, columns=sorted(ledger_required | set(features)))
    support_required = {
        "candidate_id", "clean_economic_favorable_first", "adverse_first", "timeout",
        "endpoint_favorable_margin_return", "endpoint_adverse_margin_return", "competing_risk_atr_fraction",
        "same_minute_favorable_adverse_conflict", "path_auxiliary_atr_fraction",
    }
    support = pd.read_parquet(args.supportive_labels, columns=sorted(support_required))
    labels, label_manifest = materialize_target_family_labels(
        attach_triple_barrier_context(ledger, support),
        triple_geometry=args.triple_geometry,
        triple_temperature_atr=args.triple_temperature_atr,
    )
    missing_features = sorted(set(features) - set(labels.columns))
    if missing_features:
        raise TargetFamilyScreenError(f"frozen causal feature names missing from ledger: {missing_features[:12]}")
    contract = target_family_manifest(labels, label_manifest, fold_column=args.fold_column)
    stage = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        labels.to_parquet(stage / "target_family_labels.parquet", index=False, compression="zstd")
        (stage / "target_family_manifest.json").write_text(json.dumps(contract, indent=2, default=str, sort_keys=True) + "\n")
        nested_oof_fold_plan(labels, fold_column=args.fold_column).to_parquet(stage / "nested_oof_fold_plan.parquet", index=False, compression="zstd")
        if args.fit:
            predictions, metrics, lineage = _fit(labels, features, args.fold_column)
            predictions.to_parquet(stage / "base_meta_stack_predictions.parquet", index=False, compression="zstd")
            metrics.to_parquet(stage / "base_meta_stack_results.parquet", index=False, compression="zstd")
            _target_diagnostics(predictions.loc[predictions.model_variant.eq("base_plus_meta")]).to_parquet(stage / "target_family_diagnostics.parquet", index=False, compression="zstd")
            lineage.to_parquet(stage / "nested_oof_lineage.parquet", index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA, "status": "COMPLETED_ROUND1_RESEARCH_ONLY" if args.fit else "MATERIALIZED_ROUND1_READY_FOR_FIT",
            "round": "target family screen only", "fit": bool(args.fit), "target_arms": list(TARGET_ARMS),
            "geometry": args.triple_geometry, "temperature_atr": args.triple_temperature_atr,
            "inputs": {str(path): _sha(path) for path in (args.ledger, args.supportive_labels, args.features_json)},
            "common_controls": "candidate IDs/order, H12, row-cost once, frozen features/folds/seeds, pooled-global common-bps evaluation",
            "excluded_stages": ["certainty", "distillation", "GAM", "ranking", "archetypes", "portfolio_constraints"],
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
