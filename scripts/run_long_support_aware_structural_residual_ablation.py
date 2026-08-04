#!/usr/bin/env python3
"""Stage-B support-aware residual ablation on the long structural sidecar.

Support is an outcome-derived H12 label.  It is used only as a *training*
weight or to fit a prequential classifier.  Every residual ranker is trained
on its complete, horizon-purged candidate population and every test candidate
is retained for pooled-global evaluation.

Run the independent S1/S2/S3 stage first.  S4 is intentionally gated behind
``--include-s4`` so a weight is not combined with the causal support signal
until their individual evidence exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.support_aware_pairwise_objective import (
    SupportAwarePairwiseColumns,
    SupportAwarePairwiseConfig,
    build_support_aware_pairwise_objective,
)
from extreme_price_movements.support_aware_residual_ablation import (
    SupportPredictionConfig,
    atr_residual_grade,
    bps_residual_grade,
    query_normalised_support_weights,
    strict_prequential_support_probabilities,
)
from scripts.run_long_only_executable_net_lambdarank import _fit_bps_map, _predict_ranker, _write_json
from scripts.run_long_structural_residual_ablation import (
    _finite_target,
    _load,
    _purged_meta_train,
    _tail_metrics_with_fold_rows,
    _validate_fold_partitions,
    feature_arms,
)


SCHEMA = "long_support_aware_structural_residual_ablation_v1"
SUPPORT_COLUMN = "support_h12_training_only"
SUPPORT_PROBABILITY = "prequential_support_h12_probability"
PAIRWISE_CONTROL_COLUMN = "pairwise_control_constant_false"


def _fit_weighted_ranker(
    frame: pd.DataFrame, fields: list[str], label: np.ndarray, weights: np.ndarray, *, seed: int,
):
    """Fit LambdaRank with query-only grouping and an optional training weight."""

    from lightgbm import LGBMRanker

    if len(frame) != len(label) or len(frame) != len(weights):
        raise ValueError("ranker frame, labels, and weights lost alignment")
    order = np.argsort(frame["query_id"].astype(str).to_numpy(), kind="stable")
    ordered = frame.iloc[order]
    values = ordered.loc[:, fields].replace([np.inf, -np.inf], np.nan)
    groups = ordered.groupby("query_id", sort=False, observed=True).size().to_numpy(dtype=np.int32)
    if len(groups) < 2 or (groups <= 0).any():
        raise ValueError("residual LambdaRank needs non-empty timestamp-side groups")
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=list(range(5)),
        n_estimators=500, learning_rate=0.04, num_leaves=24, max_depth=-1,
        min_child_samples=max(50, int(np.ceil(0.015 * len(values)))),
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=1.5, reg_lambda=4.0, lambdarank_truncation_level=10,
        random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(values, label[order], group=groups, sample_weight=weights[order])
    return model, {"rows": int(len(values)), "groups": int(len(groups)), "feature_count": int(len(fields))}


def _fit_custom_pairwise(
    frame: pd.DataFrame, fields: list[str], *, support_column: str,
    supported_multiplier: float, seed: int,
):
    """Fit the bounded custom logistic residual-pair objective on all rows."""

    import lightgbm as lgb

    order = np.argsort(frame["query_id"].astype(str).to_numpy(), kind="stable")
    ordered = frame.iloc[order].copy()
    residual_bps, residual_atr = _finite_target(ordered)
    ordered["candidate_residual_bps"] = residual_bps
    ordered["atr_residual"] = residual_atr
    ordered["atr_residual_grade"] = atr_residual_grade(residual_atr)
    objective = build_support_aware_pairwise_objective(
        ordered,
        columns=SupportAwarePairwiseColumns(
            support=support_column, incumbent_base_score="base_expected_bps",
        ),
        config=SupportAwarePairwiseConfig(
            max_pairs_per_query=256, random_state=seed,
            # Pairwise losses do not accept row weights.  The S4 branch is
            # the closest bounded analogue.  The control branch uses a
            # constant-false non-feature and all multipliers equal to one.
            both_supported_multiplier=float(supported_multiplier),
            winner_supported_multiplier=float(supported_multiplier),
            loser_supported_multiplier=1.0, neither_supported_multiplier=1.0,
        ),
    )
    if objective.pair_count == 0:
        raise ValueError("custom pairwise objective did not retain any bounded residual pairs")
    dataset = lgb.Dataset(
        ordered.loc[:, fields].replace([np.inf, -np.inf], np.nan),
        label=np.zeros(len(ordered), dtype=np.float32), free_raw_data=False,
    )
    model = lgb.train(
        {
            "objective": objective.lightgbm_objective(), "metric": "None", "verbosity": -1,
            "num_leaves": 24, "learning_rate": 0.04,
            "min_data_in_leaf": max(50, int(np.ceil(0.015 * len(ordered)))),
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l1": 1.5, "lambda_l2": 4.0, "seed": seed,
            "feature_pre_filter": False, "num_threads": 4,
        }, dataset, num_boost_round=500,
    )
    return model, objective.audit.to_dict()


def _support_probability_by_partition(
    source: pd.DataFrame, score_frame: pd.DataFrame, fields: list[str], *, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly prequential score and compact cutoff evidence for one fold."""

    source = source.loc[source["label_valid"].fillna(False)].copy()
    if source.empty or source[SUPPORT_COLUMN].isna().any():
        raise ValueError("support source must contain complete path labels only")
    pieces: list[pd.DataFrame] = []
    audit: list[dict] = []
    config = SupportPredictionConfig()
    for partition in ("meta_train", "meta_calibration", "test"):
        score = score_frame.loc[score_frame["meta_partition"].eq(partition)].copy()
        prediction = strict_prequential_support_probabilities(
            source, score, feature_columns=fields, support_column=SUPPORT_COLUMN,
            config=config, seed=seed,
        )
        result = score.loc[:, ["candidate_id", "__ts__", "meta_partition"]].reset_index(drop=True)
        result[SUPPORT_PROBABILITY] = prediction["predicted_support_probability"].to_numpy(np.float32)
        result["support_model_fit_cutoff_ts"] = prediction["support_model_fit_cutoff_ts"].to_numpy()
        if (result["support_model_fit_cutoff_ts"] > result["__ts__"]).any():
            raise AssertionError("support model was fitted after a scored decision")
        pieces.append(result)
        block_start = score["__ts__"].dt.floor(f"{config.refit_days}D")
        for start in sorted(block_start.drop_duplicates().tolist()):
            prior = source.loc[source["label_available_ts"] < start]
            audit.append({
                "meta_partition": partition, "score_block_start_ts": start,
                "prior_resolved_rows": int(len(prior)),
                "prior_resolved_support_rate": float(prior[SUPPORT_COLUMN].mean()) if len(prior) else np.nan,
                "prior_resolved_label_max_ts": prior["label_available_ts"].max() if len(prior) else pd.NaT,
                "strict_predicate": "label_available_ts < score_block_start_ts",
            })
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(audit)


def _native_specs(*, include_s4: bool, s4_weight: float, include_r1: bool) -> list[tuple[str, str, float | None, bool]]:
    """(feature arm, support arm, supported weight, include support probability)."""

    feature_arms = ["R3_portability_health"] + (["R1_reasoning_memberships"] if include_r1 else [])
    specs: list[tuple[str, str, float | None, bool]] = []
    for arm in feature_arms:
        specs.extend([
            (arm, "S1_uniform", None, False),
            (arm, "S2_weight_1_5", 1.5, False),
            (arm, "S2_weight_2_0", 2.0, False),
            (arm, "S2_weight_3_0", 3.0, False),
            (arm, "S3_prequential_support_probability", None, True),
        ])
        if include_s4:
            specs.append((arm, f"S4_weight_{s4_weight:g}_plus_prequential_probability", s4_weight, True))
    return specs


def _custom_control_specs(*, include_r1: bool) -> list[tuple[str, str, float | None, bool]]:
    """Minimal no-support native control(s) for the custom pairwise check."""

    arms = ["R3_portability_health"] + (["R1_reasoning_memberships"] if include_r1 else [])
    return [(arm, "S1_uniform", None, False) for arm in arms]


def _prepare_support_label(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # This is exactly the frozen H12 3-ATR *and* 1.5%-move condition under the
    # legacy sidecar's `mfe_mae_label_valid` name.  It is never copied into a
    # model feature matrix below.
    result[SUPPORT_COLUMN] = result["mfe_mae_label_valid"].astype("boolean").fillna(False).astype(bool)
    return result


def _serialise_audit_for_parquet(rows: list[dict]) -> pd.DataFrame:
    """Make bounded pair ledgers Arrow-safe without losing audit detail."""

    result = pd.DataFrame(rows)
    if "selected_pairs_by_query" in result:
        result["selected_pairs_by_query"] = result["selected_pairs_by_query"].map(
            lambda value: json.dumps(value, separators=(",", ":"), default=str)
            if isinstance(value, (list, tuple)) else value,
        )
    return result


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite populated output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the inference feature contract *before* adding the outcome-derived
    # support training label.  `feature_arms` intentionally rejects that
    # column, so this ordering is an executable leakage guard.
    if args.custom_control_only and args.include_s4:
        raise ValueError("custom control forbids S4 after the independent S2/S3 failure")
    base_frame = _load(Path(args.sidecar))
    feature_sets = feature_arms(base_frame)
    # The custom control does not even materialise an outcome-derived support
    # label.  Its pair objective receives a constant-false bookkeeping column
    # with all multipliers fixed at one.
    frame = base_frame if args.custom_control_only else _prepare_support_label(base_frame)
    support_model_fields = feature_sets["R3_portability_health"]
    if SUPPORT_COLUMN in support_model_fields or SUPPORT_PROBABILITY in support_model_fields:
        raise AssertionError("outcome-derived support label entered an inference feature contract")

    prediction_parts: list[pd.DataFrame] = []
    support_prediction_parts: list[pd.DataFrame] = []
    support_audits: list[pd.DataFrame] = []
    fit_audits: list[dict] = []
    native_specs = (
        _custom_control_specs(include_r1=bool(args.include_r1_comparison))
        if args.custom_control_only else _native_specs(
            include_s4=bool(args.include_s4), s4_weight=float(args.s4_supported_weight),
            include_r1=bool(args.include_r1_comparison),
        )
    )
    for fold_number, (fold, block) in enumerate(frame.groupby("fold", sort=True, observed=True), start=1):
        _validate_fold_partitions(block, str(fold))
        unpurged_train = block.loc[block["meta_partition"].eq("meta_train")].copy()
        calibration = block.loc[block["meta_partition"].eq("meta_calibration")].copy()
        test = block.loc[block["meta_partition"].eq("test")].copy()
        train = _purged_meta_train(unpurged_train, calibration, str(fold))
        if not args.custom_control_only:
            support_predictions, support_audit = _support_probability_by_partition(
                block, block, support_model_fields, seed=20260860 + fold_number,
            )
            support_predictions["fold"] = str(fold)
            support_audit["fold"] = str(fold)
            support_audits.append(support_audit)
            support_prediction_parts.append(support_predictions)
            block = block.merge(
                support_predictions.loc[:, ["candidate_id", SUPPORT_PROBABILITY, "support_model_fit_cutoff_ts"]],
                on="candidate_id", how="left", validate="one_to_one",
            )
        train = block.loc[block["candidate_id"].isin(train["candidate_id"])].copy()
        calibration = block.loc[block["meta_partition"].eq("meta_calibration")].copy()
        test = block.loc[block["meta_partition"].eq("test")].copy()
        if not args.custom_control_only and not np.isfinite(block[SUPPORT_PROBABILITY]).all():
            raise ValueError(f"{fold}: strict support probabilities are incomplete")

        residual_train, _ = _finite_target(train)
        label = bps_residual_grade(residual_train, moderate_bps=50.0, severe_bps=150.0)
        result = test.loc[:, ["candidate_id", "__ts__", "month", "fold", "gross_bps", "net_bps", "base_expected_bps"]].copy()
        for spec_number, (feature_arm, support_arm, supported_weight, use_probability) in enumerate(native_specs, start=1):
            fields = list(feature_sets[feature_arm])
            if use_probability:
                fields.append(SUPPORT_PROBABILITY)
            if SUPPORT_COLUMN in fields:
                raise AssertionError("support label must not be an inference feature")
            weights = (
                np.ones(len(train), dtype=np.float32)
                if supported_weight is None
                else query_normalised_support_weights(
                    train, support_column=SUPPORT_COLUMN, query_column="query_id", supported_weight=supported_weight,
                )
            )
            model, audit = _fit_weighted_ranker(
                train, fields, label, weights, seed=20260870 + fold_number * 100 + spec_number,
            )
            calibration_raw = _predict_ranker(model, calibration, fields)
            calibration_residual, _ = _finite_target(calibration)
            mapper = _fit_bps_map(calibration_raw, calibration_residual)
            score = f"{feature_arm}__{support_arm}__native_bps_50_150"
            result[score] = (
                test["base_expected_bps"].to_numpy(float) + mapper.predict(_predict_ranker(model, test, fields))
            ).astype(np.float32)
            fit_audits.append({
                "fold": str(fold), "score": score, "implementation": "native_lambdarank",
                "feature_arm": feature_arm, "support_arm": support_arm,
                "feature_count": len(fields), "supported_weight": supported_weight,
                "uses_causal_support_probability": use_probability,
                "meta_train_rows_before_horizon_purge": len(unpurged_train), "meta_train_rows": len(train),
                "meta_calibration_rows": len(calibration), "test_rows": len(test),
                "universal_residual_training": True, **audit,
            })

        if args.custom_control_only:
            # The only custom-pairwise control: same R3 features and direct
            # residual mapping as native S1, but no support label/probability.
            fields = list(feature_sets["R3_portability_health"])
            pair_train = train.copy()
            pair_train[PAIRWISE_CONTROL_COLUMN] = False
            model, pair_audit = _fit_custom_pairwise(
                pair_train, fields, support_column=PAIRWISE_CONTROL_COLUMN,
                supported_multiplier=1.0, seed=20260900 + fold_number,
            )
            calibration_raw = _predict_ranker(model, calibration, fields)
            calibration_residual, _ = _finite_target(calibration)
            mapper = _fit_bps_map(calibration_raw, calibration_residual)
            score = "R3_portability_health__S1_uniform__custom_pairwise_control_256"
            result[score] = (
                test["base_expected_bps"].to_numpy(float) + mapper.predict(_predict_ranker(model, test, fields))
            ).astype(np.float32)
            fit_audits.append({
                "fold": str(fold), "score": score, "implementation": "custom_pairwise_logistic_control",
                "feature_arm": "R3_portability_health", "support_arm": "none_constant_false_control",
                "feature_count": len(fields), "supported_weight": None,
                "uses_causal_support_probability": False, "uses_support_outcome_label": False,
                "meta_train_rows_before_horizon_purge": len(unpurged_train), "meta_train_rows": len(train),
                "meta_calibration_rows": len(calibration), "test_rows": len(test),
                "universal_residual_training": True, "max_pairs_per_query": 256, **pair_audit,
            })
        elif args.include_s4:
            # This is deliberately one predeclared custom comparison: R3,
            # the selected S4 support-probability feature, and bounded 256
            # deterministic pairs per timestamp-side group.
            fields = [*feature_sets["R3_portability_health"], SUPPORT_PROBABILITY]
            model, pair_audit = _fit_custom_pairwise(
                train, fields, support_column=SUPPORT_COLUMN,
                supported_multiplier=float(args.s4_supported_weight), seed=20260900 + fold_number,
            )
            calibration_raw = _predict_ranker(model, calibration, fields)
            calibration_residual, _ = _finite_target(calibration)
            mapper = _fit_bps_map(calibration_raw, calibration_residual)
            score = f"R3_portability_health__S4_weight_{args.s4_supported_weight:g}_plus_prequential_probability__custom_pairwise_256"
            result[score] = (
                test["base_expected_bps"].to_numpy(float) + mapper.predict(_predict_ranker(model, test, fields))
            ).astype(np.float32)
            fit_audits.append({
                "fold": str(fold), "score": score, "implementation": "custom_pairwise_logistic",
                "feature_arm": "R3_portability_health", "support_arm": "S4",
                "feature_count": len(fields), "supported_weight": args.s4_supported_weight,
                "uses_causal_support_probability": True,
                "meta_train_rows_before_horizon_purge": len(unpurged_train), "meta_train_rows": len(train),
                "meta_calibration_rows": len(calibration), "test_rows": len(test),
                "universal_residual_training": True, "max_pairs_per_query": 256, **pair_audit,
            })
        prediction_parts.append(result)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    scores = [column for column in predictions if column.startswith("R") and "__" in column]
    predictions.to_parquet(output / "raw_oof_oos_predictions.parquet", index=False, compression="zstd")
    if support_audits:
        pd.concat(support_audits, ignore_index=True).to_parquet(
            output / "support_probability_audit.parquet", index=False, compression="zstd"
        )
        pd.concat(support_prediction_parts, ignore_index=True).to_parquet(
            output / "strict_prequential_support_predictions.parquet", index=False, compression="zstd"
        )
    _serialise_audit_for_parquet(fit_audits).to_parquet(
        output / "arm_fold_audit.parquet", index=False, compression="zstd"
    )
    pd.DataFrame([
        metric for score in scores for metric in _tail_metrics_with_fold_rows(predictions, score)
    ]).to_parquet(output / "ablation_metrics.parquet", index=False, compression="zstd")
    _write_json(output / "run_manifest.json", {
        "schema": SCHEMA, "status": "complete", "side": "long",
        "sidecar": str(args.sidecar),
        "feature_arms": sorted({spec[0] for spec in native_specs}),
        "residual_target": "direct net_bps - base_expected_bps; bps grade +/-50, +/-150",
        "support_label": None if args.custom_control_only else "frozen H12 MFE/MAE support: >3 ATR and >=1.5% favourable move",
        "support_label_usage": "not materialised: custom control uses constant false pair bookkeeping" if args.custom_control_only else "training weights/custom pair weighting only; never an inference feature, filter, or admission rule",
        "support_probability": None if args.custom_control_only else "strict 14-day prequential models with label_available_ts < score_block_start_ts",
        "s4_included_after_independent_stage": bool(args.include_s4),
        "s4_supported_weight": float(args.s4_supported_weight) if args.include_s4 else None,
        "custom_pairwise": "R3 custom control: no support label/probability; all support multipliers=1; deterministic max 256 pairs/query" if args.custom_control_only else ("R3 + S4 only; deterministic max 256 pairs per timestamp-side group" if args.include_s4 else "not run until S4"),
        "evaluation": "all candidates retained; pooled global top 1/3/5/10 plus monthly/fold diagnostics",
    })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar", type=Path,
        default=ROOT / "data_perp/artifacts/long_structural_tree_meta_sidecar_20260804_v4/tree_meta_candidate_sidecar.parquet",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-r1-comparison", action="store_true")
    parser.add_argument("--include-s4", action="store_true")
    parser.add_argument("--custom-control-only", action="store_true")
    parser.add_argument("--s4-supported-weight", type=float, choices=(1.5, 2.0, 3.0), default=2.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
