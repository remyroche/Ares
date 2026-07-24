#!/usr/bin/env python3
"""Test whether active short-default mechanisms separate adverse lookalikes.

This is deliberately a low-capacity chronological discriminator.  It assesses
``P(adverse | broad mechanism active, observable pre-entry context)`` and does
not alter V11 ranks or select an inference policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from extreme_price_movements.challenger_credibility import consecutive_event_blocks


GROUP = ("short", "short_default_clean_path")
TARGET = "bad_residual_event_target"
MECHANISM = ("short_covering_score_market", "funding_confirmed_long_flush")
UNCERTAINTY_COLUMNS = (
    "ensemble_risk_std",
    "neighbor_shrunken_adverse_rate",
    "neighbor_weighted_ev_std",
    "neighbor_effective_count",
)
FEATURES = (
    "negative_breadth_pct",
    "extreme_negative_breadth_pct",
    "downside_breadth_intensity",
    "median_alt_minus_btc",
    "breadth_dispersion",
    "btc_oi_dominance_z_ratio",
    "btc_ex_eth_oi_dominance_z_ratio",
    "short_covering_score_market",
    "funding_confirmed_long_flush",
    "funding_deleveraging_divergence",
    "flush_recovery_state",
    "post_flush_leverage_rebuild",
    "fragmented_flush_recovery",
    "range_climax_reversal",
    "false_clean_short",
    "short_signal_recovery_conflict",
    "late_short_after_deleveraging",
    "mkt_regime_change__funding__delta_1h",
    "mkt_regime_change__funding__acceleration_1h",
    "mkt_regime_change__funding__cumulative_change_2d",
    "mkt_regime_change__oi_contraction__delta_1h",
    "mkt_regime_change__oi_contraction__acceleration_1h",
    "mkt_regime_change__oi_contraction__cumulative_change_2d",
    "mkt_regime_change__negative_breadth__delta_1h",
    "mkt_regime_change__eth_correlation__cumulative_change_2d",
    "mkt_regime_change__btc_alt_relative_strength__cumulative_change_2d",
    "resid_event_aegmm_expected_directional_ev_divergence",
    "resid_event_aegmm_expected_persistent_material_nontail",
    "resid_event_aegmm_expected_persistence_strength",
    "market_state_transition_entropy_5d",
    "market_state_persistence_5d",
    "short_default_damage_integral_5d",
    "short_default_adverse_duration_5d_norm",
    "recovery_failure_score_24h",
    "conditional_ensemble_disagreement",
    "conditional_neighbor_adverse_rate",
    "conditional_neighbor_ev_dispersion",
    "conditional_neighbor_reliability",
    "conditional_disagreement_x_ev_dispersion",
    "conditional_disagreement_x_ev_dispersion_x_reliability",
)


def _mechanism_score(frame: pd.DataFrame) -> np.ndarray:
    left = pd.to_numeric(frame[MECHANISM[0]], errors="coerce").to_numpy(np.float32)
    right = pd.to_numeric(frame[MECHANISM[1]], errors="coerce").to_numpy(np.float32)
    return np.maximum(left, 0.0) * np.maximum(right, 0.0)


def _event_blocks(frame: pd.DataFrame) -> pd.Series:
    """Use outcome-defined blocks only for train weights and OOF assessment."""

    daily = (
        frame.assign(day=frame["__ts__"].dt.floor("D"))
        .groupby("day", observed=True)[TARGET]
        .max()
        .reset_index()
    )
    labels = consecutive_event_blocks(daily["day"], daily[TARGET].astype(bool))
    mapping = dict(zip(daily["day"], labels, strict=True))
    return frame["__ts__"].dt.floor("D").map(mapping).fillna("normal")


def _event_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each adverse episode comparable influence without making it a feature."""

    target = frame[TARGET].to_numpy(np.int8)
    blocks = _event_blocks(frame).to_numpy(str)
    weights = np.ones(len(frame), dtype=np.float32)
    adverse = target.astype(bool)
    for block in np.unique(blocks[adverse]):
        mask = adverse & (blocks == block)
        weights[mask] = 1.0 / max(int(mask.sum()), 1)
    if adverse.any():
        weights[adverse] *= max(int((~adverse).sum()), 1) / max(int(adverse.sum()), 1)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def _available_features(frame: pd.DataFrame) -> list[str]:
    return [column for column in FEATURES if column in frame.columns and frame[column].notna().any()]


def _add_uncertainty_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the fixed observable D, N, V, R mechanism geometry.

    The later RobustScaler is fit on each prior chronological fold.  The raw
    interaction values are therefore not calibrated using the scored period.
    """

    output = frame.copy()
    disagreement = pd.to_numeric(output["ensemble_risk_std"], errors="coerce")
    adverse_rate = pd.to_numeric(
        output["neighbor_shrunken_adverse_rate"], errors="coerce"
    )
    ev_dispersion = pd.to_numeric(
        output["neighbor_weighted_ev_std"], errors="coerce")
    n_eff = pd.to_numeric(output["neighbor_effective_count"], errors="coerce")
    reliability = n_eff / (n_eff + 20.0)
    output["conditional_ensemble_disagreement"] = disagreement.astype(np.float32)
    output["conditional_neighbor_adverse_rate"] = adverse_rate.astype(np.float32)
    output["conditional_neighbor_ev_dispersion"] = ev_dispersion.astype(np.float32)
    output["conditional_neighbor_reliability"] = reliability.astype(np.float32)
    output["conditional_disagreement_x_ev_dispersion"] = (
        disagreement * ev_dispersion
    ).astype(np.float32)
    output["conditional_disagreement_x_ev_dispersion_x_reliability"] = (
        disagreement * ev_dispersion * reliability
    ).astype(np.float32)
    return output


def _attach_diagnostics(
    frame: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    stage: str,
) -> pd.DataFrame:
    keys = ["__ts__", "side_name", "archetype_policy_key"]
    available = set(diagnostics.columns)
    missing = sorted(set(UNCERTAINTY_COLUMNS).difference(available))
    if missing:
        raise ValueError(f"Diagnostic source is missing required uncertainty fields: {missing}")
    local = diagnostics.loc[
        diagnostics["stage"].eq(stage), keys + list(UNCERTAINTY_COLUMNS)
    ].drop_duplicates(keys, keep="last")
    result = frame.merge(local, on=keys, how="left", validate="many_to_one")
    return _add_uncertainty_interactions(result)


def _screen_features(train: pd.DataFrame, columns: list[str], *, limit: int = 8) -> list[str]:
    target = train[TARGET].to_numpy(np.int8)
    score: list[tuple[float, str]] = []
    for column in columns:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 40 or np.unique(target[finite]).size < 2:
            continue
        auc = roc_auc_score(target[finite], values[finite])
        score.append((abs(float(auc) - 0.5), column))
    return [column for _, column in sorted(score, reverse=True)[:limit]]


def _matrix(frame: pd.DataFrame, columns: list[str], medians: np.ndarray) -> np.ndarray:
    data = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32, copy=False)
    return np.where(np.isfinite(data), data, medians).astype(np.float32, copy=False)


def _fold_metrics(local: pd.DataFrame, risk: np.ndarray, cutoff: float) -> dict[str, float]:
    target = local[TARGET].to_numpy(np.int8)
    selected = risk >= cutoff
    prevalence = float(target.mean())
    precision = float(target[selected].mean()) if selected.any() else np.nan
    return {
        "rows": int(len(local)),
        "adverse_support": int(target.sum()),
        "adverse_prevalence": prevalence,
        "roc_auc": float(roc_auc_score(target, risk)) if np.unique(target).size > 1 else np.nan,
        "average_precision": float(average_precision_score(target, risk)) if target.sum() else np.nan,
        "top10_precision": precision,
        "top10_lift": precision / prevalence if prevalence > 0.0 and np.isfinite(precision) else np.nan,
        "top10_fpr": float(((risk >= cutoff) & (target == 0)).sum() / max((target == 0).sum(), 1)),
        "precision_gain_vs_mechanism": precision - prevalence if np.isfinite(precision) else np.nan,
    }


def _event_block_metrics(local: pd.DataFrame, risk: np.ndarray, cutoff: float) -> pd.DataFrame:
    result = local.loc[:, ["__ts__", TARGET]].copy()
    result["event_block"] = _event_blocks(local).to_numpy(str)
    result["selected"] = risk >= cutoff
    rows: list[dict[str, object]] = []
    for block, group in result.groupby("event_block", observed=True):
        if block == "normal" or int(group[TARGET].sum()) == 0:
            continue
        prevalence = float(group[TARGET].mean())
        selected = group["selected"].to_numpy(bool)
        precision = float(group.loc[selected, TARGET].mean()) if selected.any() else np.nan
        rows.append(
            {
                "event_block": str(block),
                "rows": int(len(group)),
                "adverse_support": int(group[TARGET].sum()),
                "prevalence": prevalence,
                "top10_precision": precision,
                "top10_lift": precision / prevalence if np.isfinite(precision) and prevalence > 0 else np.nan,
                "top10_fpr": float(((selected) & group[TARGET].eq(0).to_numpy()).sum() / max(int(group[TARGET].eq(0).sum()), 1)),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    if args.train_context is not None or args.oos_context is not None:
        if args.train_context is None or args.oos_context is None:
            raise ValueError("--train-context and --oos-context must be supplied together")
        context_manifest = args.context_manifest or args.train_context.parent / "manifest.json"
        if not context_manifest.exists():
            raise FileNotFoundError(
                f"Context manifest is required to prove causal parity: {context_manifest}"
            )
        context_contract = json.loads(context_manifest.read_text())
        if context_contract.get("status") != "parity_passed":
            manifest = {
                "schema": "short_default_conditional_mechanism_discrimination_v2",
                "scope": {"side": GROUP[0], "archetype": GROUP[1]},
                "promotion_status": "blocked_context_parity_failed",
                "context_manifest": str(context_manifest),
                "context_status": context_contract.get("status"),
                "required_action": (
                    "Reconstruct the train-OOF market context with proven OOS parity. "
                    "Do not train the conditional discriminator on non-parity context."
                ),
            }
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            return manifest
    # Context artifacts intentionally contain keys plus the two backfilled
    # observable fields. Merge them onto the immutable V11 ledgers instead of
    # treating them as replacement prediction tables.
    train = pd.read_parquet(args.v11_dir / "train_oof_predictions.parquet")
    evaluated = pd.read_parquet(args.v11_dir / "oos_predictions.parquet")
    if args.train_context is not None:
        keys = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
        def attach_context(ledger: pd.DataFrame, path: Path) -> pd.DataFrame:
            context = pd.read_parquet(path)
            missing = sorted(set([*keys, *MECHANISM]).difference(context.columns))
            if missing:
                raise ValueError(f"Context artifact is missing required columns: {missing}")
            return ledger.drop(
                columns=[column for column in MECHANISM if column in ledger]
            ).merge(
                context.loc[:, [*keys, *MECHANISM]],
                on=keys,
                how="left",
                validate="one_to_one",
                copy=False,
            )
        train = attach_context(train, args.train_context)
        evaluated = attach_context(evaluated, args.oos_context)
    for frame in (train, evaluated):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    train = train.loc[train["side_name"].eq(GROUP[0]) & train["archetype_policy_key"].eq(GROUP[1])].copy()
    evaluated = evaluated.loc[evaluated["side_name"].eq(GROUP[0]) & evaluated["archetype_policy_key"].eq(GROUP[1])].copy()
    if args.diagnostics is None:
        manifest = {
            "schema": "short_default_conditional_mechanism_discrimination_v2",
            "scope": {"side": GROUP[0], "archetype": GROUP[1]},
            "promotion_status": "blocked_missing_uncertainty_diagnostics",
            "required_uncertainty_columns": list(UNCERTAINTY_COLUMNS),
            "required_action": (
                "Supply the frozen V11 state-distinguishability diagnostics so the "
                "D x V and D x V x R lookalike interactions are evaluated causally."
            ),
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    diagnostics = pd.read_parquet(args.diagnostics / "state_distinguishability_predictions.parquet")
    diagnostics["__ts__"] = pd.to_datetime(diagnostics["__ts__"], utc=True)
    try:
        train = _attach_diagnostics(train, diagnostics, stage="train_oof")
        evaluated = _attach_diagnostics(evaluated, diagnostics, stage="eval_oos")
    except ValueError as exc:
        manifest = {
            "schema": "short_default_conditional_mechanism_discrimination_v2",
            "scope": {"side": GROUP[0], "archetype": GROUP[1]},
            "promotion_status": "blocked_missing_uncertainty_diagnostics",
            "error": str(exc),
            "required_uncertainty_columns": list(UNCERTAINTY_COLUMNS),
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    columns = _available_features(train)
    missing_mechanism = [column for column in MECHANISM if column not in train]
    missing_interactions = [
        column for column in (
            "conditional_disagreement_x_ev_dispersion",
            "conditional_disagreement_x_ev_dispersion_x_reliability",
        ) if column not in columns
    ]
    if TARGET not in train or missing_mechanism or missing_interactions or not columns:
        manifest = {
            "schema": "short_default_conditional_mechanism_discrimination_v2",
            "scope": {"side": GROUP[0], "archetype": GROUP[1]},
            "promotion_status": "blocked_missing_train_feature_parity",
            "missing_train_columns": sorted(set(
                missing_mechanism
                + missing_interactions
                + ([] if TARGET in train else [TARGET])
            )),
            "available_candidate_features": columns,
            "required_action": (
                "Materialize the negative-residual market context onto the V11 train-OOF "
                "ledger from the same point-in-time feature store, then rerun. OOS-only "
                "features are not permitted as a substitute."
            ),
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest
    train["mechanism_strength"] = _mechanism_score(train)
    evaluated["mechanism_strength"] = _mechanism_score(evaluated)
    # All thresholds and feature ranking are fitted only from each prior period.
    frequency = str(args.fold_frequency).upper()
    periods = sorted(
        pd.Period(value, freq=frequency)
        for value in train["__ts__"].dt.tz_localize(None).dt.to_period(frequency).unique()
    )
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for period in periods[1:]:
        start = pd.Timestamp(period.start_time, tz="UTC")
        end = pd.Timestamp(period.end_time, tz="UTC") + pd.Timedelta(days=1)
        prior = train.loc[train["__ts__"].lt(start)].copy()
        score = train.loc[train["__ts__"].ge(start) & train["__ts__"].lt(end)].copy()
        if prior.empty or score.empty:
            continue
        active_cutoff = float(np.nanquantile(prior["mechanism_strength"], args.mechanism_quantile))
        fit = prior.loc[prior["mechanism_strength"].ge(active_cutoff)].copy()
        check = score.loc[score["mechanism_strength"].ge(active_cutoff)].copy()
        if len(fit) < args.min_train_rows or fit[TARGET].sum() < args.min_adverse_rows or len(check) < 20:
            continue
        selected = _screen_features(fit, columns, limit=args.max_features)
        if not selected:
            continue
        medians = fit.loc[:, selected].apply(pd.to_numeric, errors="coerce").median().fillna(0.0).to_numpy(np.float32)
        scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(_matrix(fit, selected, medians))
        weights = _event_balanced_weights(fit)
        model = LogisticRegression(
            penalty="l1", l1_ratio=1.0, solver="saga", C=args.l1_c,
            class_weight="balanced", max_iter=2_000, random_state=args.seed,
        ).fit(
            scaler.transform(_matrix(fit, selected, medians)),
            fit[TARGET].to_numpy(np.int8),
            sample_weight=weights,
        )
        train_risk = model.predict_proba(scaler.transform(_matrix(fit, selected, medians)))[:, 1]
        risk = model.predict_proba(scaler.transform(_matrix(check, selected, medians)))[:, 1]
        cutoff = float(np.quantile(train_risk, 0.90))
        metrics = _fold_metrics(check, risk, cutoff)
        baseline_cutoff = float(np.quantile(fit["mechanism_strength"], 0.90))
        baseline = _fold_metrics(
            check,
            check["mechanism_strength"].to_numpy(np.float32),
            baseline_cutoff,
        )
        block_metrics = _event_block_metrics(check, risk, cutoff)
        positive_block_count = int(block_metrics["top10_lift"].gt(1.0).sum()) if not block_metrics.empty else 0
        rows.append({
            "fold": str(period),
            "active_cutoff": active_cutoff,
            "risk_cutoff": cutoff,
            "baseline_mechanism_cutoff": baseline_cutoff,
            "selected_features": json.dumps(selected),
            "event_balanced_weighting": True,
            "event_block_count": int(len(block_metrics)),
            "positive_event_block_count": positive_block_count,
            "delta_top10_lift_vs_mechanism": metrics["top10_lift"] - baseline["top10_lift"],
            "delta_top10_fpr_vs_mechanism": metrics["top10_fpr"] - baseline["top10_fpr"],
            **metrics,
        })
        if not block_metrics.empty:
            block_metrics.insert(0, "fold", str(period))
            block_metrics.to_csv(args.output / f"event_block_metrics_{period}.csv", index=False)
        part = check.loc[:, ["__ts__", "__symbol__", "side_name", "archetype_policy_key", TARGET]].copy()
        part["fold"] = str(period)
        part["mechanism_strength"] = check["mechanism_strength"].to_numpy(np.float32)
        part["mechanism_probability"] = risk.astype(np.float32)
        part["mechanism_support"] = len(fit)
        part["mechanism_neighbor_reliability"] = float(len(fit) / (len(fit) + 150.0))
        part["mechanism_historical_adverse_rate"] = float(fit[TARGET].mean())
        part["mechanism_incremental_lift"] = metrics["top10_lift"]
        predictions.append(part)
    # The final OOS score is fitted only on all train-OOF rows; it is a single untouched check.
    final_cutoff = float(np.nanquantile(train["mechanism_strength"], args.mechanism_quantile))
    fit = train.loc[train["mechanism_strength"].ge(final_cutoff)].copy()
    final_features = _screen_features(fit, columns, limit=args.max_features)
    final_metrics: dict[str, object] = {"final_oos_available": False}
    if len(fit) >= args.min_train_rows and fit[TARGET].sum() >= args.min_adverse_rows and final_features:
        medians = fit.loc[:, final_features].apply(pd.to_numeric, errors="coerce").median().fillna(0.0).to_numpy(np.float32)
        scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(_matrix(fit, final_features, medians))
        model = LogisticRegression(
            penalty="l1", l1_ratio=1.0, solver="saga", C=args.l1_c,
            class_weight="balanced", max_iter=2_000, random_state=args.seed,
        ).fit(
            scaler.transform(_matrix(fit, final_features, medians)),
            fit[TARGET].to_numpy(np.int8),
            sample_weight=_event_balanced_weights(fit),
        )
        check = evaluated.loc[evaluated["mechanism_strength"].ge(final_cutoff)].copy()
        if len(check) and check[TARGET].nunique() > 1:
            train_risk = model.predict_proba(scaler.transform(_matrix(fit, final_features, medians)))[:, 1]
            risk = model.predict_proba(scaler.transform(_matrix(check, final_features, medians)))[:, 1]
            metrics = _fold_metrics(check, risk, float(np.quantile(train_risk, 0.90)))
            final_metrics = {"final_oos_available": True, "selected_features": final_features, **metrics}
            part = check.loc[:, ["__ts__", "__symbol__", "side_name", "archetype_policy_key", TARGET]].copy()
            part["fold"] = "final_oos"
            part["mechanism_strength"] = check["mechanism_strength"].to_numpy(np.float32)
            part["mechanism_probability"] = risk.astype(np.float32)
            part["mechanism_support"] = len(fit)
            part["mechanism_neighbor_reliability"] = float(len(fit) / (len(fit) + 150.0))
            part["mechanism_historical_adverse_rate"] = float(fit[TARGET].mean())
            part["mechanism_incremental_lift"] = metrics["top10_lift"]
            predictions.append(part)
    report = pd.DataFrame(rows)
    report.to_csv(args.output / "chronological_conditional_metrics.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True, copy=False).to_parquet(args.output / "conditional_mechanism_predictions.parquet", index=False, compression="zstd")
    promoted = bool(
        len(report) >= 3
        and report["top10_lift"].ge(1.5).all()
        and report["top10_fpr"].le(0.15).all()
        and report["precision_gain_vs_mechanism"].gt(0.0).all()
        and report["delta_top10_lift_vs_mechanism"].gt(0.0).all()
        and report["delta_top10_fpr_vs_mechanism"].lt(0.0).all()
        and report["positive_event_block_count"].ge(3).all()
    )
    manifest = {
        "schema": "short_default_conditional_mechanism_discrimination_v2",
        "mechanism": "short_covering_score_market_positive_x_funding_confirmed_long_flush_positive",
        "scope": {"side": GROUP[0], "archetype": GROUP[1]},
        "feature_contract": "observable V11 pre-entry features only; outcome columns excluded",
        "conditional_geometry": {
            "D": "ensemble_risk_std",
            "N": "neighbor_shrunken_adverse_rate",
            "V": "neighbor_weighted_ev_std",
            "R": "neighbor_effective_count / (neighbor_effective_count + 20)",
            "required_interactions": [
                "conditional_disagreement_x_ev_dispersion",
                "conditional_disagreement_x_ev_dispersion_x_reliability",
            ],
        },
        "fold_count": len(report),
        "fold_frequency": frequency,
        "promotion_status": "validated_production_candidate" if promoted else "diagnostic_only",
        "promotion_rules": (
            "all chronological folds: lift>=1.5, FPR<=0.15, positive precision gain, "
            "incremental lift vs broad mechanism >0, incremental FPR <0, and at least "
            "three positively separated adverse blocks; at least 3 folds"
        ),
        "final_oos": final_metrics,
        "leakage_contract": (
            "Mechanism threshold, feature screen, event-balanced weights, scaler, and sparse classifier are fit "
            "only on prior chronological rows. Event blocks are outcome-derived training/evaluation units, never "
            "inference features. Final OOS is not used for feature or threshold choice."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-context", type=Path, help="Parity-materialized V11 train-OOF ledger with residual market context.")
    parser.add_argument("--oos-context", type=Path, help="Parity-materialized V11 OOS ledger with residual market context.")
    parser.add_argument("--context-manifest", type=Path, help="Causal context parity manifest; must report status=parity_passed.")
    parser.add_argument("--diagnostics", type=Path, help="Frozen V11 state-distinguishability diagnostic directory.")
    parser.add_argument("--mechanism-quantile", type=float, default=0.80)
    parser.add_argument(
        "--fold-frequency",
        default="M",
        choices=["M", "Q"],
        help="Chronological OOF diagnostic cadence; final OOS remains untouched.",
    )
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--l1-c", type=float, default=0.08)
    parser.add_argument("--min-train-rows", type=int, default=150)
    parser.add_argument("--min-adverse-rows", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
