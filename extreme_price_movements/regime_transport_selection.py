"""Chronological transport audit for continuous regime/context candidates.

This is a *selection diagnostic*, not a regime generator and not a model
feature transform.  It deliberately refuses cluster IDs, posterior vectors and
state memberships: a field can only be admitted as a continuous context when
it has supported coverage, contributes to a causal thresholded opportunity
model in both within-era and cross-era tests, and is not mainly an era tag.

The expensive part is bounded by deterministic row caps and one cheap linear
classifier per feature/fold.  Crucially, economic MDA is performed only on the
chronologically held-out rows after a train-only model fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCHEMA = "regime_transport_context_audit_v1"
FORBIDDEN_MEMBERSHIP_TOKENS = (
    "membership", "posterior", "state_p_", "state_probability", "cluster",
    "gmm", "archetype_id", "regime_id", "state_id",
)
CONTROLLER_TOKENS = ("controller", "trust", "admission", "gate", "policy")


@dataclass(frozen=True)
class TransportAuditConfig:
    timestamp_column: str = "__ts__"
    era_column: str = "era"
    target_column: str = "execution_net_ev_12h"
    candidate_id_column: str = "candidate_id"
    threshold_bps: float = 0.0
    embargo_hours: float = 12.0
    within_era_train_fraction: float = 0.60
    top_fraction: float = 0.10
    min_rows_per_split: int = 300
    min_coverage: float = 0.90
    max_train_rows: int = 30_000
    max_eval_rows: int = 12_000
    random_state: int = 20260803
    invariant_min_transport_bps: float = 0.0
    smoothly_conditioned_min_transport_bps: float = -2.0
    max_era_proxy_importance: float = 0.010
    min_direction_consistency: float = 0.75


@dataclass(frozen=True)
class TransportAuditResult:
    feature_audit: pd.DataFrame
    split_mda: pd.DataFrame
    era_proxy: pd.DataFrame
    manifest: dict[str, Any]


def _forbidden(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in FORBIDDEN_MEMBERSHIP_TOKENS)


def _controller(name: str) -> bool:
    return any(token in str(name).lower() for token in CONTROLLER_TOKENS)


def _stable_sample(rows: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(rows) <= int(maximum):
        return rows
    # Candidate identity makes capping invariant to accidental input ordering.
    ordered = rows.sort_values("__audit_hash__", kind="stable")
    generator = np.random.default_rng(int(seed))
    position = np.sort(generator.choice(len(ordered), size=int(maximum), replace=False))
    return ordered.iloc[position]


def _prepare(
    frame: pd.DataFrame,
    features: Sequence[str],
    config: TransportAuditConfig,
    reference_features: Sequence[str] = (),
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    required = {config.timestamp_column, config.era_column, config.target_column, config.candidate_id_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"transport audit missing required columns: {sorted(missing)}")
    if len(set(features)) != len(features):
        raise ValueError("candidate feature names must be unique")
    unknown = set(features).difference(frame.columns)
    if unknown:
        raise KeyError(f"candidate features absent from frame: {sorted(unknown)[:8]}")
    out = frame.loc[:, [*required, *reference_features, *features]].copy()
    out[config.timestamp_column] = pd.to_datetime(out[config.timestamp_column], utc=True, errors="raise")
    out[config.target_column] = pd.to_numeric(out[config.target_column], errors="coerce")
    out = out.loc[out[config.target_column].notna()].copy()
    if out.empty:
        raise ValueError("no finite economic target rows")
    if out[config.candidate_id_column].duplicated().any():
        raise ValueError("candidate identities must be unique for deterministic held-out permutation")
    out = out.sort_values([config.timestamp_column, config.candidate_id_column], kind="stable").reset_index(drop=True)
    out["__audit_hash__"] = pd.util.hash_pandas_object(out[config.candidate_id_column].astype(str), index=False).to_numpy(np.uint64)
    out["__label__"] = out[config.target_column].mul(10_000.0).ge(float(config.threshold_bps)).astype(np.int8)
    if out["__label__"].nunique() < 2:
        raise ValueError("thresholded target is constant; change threshold_bps or population")
    coverage_rows: list[dict[str, Any]] = []
    admitted: list[str] = []
    for name in features:
        numeric = pd.to_numeric(out[name], errors="coerce")
        coverage = float(numeric.notna().mean())
        nonconstant = bool(numeric.nunique(dropna=True) > 1)
        forbidden = _forbidden(name)
        accepted = coverage >= float(config.min_coverage) and nonconstant and not forbidden
        coverage_rows.append({"feature": name, "coverage": coverage, "nonconstant": nonconstant, "membership_or_cluster_field": forbidden, "coverage_gate_pass": accepted})
        if accepted:
            out[name] = numeric
            admitted.append(name)
    return out, admitted, pd.DataFrame(coverage_rows)


def _ordered_eras(frame: pd.DataFrame, config: TransportAuditConfig) -> list[str]:
    summary = frame.groupby(config.era_column, observed=True)[config.timestamp_column].min().sort_values(kind="stable")
    return [str(value) for value in summary.index]


def _splits(frame: pd.DataFrame, config: TransportAuditConfig) -> list[dict[str, Any]]:
    """Build only chronological train -> later held-out evaluations."""

    ts, era = config.timestamp_column, config.era_column
    embargo = pd.Timedelta(hours=float(config.embargo_hours))
    ordered = _ordered_eras(frame, config)
    outputs: list[dict[str, Any]] = []
    for position, name in enumerate(ordered):
        local = frame.loc[frame[era].astype(str).eq(name)].sort_values(ts, kind="stable")
        cut_position = int(np.floor(len(local) * float(config.within_era_train_fraction)))
        if 0 < cut_position < len(local):
            start = local.iloc[cut_position][ts]
            train = local.loc[local[ts].lt(start - embargo)]
            evaluate = local.loc[local[ts].ge(start)]
            outputs.append({"scope": "within_era", "train_eras": name, "test_era": name, "train": train, "evaluate": evaluate})
        if position:
            start = local[ts].min()
            train = frame.loc[frame[ts].lt(start - embargo)]
            evaluate = local
            outputs.append({"scope": "cross_era", "train_eras": "|".join(ordered[:position]), "test_era": name, "train": train, "evaluate": evaluate})
    return [item for item in outputs if len(item["train"]) >= int(config.min_rows_per_split) and len(item["evaluate"]) >= int(config.min_rows_per_split) and item["train"]["__label__"].nunique() == 2]


def _pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.25, max_iter=300, solver="lbfgs")),
    ])


def _score_model(train: pd.DataFrame, evaluate: pd.DataFrame, features: Sequence[str], config: TransportAuditConfig, seed: int) -> tuple[np.ndarray, float]:
    used = _stable_sample(train, config.max_train_rows, seed)
    model = _pipeline()
    model.fit(used.loc[:, list(features)], used["__label__"])
    score = model.predict_proba(evaluate.loc[:, list(features)])[:, 1]
    coefficient = float(model.named_steps["model"].coef_[0, -1])
    return score, coefficient


def _economic_score(rows: pd.DataFrame, score: np.ndarray, config: TransportAuditConfig) -> tuple[float, float]:
    count = max(1, int(np.ceil(len(rows) * float(config.top_fraction))))
    order = np.lexsort((rows[config.candidate_id_column].astype(str).to_numpy(), -np.asarray(score, dtype=float)))
    selected = rows.iloc[order[:count]][config.target_column].to_numpy(float)
    rank_ic = pd.Series(score).rank(method="average").corr(rows[config.target_column].rank(method="average"))
    return float(np.mean(selected) * 10_000.0), float(rank_ic) if np.isfinite(rank_ic) else float("nan")


def _permute_column(rows: pd.DataFrame, feature: str, config: TransportAuditConfig) -> pd.DataFrame:
    # Deterministic held-out candidate-id rotation preserves the univariate
    # distribution but destroys the row-level relationship.  It never touches
    # train data or labels.
    out = rows.copy()
    order = np.argsort(rows[config.candidate_id_column].astype(str).to_numpy(), kind="stable")
    donor = np.roll(order, max(1, len(order) // 2))
    out.loc[:, feature] = rows.iloc[donor][feature].to_numpy()
    return out


def _mda(frame: pd.DataFrame, features: Sequence[str], reference_features: Sequence[str], config: TransportAuditConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = list(reference_features)
    for split_number, split in enumerate(_splits(frame, config)):
        train = _stable_sample(split["train"], config.max_train_rows, config.random_state + split_number)
        evaluate = _stable_sample(split["evaluate"], config.max_eval_rows, config.random_state + 10_000 + split_number)
        for feature_number, feature in enumerate(features):
            contract = list(dict.fromkeys([*baseline, feature]))
            score, coefficient = _score_model(train, evaluate, contract, config, config.random_state + split_number * 101 + feature_number)
            full_bps, full_ic = _economic_score(evaluate, score, config)
            permuted = _permute_column(evaluate, feature, config)
            permuted_score, _ = _score_model(train, permuted, contract, config, config.random_state + split_number * 101 + feature_number)
            permuted_bps, permuted_ic = _economic_score(evaluate, permuted_score, config)
            rows.append({
                "scope": split["scope"], "split_number": split_number, "train_eras": split["train_eras"], "test_era": split["test_era"], "feature": feature,
                "train_rows": len(train), "eval_rows": len(evaluate), "full_top_net_bps": full_bps, "permuted_top_net_bps": permuted_bps,
                "economic_mda_bps": full_bps - permuted_bps, "full_rank_ic": full_ic, "permuted_rank_ic": permuted_ic,
                "rank_ic_mda": full_ic - permuted_ic, "standardized_effect": coefficient,
            })
    return pd.DataFrame(rows)


def _era_proxy(frame: pd.DataFrame, features: Sequence[str], config: TransportAuditConfig) -> pd.DataFrame:
    """Return a bounded *diagnostic-only* era-separation proxy.

    The early segment of each era trains the classifier and its later segment
    evaluates it.  It has no outcome access and is never an inference input;
    its only purpose is to flag fields that primarily reconstruct era identity.
    This is disclosed as a representation-selection diagnostic rather than
    untouched temporal performance.
    """

    ts, era = config.timestamp_column, config.era_column
    train_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []
    for name in _ordered_eras(frame, config):
        local = frame.loc[frame[era].astype(str).eq(name)].sort_values(ts, kind="stable")
        cut = int(np.floor(len(local) * float(config.within_era_train_fraction)))
        if cut >= int(config.min_rows_per_split) and len(local) - cut >= int(config.min_rows_per_split):
            train_parts.append(local.iloc[:cut])
            eval_parts.append(local.iloc[cut:])
    if len(train_parts) < 2:
        return pd.DataFrame(columns=["feature", "era_proxy_importance", "era_proxy_accuracy", "era_proxy_baseline_accuracy", "diagnostic_only"])
    train = pd.concat(train_parts, ignore_index=True)
    evaluate = pd.concat(eval_parts, ignore_index=True)
    if train[era].nunique() < 2:
        return pd.DataFrame(columns=["feature", "era_proxy_importance", "era_proxy_accuracy", "era_proxy_baseline_accuracy", "diagnostic_only"])
    train = _stable_sample(train, config.max_train_rows, config.random_state + 77)
    evaluate = _stable_sample(evaluate, config.max_eval_rows, config.random_state + 78)
    output: list[dict[str, Any]] = []
    baseline = float(evaluate[era].value_counts(normalize=True).max())
    for position, feature in enumerate(features):
        model = _pipeline()
        model.fit(train.loc[:, [feature]], train[era].astype(str))
        accuracy = float(np.mean(model.predict(evaluate.loc[:, [feature]]) == evaluate[era].astype(str).to_numpy()))
        permuted = _permute_column(evaluate, feature, config)
        permuted_accuracy = float(np.mean(model.predict(permuted.loc[:, [feature]]) == evaluate[era].astype(str).to_numpy()))
        output.append({"feature": feature, "era_proxy_importance": accuracy - permuted_accuracy, "era_proxy_accuracy": accuracy, "era_proxy_baseline_accuracy": baseline, "diagnostic_only": True, "proxy_seed": config.random_state + position})
    return pd.DataFrame(output)


def _classification(audit: pd.DataFrame, config: TransportAuditConfig) -> pd.Series:
    result: list[str] = []
    for row in audit.itertuples(index=False):
        if not bool(row.coverage_gate_pass):
            result.append("REJECTED")
        elif _controller(row.feature):
            result.append("CONTROLLER_DIAGNOSTIC")
        elif row.within_era_mda_bps < float(config.smoothly_conditioned_min_transport_bps) or row.transport_score < float(config.smoothly_conditioned_min_transport_bps):
            result.append("ERA_SHORTCUT" if row.era_proxy_importance > float(config.max_era_proxy_importance) else "REJECTED")
        elif row.effect_direction_consistency >= float(config.min_direction_consistency) and row.within_era_mda_bps >= float(config.invariant_min_transport_bps) and row.transport_score >= float(config.invariant_min_transport_bps) and row.era_proxy_importance <= float(config.max_era_proxy_importance):
            result.append("INVARIANT_CORE")
        elif row.transport_score >= float(config.smoothly_conditioned_min_transport_bps):
            result.append("SMOOTHLY_CONDITIONED")
        else:
            result.append("REJECTED")
    return pd.Series(result, index=audit.index, dtype="object")


def audit_continuous_context_transport(
    frame: pd.DataFrame,
    *,
    candidate_features: Sequence[str],
    reference_features: Sequence[str] = (),
    config: TransportAuditConfig = TransportAuditConfig(),
) -> TransportAuditResult:
    """Audit continuous candidate context fields under a causal protocol.

    `reference_features` are the frozen non-regime score/context fields already
    in the prediction contract.  Every candidate is tested incrementally on
    top of that reference contract.  All model fits use a thresholded economic
    label (`target_bps >= threshold_bps`), whereas economic MDA continues to
    report the realised continuous net bps of the globally top-ranked rows.
    """

    if any(_forbidden(name) for name in reference_features):
        raise ValueError("reference features must not contain memberships, cluster IDs, or state fields")
    prepared, admitted, coverage = _prepare(frame, list(candidate_features), config, reference_features)
    missing_reference = set(reference_features).difference(prepared.columns)
    if missing_reference:
        raise KeyError(f"reference features absent from frame: {sorted(missing_reference)}")
    for name in reference_features:
        values = pd.to_numeric(prepared[name], errors="coerce")
        if values.notna().mean() < float(config.min_coverage) or values.nunique(dropna=True) <= 1:
            raise ValueError(f"reference feature fails supported-coverage/nonconstant contract: {name}")
        prepared[name] = values
    if not admitted:
        empty = coverage.assign(
            within_era_mda_bps=np.nan, cross_era_mda_bps=np.nan, effect_direction_consistency=np.nan,
            era_proxy_importance=np.nan, transport_score=np.nan, classification="REJECTED",
        )
        return TransportAuditResult(empty, pd.DataFrame(), pd.DataFrame(), {"schema": SCHEMA, "status": "NO_ADMITTED_CONTINUOUS_FIELDS", "config": asdict(config)})
    mda = _mda(prepared, admitted, reference_features, config)
    proxy = _era_proxy(prepared, admitted, config)
    aggregate = coverage.set_index("feature").loc[list(candidate_features)].reset_index()
    if not mda.empty:
        within = mda.loc[mda.scope.eq("within_era")].groupby("feature", observed=True)["economic_mda_bps"].median().rename("within_era_mda_bps")
        cross_values = mda.loc[mda.scope.eq("cross_era")].groupby("feature", observed=True)["economic_mda_bps"]
        cross = cross_values.median().rename("cross_era_mda_bps")
        cross_mad = cross_values.apply(lambda values: float(np.median(np.abs(values - np.median(values))))).rename("cross_era_mda_mad_bps")
        direction = mda.groupby("feature", observed=True)["standardized_effect"].apply(lambda values: float(max((values.gt(0).mean()), (values.lt(0).mean())))).rename("effect_direction_consistency")
        aggregate = aggregate.merge(within, on="feature", how="left").merge(cross, on="feature", how="left").merge(cross_mad, on="feature", how="left").merge(direction, on="feature", how="left")
    else:
        aggregate["within_era_mda_bps"] = np.nan
        aggregate["cross_era_mda_bps"] = np.nan
        aggregate["cross_era_mda_mad_bps"] = np.nan
        aggregate["effect_direction_consistency"] = np.nan
    aggregate = aggregate.merge(proxy, on="feature", how="left")
    aggregate["era_proxy_importance"] = aggregate["era_proxy_importance"].fillna(0.0)
    # The requested transport score regularises cross-era value by dispersion.
    # Within-era MDA remains an independent promotion gate in _classification,
    # preventing an apparently portable field that has no useful local effect.
    aggregate["transport_score"] = aggregate["cross_era_mda_bps"] - 0.5 * aggregate["cross_era_mda_mad_bps"]
    aggregate["classification"] = _classification(aggregate, config)
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETED_DIAGNOSTIC",
        "config": asdict(config),
        "label": {"kind": "thresholded_net_economic_label", "target_column": config.target_column, "threshold_bps": config.threshold_bps},
        "causality": {"economic_mda": "train strictly before held-out timestamp minus embargo; held-out candidate-id rotation only", "era_proxy": "diagnostic-only early-era train/later-era evaluation; no outcomes; not an untouched-OOS result"},
        "selection_contract": {"coverage_minimum": config.min_coverage, "forbidden_fields": list(FORBIDDEN_MEMBERSHIP_TOKENS), "cluster_memberships_excluded": True, "global_ranking": True},
        "counts": {"input_rows": int(len(frame)), "valid_target_rows": int(len(prepared)), "candidate_features": int(len(candidate_features)), "admitted_continuous_features": int(len(admitted)), "mda_rows": int(len(mda))},
    }
    return TransportAuditResult(aggregate.sort_values(["classification", "transport_score", "feature"], ascending=[True, False, True], kind="stable").reset_index(drop=True), mda, proxy, manifest)


__all__ = [
    "SCHEMA", "FORBIDDEN_MEMBERSHIP_TOKENS", "TransportAuditConfig", "TransportAuditResult",
    "audit_continuous_context_transport",
]
