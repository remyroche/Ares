"""Fold-local discovery of regime and transition interaction candidates.

The module ranks two deliberately separate questions:

* which observable predictor interacts with a *regime probability*; and
* which observable predictor interacts with a *transition probability*.

It is research-only.  All discovery rows must belong to one already-purged
training fold.  Neither target/outcome-like columns nor post-entry fields may
be predictor inputs, and a transition probability is never re-labelled as a
regime probability (or conversely).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


INTERACTION_DISCOVERY_SCHEMA = "fold_local_regime_interaction_discovery_v1"
REGIME_PROBABILITY_PREFIX = "regime_prob__"
TRANSITION_PROBABILITY_PREFIX = "transition_prob__"
# Regime/transition heads expose these canonical names.  Interaction discovery
# uses the shorter names above as an input namespace, so the one-to-one adapter
# is explicit and can never make a transition probability look like a regime
# probability (or vice versa).
REGIME_STATE_PROBABILITY_PREFIX = "regime_state_p__"
TRANSITION_STATE_PROBABILITY_PREFIX = "transition_state_p__"
FORBIDDEN_PREDICTOR_TOKENS: tuple[str, ...] = (
    "target",
    "label",
    "outcome",
    "post_entry",
    "postentry",
    "future",
    "realized_pnl",
    "realised_pnl",
    "realized_ev",
    "realised_ev",
    "realized_outcome",
    "realised_outcome",
    "mfe",
    "mae",
    "pnl",
    "net_ev",
    "gross_ev",
    "ev_after",
    "exit",
    "timeout",
    "time_to",
    "barrier",
)


class TreeInteractionUnavailableError(RuntimeError):
    """Raised when an exact tree-SHAP interaction calculation is unavailable."""


@dataclass(frozen=True)
class InteractionDiscoveryConfig:
    """Bounded controls for one fold-local discovery run."""

    fold_id: str
    max_rows: int = 12_000
    stratify_bins: int = 8
    stability_subsamples: int = 3
    permutation_repeats: int = 3
    min_probability_support: float = 24.0
    top_n: int = 40
    random_state: int = 20260730
    require_model_training_identity: bool = True


def _is_forbidden(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in FORBIDDEN_PREDICTOR_TOKENS)


def _require_index_alignment(
    predictors: pd.DataFrame,
    target: pd.Series,
    probabilities: pd.DataFrame,
    *,
    name: str,
) -> None:
    if not predictors.index.equals(target.index):
        raise ValueError("predictor and target identities must match exactly")
    if not predictors.index.equals(probabilities.index):
        raise ValueError(f"predictor and {name} probability identities must match exactly")
    if not predictors.index.is_unique:
        raise ValueError("fold-local discovery rows require unique identities")


def _validate_probability_namespace(
    probabilities: pd.DataFrame,
    *,
    prefix: str,
    other_prefix: str,
    name: str,
) -> list[str]:
    columns = [str(column) for column in probabilities.columns]
    wrong = [column for column in columns if not column.startswith(prefix)]
    if wrong:
        raise ValueError(
            f"{name} probabilities must use the {prefix!r} namespace: {wrong[:6]}"
        )
    cross = [column for column in columns if column.startswith(other_prefix)]
    if cross:
        raise ValueError(f"{name} probabilities contain the other namespace: {cross[:6]}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} probability columns must be unique")
    values = probabilities.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} probabilities must be finite values in [0, 1]")
    return columns


def adapt_state_probability_namespace(
    probabilities: pd.DataFrame,
    *,
    kind: str,
) -> tuple[pd.DataFrame, str]:
    """Return interaction-safe aliases for one homogeneous OOF probability frame.

    The adapter is intentionally a rename, not a transform.  It accepts either
    the canonical head output (``*_state_p__``) or the established interaction
    alias (``*_prob__``), rejects a mixed/foreign namespace, and preserves the
    row index exactly.  Call it only after OOF probabilities were generated;
    it cannot be used to combine head outputs.
    """

    mapping = {
        "regime": (REGIME_STATE_PROBABILITY_PREFIX, REGIME_PROBABILITY_PREFIX, TRANSITION_STATE_PROBABILITY_PREFIX, TRANSITION_PROBABILITY_PREFIX),
        "transition": (TRANSITION_STATE_PROBABILITY_PREFIX, TRANSITION_PROBABILITY_PREFIX, REGIME_STATE_PROBABILITY_PREFIX, REGIME_PROBABILITY_PREFIX),
    }
    if kind not in mapping:
        raise ValueError("kind must be 'regime' or 'transition'")
    canonical, alias, foreign_canonical, foreign_alias = mapping[kind]
    columns = [str(column) for column in probabilities.columns]
    if not columns:
        raise ValueError(f"{kind} probabilities cannot be empty")
    foreign = [name for name in columns if name.startswith((foreign_canonical, foreign_alias))]
    if foreign:
        raise ValueError(f"{kind} probabilities contain the other head namespace: {foreign[:6]}")
    canonical_columns = [name for name in columns if name.startswith(canonical)]
    alias_columns = [name for name in columns if name.startswith(alias)]
    if len(canonical_columns) == len(columns):
        renamed = probabilities.copy()
        renamed.columns = [f"{alias}{name[len(canonical):]}" for name in canonical_columns]
        return renamed, "canonical_state_output_renamed"
    if len(alias_columns) == len(columns):
        return probabilities.copy(), "interaction_alias_passthrough"
    raise ValueError(
        f"{kind} probabilities must use exactly one namespace: {canonical!r} or {alias!r}"
    )


def _validate_predictors(predictors: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in predictors.columns]
    forbidden = [column for column in columns if _is_forbidden(column)]
    if forbidden:
        raise ValueError(f"predictors contain forbidden outcome/post-entry fields: {forbidden[:8]}")
    namespace = [
        column
        for column in columns
        if column.startswith((
            REGIME_PROBABILITY_PREFIX,
            TRANSITION_PROBABILITY_PREFIX,
            REGIME_STATE_PROBABILITY_PREFIX,
            TRANSITION_STATE_PROBABILITY_PREFIX,
        ))
    ]
    if namespace:
        raise ValueError(
            "predictors must not contain regime/transition probability namespaces; "
            f"pass them separately: {namespace[:8]}"
        )
    non_numeric = [
        column for column in columns if not pd.api.types.is_numeric_dtype(predictors[column])
    ]
    if non_numeric:
        raise TypeError(f"predictors must be numeric: {non_numeric[:8]}")
    if not columns:
        raise ValueError("at least one observable predictor is required")
    return columns


def _stratify_labels(target: np.ndarray, bins: int) -> np.ndarray:
    finite = np.isfinite(target)
    if not finite.all():
        raise ValueError("fold-local target must be finite")
    unique = np.unique(target)
    if len(unique) <= max(2, int(bins)):
        return pd.factorize(target, sort=True)[0]
    try:
        return pd.qcut(target, q=min(int(bins), len(unique)), labels=False, duplicates="drop").to_numpy(dtype=int)
    except (TypeError, ValueError):
        return pd.factorize(target, sort=True)[0]


def deterministic_stratified_subsample_positions(
    target: Sequence[float] | pd.Series | np.ndarray,
    *,
    max_rows: int,
    bins: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Deterministically retain a representative, target-stratified subset."""

    values = np.asarray(target, dtype=float)
    n = len(values)
    if max_rows <= 0 or n <= int(max_rows):
        return np.arange(n, dtype=np.int64)
    labels = _stratify_labels(values, bins)
    rng = np.random.default_rng(int(seed))
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    ideal = np.asarray([len(group) * int(max_rows) / n for group in groups], dtype=float)
    take = np.floor(ideal).astype(int)
    take[(take == 0) & (np.asarray([len(group) for group in groups]) > 0)] = 1
    while take.sum() > int(max_rows):
        eligible = np.flatnonzero(take > 1)
        if not len(eligible):
            break
        take[eligible[np.argmax(take[eligible] - ideal[eligible])]] -= 1
    while take.sum() < int(max_rows):
        eligible = np.flatnonzero(take < np.asarray([len(group) for group in groups]))
        if not len(eligible):
            break
        take[eligible[np.argmax(ideal[eligible] - take[eligible])]] += 1
    selected = [
        np.sort(rng.choice(group, size=min(int(count), len(group)), replace=False))
        for group, count in zip(groups, take)
        if count > 0
    ]
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def _as_interaction_tensor(raw: Any, *, rows: int, columns: int) -> np.ndarray:
    """Normalize exact tree-SHAP interaction output to [row, feature, feature]."""

    values = raw[-1] if isinstance(raw, list) else raw
    array = np.asarray(values, dtype=float)
    if array.ndim == 4:  # class, row, feature, feature
        array = array[-1]
    if array.shape != (rows, columns, columns):
        raise TreeInteractionUnavailableError(
            "tree interaction API returned an unexpected shape; expected "
            f"({rows}, {columns}, {columns}), received {array.shape}"
        )
    if not np.isfinite(array).all():
        raise TreeInteractionUnavailableError("tree interaction API returned non-finite values")
    return array


def exact_tree_shap_interactions(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Return exact tree interaction values or fail clearly.

    A model may expose an exact ``shap_interaction_values`` method.  Otherwise
    we invoke TreeSHAP's exact tree explainer.  There is intentionally no
    surrogate or approximate interaction fallback: a candidate must not be
    advertised as tree-SHAP-derived when it is not.
    """

    try:
        if hasattr(model, "shap_interaction_values"):
            return _as_interaction_tensor(
                model.shap_interaction_values(x), rows=len(x), columns=x.shape[1]
            )
        import shap  # type: ignore[import-not-found]

        explainer = shap.TreeExplainer(model)
        return _as_interaction_tensor(
            explainer.shap_interaction_values(x), rows=len(x), columns=x.shape[1]
        )
    except TreeInteractionUnavailableError:
        raise
    except Exception as exc:
        raise TreeInteractionUnavailableError(
            "exact tree-SHAP interaction discovery is unavailable for this model; "
            "supply a tree model supported by shap.TreeExplainer or an estimator "
            "with shap_interaction_values(X)."
        ) from exc


def _scalar_prediction(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        value = np.asarray(model.predict_proba(x), dtype=float)
        if value.ndim == 2:
            value = value[:, -1]
    elif hasattr(model, "predict"):
        value = np.asarray(model.predict(x), dtype=float)
        if value.ndim == 2:
            value = value[:, -1]
    else:
        raise TypeError("tree interaction model must expose predict or predict_proba")
    if value.shape != (len(x),) or not np.isfinite(value).all():
        raise ValueError("model prediction must be one finite scalar per discovery row")
    return value


def _weighted_squared_loss(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average((target - prediction) ** 2, weights=np.maximum(weight, 1e-12)))


def _conditional_permutation_importance(
    model: Any,
    x: pd.DataFrame,
    target: np.ndarray,
    probability: np.ndarray,
    feature: str,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    support = float(np.sum(probability))
    if support <= 0.0:
        return np.nan, np.nan
    baseline = _weighted_squared_loss(target, _scalar_prediction(model, x), probability)
    rng = np.random.default_rng(int(seed))
    effects: list[float] = []
    for _ in range(max(1, int(repeats))):
        permuted = x.copy()
        permuted[feature] = rng.permutation(permuted[feature].to_numpy())
        loss = _weighted_squared_loss(target, _scalar_prediction(model, permuted), probability)
        effects.append(loss - baseline)
    return float(np.mean(effects)), float(np.std(effects, ddof=0))


def _candidate_rows(
    interaction: np.ndarray,
    *,
    model_columns: list[str],
    predictor_columns: list[str],
    probability_columns: list[str],
    namespace: str,
    x: pd.DataFrame,
    target: np.ndarray,
    model: Any,
    config: InteractionDiscoveryConfig,
    repeat: int,
) -> list[dict[str, Any]]:
    feature_pos = {name: position for position, name in enumerate(model_columns)}
    records: list[dict[str, Any]] = []
    for probability_name in probability_columns:
        probability = x[probability_name].to_numpy(dtype=float)
        support = float(np.sum(probability))
        if support < float(config.min_probability_support):
            continue
        p_pos = feature_pos[probability_name]
        for predictor_name in predictor_columns:
            f_pos = feature_pos[predictor_name]
            interaction_score = float(np.mean(np.abs(interaction[:, p_pos, f_pos])))
            perm_mean, perm_std = _conditional_permutation_importance(
                model,
                x,
                target,
                probability,
                predictor_name,
                repeats=config.permutation_repeats,
                seed=config.random_state + repeat * 10_007 + p_pos * 101 + f_pos,
            )
            records.append(
                {
                    "namespace": namespace,
                    "probability_column": probability_name,
                    "predictor": predictor_name,
                    "subsample": int(repeat),
                    "sample_rows": int(len(x)),
                    "probability_support": support,
                    "shap_interaction": interaction_score,
                    "conditional_permutation_importance": perm_mean,
                    "conditional_permutation_std": perm_std,
                }
            )
    return records


def _aggregate_candidates(rows: pd.DataFrame, config: InteractionDiscoveryConfig) -> pd.DataFrame:
    columns = [
        "namespace",
        "probability_column",
        "predictor",
        "support_rows_mean",
        "sample_rows_mean",
        "shap_interaction_mean",
        "shap_interaction_std",
        "shap_stability",
        "conditional_permutation_mean",
        "conditional_permutation_std",
        "permutation_stability",
        "combined_score",
        "subsamples",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    grouped = rows.groupby(["namespace", "probability_column", "predictor"], observed=True, sort=True)
    result = grouped.agg(
        support_rows_mean=("probability_support", "mean"),
        sample_rows_mean=("sample_rows", "mean"),
        shap_interaction_mean=("shap_interaction", "mean"),
        shap_interaction_std=("shap_interaction", "std"),
        shap_stability=("shap_interaction", lambda value: float(np.mean(np.asarray(value) > 0.0))),
        conditional_permutation_mean=("conditional_permutation_importance", "mean"),
        conditional_permutation_std=("conditional_permutation_importance", "std"),
        permutation_stability=("conditional_permutation_importance", lambda value: float(np.mean(np.asarray(value) > 0.0))),
        subsamples=("subsample", "nunique"),
    ).reset_index()
    for column in ("shap_interaction_std", "conditional_permutation_std"):
        result[column] = result[column].fillna(0.0)
    # Percentile ranks keep TreeSHAP magnitude and conditional loss movement on
    # comparable scales while preserving their independent diagnostics.
    shap_rank = result["shap_interaction_mean"].rank(pct=True, method="average")
    perm_rank = result["conditional_permutation_mean"].rank(pct=True, method="average")
    result["combined_score"] = 0.5 * shap_rank + 0.5 * perm_rank
    return result.sort_values(
        ["combined_score", "shap_interaction_mean", "conditional_permutation_mean"],
        ascending=False,
        kind="stable",
    ).head(int(config.top_n)).reindex(columns=columns)


def discover_fold_local_regime_interactions(
    predictors: pd.DataFrame,
    target: Sequence[float] | pd.Series | np.ndarray,
    regime_probabilities: pd.DataFrame,
    transition_probabilities: pd.DataFrame,
    *,
    model: Any,
    config: InteractionDiscoveryConfig,
    model_training_row_ids: Sequence[Any] | pd.Index | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Discover ranked regime×feature and transition×feature candidates.

    ``predictors`` is exactly the training fold after temporal purging.  If
    ``require_model_training_identity`` is true, callers must attest that the
    supplied tree model was fit on exactly those row identities; this prevents
    a prefit model from silently incorporating an evaluation fold.
    """

    if not str(config.fold_id).strip():
        raise ValueError("fold_id is required: discovery must be fold-local")
    y = pd.Series(np.asarray(target, dtype=float), index=predictors.index)
    if len(y) != len(predictors):
        raise ValueError("target length must equal fold-local predictor rows")
    predictor_columns = _validate_predictors(predictors)
    regime_probabilities, regime_namespace_source = adapt_state_probability_namespace(
        regime_probabilities, kind="regime"
    )
    transition_probabilities, transition_namespace_source = adapt_state_probability_namespace(
        transition_probabilities, kind="transition"
    )
    _require_index_alignment(predictors, y, regime_probabilities, name="regime")
    _require_index_alignment(predictors, y, transition_probabilities, name="transition")
    regime_columns = _validate_probability_namespace(
        regime_probabilities,
        prefix=REGIME_PROBABILITY_PREFIX,
        other_prefix=TRANSITION_PROBABILITY_PREFIX,
        name="regime",
    )
    transition_columns = _validate_probability_namespace(
        transition_probabilities,
        prefix=TRANSITION_PROBABILITY_PREFIX,
        other_prefix=REGIME_PROBABILITY_PREFIX,
        name="transition",
    )
    overlap = set(regime_columns).intersection(transition_columns)
    if overlap:
        raise ValueError(f"regime and transition namespaces overlap: {sorted(overlap)[:6]}")
    if config.require_model_training_identity:
        if model_training_row_ids is None:
            raise ValueError("model_training_row_ids is required to attest fold-local train-only fitting")
        provided = pd.Index(model_training_row_ids)
        if not provided.equals(predictors.index):
            raise ValueError("model_training_row_ids must exactly equal discovery-fold row identities")
    model_columns = [*predictor_columns, *regime_columns, *transition_columns]
    x_full = pd.concat(
        [predictors.loc[:, predictor_columns], regime_probabilities.loc[:, regime_columns], transition_probabilities.loc[:, transition_columns]],
        axis=1,
    )
    if not np.isfinite(x_full.to_numpy(dtype=float)).all():
        raise ValueError("tree interaction inputs must be finite; impute inside the fold before discovery")
    positions_by_repeat = [
        deterministic_stratified_subsample_positions(
            y.to_numpy(dtype=float),
            max_rows=config.max_rows,
            bins=config.stratify_bins,
            seed=config.random_state + repeat * 7_919,
        )
        for repeat in range(max(1, int(config.stability_subsamples)))
    ]
    records: list[dict[str, Any]] = []
    for repeat, positions in enumerate(positions_by_repeat):
        x = x_full.iloc[positions]
        interaction = exact_tree_shap_interactions(model, x)
        local_target = y.iloc[positions].to_numpy(dtype=float)
        records.extend(
            _candidate_rows(
                interaction,
                model_columns=model_columns,
                predictor_columns=predictor_columns,
                probability_columns=regime_columns,
                namespace="regime_probability",
                x=x,
                target=local_target,
                model=model,
                config=config,
                repeat=repeat,
            )
        )
        records.extend(
            _candidate_rows(
                interaction,
                model_columns=model_columns,
                predictor_columns=predictor_columns,
                probability_columns=transition_columns,
                namespace="transition_probability",
                x=x,
                target=local_target,
                model=model,
                config=config,
                repeat=repeat,
            )
        )
    all_rows = pd.DataFrame.from_records(records)
    regime_result = _aggregate_candidates(
        all_rows.loc[all_rows.get("namespace", pd.Series(dtype=str)).eq("regime_probability")], config
    )
    transition_result = _aggregate_candidates(
        all_rows.loc[all_rows.get("namespace", pd.Series(dtype=str)).eq("transition_probability")], config
    )
    metadata = {
        "schema": INTERACTION_DISCOVERY_SCHEMA,
        "research_only": True,
        "fold_id": str(config.fold_id),
        "train_only_attested": bool(config.require_model_training_identity),
        "rows": int(len(predictors)),
        "model_columns": model_columns,
        "predictor_columns": predictor_columns,
        "regime_probability_columns": regime_columns,
        "transition_probability_columns": transition_columns,
        "regime_probability_namespace_source": regime_namespace_source,
        "transition_probability_namespace_source": transition_namespace_source,
        "probability_adapter": {
            "regime_state_p__": "regime_prob__",
            "transition_state_p__": "transition_prob__",
            "invariant": "renames one homogeneous post-OOF head output only; never merges or cross-uses heads",
        },
        "stability_subsamples": int(len(positions_by_repeat)),
        "subsample_rows": [int(len(positions)) for positions in positions_by_repeat],
        "tree_interaction_method": "exact_tree_shap",
        "outcome_predictor_denylist_enforced": True,
    }
    return regime_result, transition_result, metadata


__all__ = [
    "FORBIDDEN_PREDICTOR_TOKENS",
    "INTERACTION_DISCOVERY_SCHEMA",
    "InteractionDiscoveryConfig",
    "REGIME_PROBABILITY_PREFIX",
    "REGIME_STATE_PROBABILITY_PREFIX",
    "TRANSITION_PROBABILITY_PREFIX",
    "TRANSITION_STATE_PROBABILITY_PREFIX",
    "TreeInteractionUnavailableError",
    "adapt_state_probability_namespace",
    "deterministic_stratified_subsample_positions",
    "discover_fold_local_regime_interactions",
    "exact_tree_shap_interactions",
]
