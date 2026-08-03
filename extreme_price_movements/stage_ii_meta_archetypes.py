"""Stage-II causal conversion-archetype features.

This module deliberately stops before fitting a trading/residual expert.  It
discovers *side-local*, realised-path conversion modes on a training slice,
then trains a causal recogniser which emits only soft memberships for later
rows.  The train-only mean conversion residual of each mode is exposed as a
single, membership-weighted prior.  Stage III can consume those values as
context for one shared residual expert; this module never routes a candidate
to a local trading model or emits a hard archetype id.

The strict-OOF helper is the supported way to create research features.  For a
validation timestamp ``t`` every clustering, classifier and residual prior is
fit only on rows satisfying ``label_available_ts < t``.  This is intentionally
stricter than merely training on an earlier decision timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


META_ARCHETYPE_PREFIX = "meta_conversion_arch_"
UNKNOWN_MEMBERSHIP_COLUMN = f"{META_ARCHETYPE_PREFIX}prob__unknown"
OUTCOME_SUFFIXES = (
    "exact_net_bps",
    "first_touch",
    "mfe",
    "mae",
    "timeout",
    "event_",
    "retention",
    "giveback",
    "future_",
    "path_",
)


@dataclass(frozen=True)
class StageIIMetaArchetypeConfig:
    """Contract for side-local conversion-mode discovery.

    ``path_descriptor_cols`` are realised path coordinates.  They are used to
    discover modes on the train slice only, are forbidden at transform time,
    and must never be placed in ``causal_feature_cols``.  The base expected-net
    column is a strict-OOF base handoff and remains a valid causal input.
    """

    decision_ts_col: str = "decision_ts"
    label_available_ts_col: str = "label_available_ts"
    side_col: str = "side_name"
    exact_net_col: str = "exact_net_bps"
    base_expected_net_col: str = "prequential_base_expected_net_bps"
    path_descriptor_cols: tuple[str, ...] = ()
    components: int = 4
    min_side_rows: int = 500
    min_component_rows: int = 25
    min_train_rows: int = 1_000
    oof_folds: int = 4
    classifier_c: float = 0.5
    random_state: int = 20260803


@dataclass
class _SideModel:
    side: str
    causal_features: list[str]
    input_scaler: RobustScaler
    outcome_scaler: RobustScaler
    clusterer: GaussianMixture
    classifier: LogisticRegression | None
    classifier_classes: np.ndarray
    # ``rank`` is intentionally ordered by train-only mean conversion residual
    # to give the fixed output dimensions stable economic interpretation.
    original_to_rank: dict[int, int]
    residual_by_rank: np.ndarray
    support_by_rank: np.ndarray


@dataclass
class StageIIMetaArchetypeOOFResult:
    """OOF features plus non-feature lineage and train-side catalogues."""

    features: pd.DataFrame
    fold_audit: pd.DataFrame
    catalog: pd.DataFrame
    # Realised-path memberships are deliberately separated from ``features``.
    # They support causal-predictability diagnostics only and are never an
    # inference/model input.
    diagnostic_truth_memberships: pd.DataFrame
    manifest: dict[str, Any]


def membership_feature_names(components: int) -> list[str]:
    if int(components) < 2:
        raise ValueError("components must be at least two")
    return [f"{META_ARCHETYPE_PREFIX}prob__{rank}" for rank in range(int(components))]


def stage_ii_feature_names(components: int) -> list[str]:
    return [
        *membership_feature_names(components),
        UNKNOWN_MEMBERSHIP_COLUMN,
        f"{META_ARCHETYPE_PREFIX}prior_residual_bps",
        f"{META_ARCHETYPE_PREFIX}entropy",
        f"{META_ARCHETYPE_PREFIX}confidence",
        f"{META_ARCHETYPE_PREFIX}support_log1p",
        f"{META_ARCHETYPE_PREFIX}available",
    ]


def _as_utc(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} must contain only valid UTC timestamps")
    return result


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _outcome_columns(config: StageIIMetaArchetypeConfig) -> set[str]:
    return {config.exact_net_col, *config.path_descriptor_cols}


def _validate_causal_features(
    frame: pd.DataFrame,
    causal_feature_cols: Iterable[str],
    config: StageIIMetaArchetypeConfig,
) -> list[str]:
    features = list(dict.fromkeys(str(name) for name in causal_feature_cols))
    if not features:
        raise ValueError("Stage II requires an explicit non-empty causal feature list")
    illegal = sorted(set(features).intersection(_outcome_columns(config)))
    if illegal:
        raise ValueError(f"Realised path/outcome columns cannot be causal features: {illegal}")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Causal feature columns are missing: {missing}")
    non_numeric = [name for name in features if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        raise ValueError(f"Causal feature columns must be numeric: {non_numeric}")
    return features


def _causal_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=np.float64, copy=True)
    # The imputer is fit on the permitted training slice below.  This temporary
    # fill is only used after those train medians have been written into the
    # matrix by ``_impute_from_training``.
    return array


def _impute_from_training(train: np.ndarray, apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_out = np.where(np.isfinite(train), train, medians)
    apply_out = np.where(np.isfinite(apply), apply, medians)
    return train_out.astype(np.float32), apply_out.astype(np.float32)


def _path_matrix(frame: pd.DataFrame, config: StageIIMetaArchetypeConfig) -> tuple[np.ndarray, np.ndarray]:
    net = _numeric(frame, config.exact_net_col).to_numpy(dtype=np.float64)
    base = _numeric(frame, config.base_expected_net_col).to_numpy(dtype=np.float64)
    residual = net - base
    columns = [residual, net]
    valid = np.isfinite(residual) & np.isfinite(net)
    for name in config.path_descriptor_cols:
        values = _numeric(frame, name).to_numpy(dtype=np.float64)
        columns.append(values)
        valid &= np.isfinite(values)
    return np.column_stack(columns).astype(np.float32), valid


def _safe_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    denom = np.log(max(probabilities.shape[1], 2))
    return (-np.sum(clipped * np.log(clipped), axis=1) / denom).astype(np.float32)


class SideLocalMetaArchetypeState:
    """A frozen side-local conversion-archetype representation.

    The only OOS operation is a causal classifier prediction and a weighted
    lookup of train-side residual priors.  No realised path values, hard
    assignment, side expert, or policy action is exposed by :meth:`transform`.
    """

    def __init__(
        self,
        config: StageIIMetaArchetypeConfig,
        causal_feature_cols: Sequence[str],
    ) -> None:
        self.config = config
        self.causal_feature_cols = list(causal_feature_cols)
        self.side_models: dict[str, _SideModel] = {}
        self.catalog_: pd.DataFrame = pd.DataFrame()

    def fit(self, train: pd.DataFrame) -> "SideLocalMetaArchetypeState":
        cfg = self.config
        if not cfg.path_descriptor_cols:
            raise ValueError(
                "Stage-II conversion archetypes require at least one realised "
                "path_descriptor_col; residual/net alone is not path-defined"
            )
        features = _validate_causal_features(train, self.causal_feature_cols, cfg)
        if cfg.side_col not in train.columns:
            raise ValueError(f"Missing side column: {cfg.side_col}")
        _, outcome_valid = _path_matrix(train, cfg)
        rows: list[dict[str, Any]] = []
        side = train[cfg.side_col].astype(str).str.lower()
        for side_name, idx in side.groupby(side, sort=True).groups.items():
            positions = train.index.get_indexer(idx)
            subset = train.iloc[positions]
            descriptor, valid = _path_matrix(subset, cfg)
            if int(valid.sum()) < max(cfg.min_side_rows, cfg.components * cfg.min_component_rows):
                continue
            x_raw = _causal_matrix(subset, features)
            x_train, _ = _impute_from_training(x_raw[valid], x_raw[valid])
            descriptor_train = descriptor[valid]
            max_components = min(
                int(cfg.components),
                max(2, int(valid.sum()) // int(cfg.min_component_rows)),
            )
            if max_components < 2:
                continue
            outcome_scaler = RobustScaler().fit(descriptor_train)
            descriptor_scaled = outcome_scaler.transform(descriptor_train)
            clusterer: GaussianMixture | None = None
            hard_original: np.ndarray | None = None
            components = 0
            # GMMs occasionally leave a nominal component without enough hard
            # support.  Rather than publish a NaN/phantom archetype, shrink K
            # deterministically and keep the representation explicitly small.
            for proposed_components in range(max_components, 1, -1):
                candidate = GaussianMixture(
                    n_components=proposed_components,
                    covariance_type="diag",
                    reg_covar=1e-4,
                    random_state=int(cfg.random_state),
                    n_init=2,
                ).fit(descriptor_scaled)
                candidate_hard = candidate.predict(descriptor_scaled).astype(np.int32)
                support = np.bincount(candidate_hard, minlength=proposed_components)
                if np.all(support >= int(cfg.min_component_rows)):
                    clusterer = candidate
                    hard_original = candidate_hard
                    components = proposed_components
                    break
            if clusterer is None or hard_original is None:
                continue
            residual = descriptor_train[:, 0].astype(np.float64)
            means = np.asarray(
                [float(residual[hard_original == component].mean()) for component in range(components)],
                dtype=np.float64,
            )
            # Stable rank: least favourable conversion first, ties by original
            # GMM index.  Thus an output slot is never a fold-local arbitrary id.
            ordered_original = np.lexsort((np.arange(components), means))
            original_to_rank = {int(original): int(rank) for rank, original in enumerate(ordered_original)}
            y_rank = np.asarray([original_to_rank[int(value)] for value in hard_original], dtype=np.int32)
            residual_by_rank = np.zeros(int(cfg.components), dtype=np.float32)
            support_by_rank = np.zeros(int(cfg.components), dtype=np.float32)
            for original, rank in original_to_rank.items():
                mask = hard_original == original
                residual_by_rank[rank] = np.float32(residual[mask].mean())
                support_by_rank[rank] = np.float32(mask.sum())
                rows.append(
                    {
                        "side": str(side_name),
                        "rank": int(rank),
                        "support_rows": int(mask.sum()),
                        "mean_conversion_residual_bps": float(residual[mask].mean()),
                        "mean_exact_net_bps": float(descriptor_train[mask, 1].mean()),
                        "path_dimensions": int(descriptor_train.shape[1]),
                    }
                )
            input_scaler = RobustScaler().fit(x_train)
            classifier: LogisticRegression | None
            classes = np.unique(y_rank)
            if len(classes) >= 2:
                classifier = LogisticRegression(
                    C=float(cfg.classifier_c),
                    max_iter=500,
                    random_state=int(cfg.random_state),
                ).fit(input_scaler.transform(x_train), y_rank)
                classifier_classes = classifier.classes_.astype(np.int32, copy=False)
            else:
                classifier = None
                classifier_classes = classes.astype(np.int32, copy=False)
            self.side_models[str(side_name)] = _SideModel(
                side=str(side_name),
                causal_features=features,
                input_scaler=input_scaler,
                outcome_scaler=outcome_scaler,
                clusterer=clusterer,
                classifier=classifier,
                classifier_classes=classifier_classes,
                original_to_rank=original_to_rank,
                residual_by_rank=residual_by_rank,
                support_by_rank=support_by_rank,
            )
        self.catalog_ = pd.DataFrame(rows).sort_values(["side", "rank"], kind="stable") if rows else pd.DataFrame(
            columns=["side", "rank", "support_rows", "mean_conversion_residual_bps", "mean_exact_net_bps", "path_dimensions"]
        )
        return self

    def diagnostic_realised_memberships(self, labelled_rows: pd.DataFrame) -> pd.DataFrame:
        """Return realised-path memberships for evaluation, never inference.

        This method is intentionally distinct from :meth:`transform`: callers
        may use its output for held-out Brier/log-loss and temporal-stability
        reports, but must not append it to causal feature matrices.
        """

        cfg = self.config
        output = pd.DataFrame(
            np.nan,
            index=labelled_rows.index,
            columns=membership_feature_names(cfg.components),
            dtype=np.float32,
        )
        if cfg.side_col not in labelled_rows.columns:
            raise ValueError(f"Missing side column: {cfg.side_col}")
        side = labelled_rows[cfg.side_col].astype(str).str.lower()
        for side_name, idx in side.groupby(side, sort=False).groups.items():
            model = self.side_models.get(str(side_name))
            if model is None:
                continue
            positions = labelled_rows.index.get_indexer(idx)
            subset = labelled_rows.iloc[positions]
            descriptor, valid = _path_matrix(subset, cfg)
            if not valid.any():
                continue
            original = model.clusterer.predict_proba(
                model.outcome_scaler.transform(descriptor[valid])
            )
            ranked = np.zeros((int(valid.sum()), int(cfg.components)), dtype=np.float32)
            for original_component, rank in model.original_to_rank.items():
                ranked[:, rank] = original[:, original_component]
            output.loc[subset.index[valid], :] = ranked
        return output

    def transform(self, oos_without_outcomes: pd.DataFrame) -> pd.DataFrame:
        """Emit causal soft memberships and the train-only residual prior."""

        cfg = self.config
        leaked = sorted(set(oos_without_outcomes.columns).intersection(_outcome_columns(cfg)))
        if leaked:
            raise ValueError(f"OOS/inference frame received realised path columns: {leaked}")
        _validate_causal_features(oos_without_outcomes, self.causal_feature_cols, cfg)
        if cfg.side_col not in oos_without_outcomes.columns:
            raise ValueError(f"Missing side column: {cfg.side_col}")
        names = stage_ii_feature_names(cfg.components)
        out = pd.DataFrame(0.0, index=oos_without_outcomes.index, columns=names, dtype=np.float32)
        out[UNKNOWN_MEMBERSHIP_COLUMN] = np.float32(1.0)
        side = oos_without_outcomes[cfg.side_col].astype(str).str.lower()
        for side_name, idx in side.groupby(side, sort=False).groups.items():
            model = self.side_models.get(str(side_name))
            if model is None:
                continue
            positions = oos_without_outcomes.index.get_indexer(idx)
            subset = oos_without_outcomes.iloc[positions]
            x_raw = _causal_matrix(subset, model.causal_features)
            # Recover the medians from RobustScaler's centre: this is exactly
            # the median fitted on the allowed train slice.
            medians = np.asarray(model.input_scaler.center_, dtype=np.float64)
            x_filled = np.where(np.isfinite(x_raw), x_raw, medians).astype(np.float32)
            probabilities = np.zeros((len(subset), int(cfg.components)), dtype=np.float32)
            if model.classifier is None:
                rank = int(model.classifier_classes[0]) if len(model.classifier_classes) else 0
                probabilities[:, rank] = 1.0
            else:
                learned = model.classifier.predict_proba(model.input_scaler.transform(x_filled))
                for col, rank in enumerate(model.classifier_classes):
                    probabilities[:, int(rank)] = learned[:, col]
            target_index = subset.index
            for rank, name in enumerate(membership_feature_names(cfg.components)):
                out.loc[target_index, name] = probabilities[:, rank]
            out.loc[target_index, UNKNOWN_MEMBERSHIP_COLUMN] = 0.0
            out.loc[target_index, f"{META_ARCHETYPE_PREFIX}prior_residual_bps"] = probabilities @ model.residual_by_rank
            out.loc[target_index, f"{META_ARCHETYPE_PREFIX}entropy"] = _safe_entropy(probabilities)
            out.loc[target_index, f"{META_ARCHETYPE_PREFIX}confidence"] = probabilities.max(axis=1)
            expected_support = probabilities @ model.support_by_rank
            out.loc[target_index, f"{META_ARCHETYPE_PREFIX}support_log1p"] = np.log1p(expected_support)
            out.loc[target_index, f"{META_ARCHETYPE_PREFIX}available"] = 1.0
        return out.astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "stage_ii_meta_conversion_archetypes_v1",
            "construction": "side_local_realised_path_clusters_then_causal_soft_memberships",
            "hard_routing": False,
            "local_trading_experts": False,
            "causal_feature_cols": list(self.causal_feature_cols),
            "path_descriptor_cols": list(self.config.path_descriptor_cols),
            "base_handoff": self.config.base_expected_net_col,
            "residual_definition": f"{self.config.exact_net_col} - {self.config.base_expected_net_col}",
            "oos_contract": "transform rejects realised path columns",
            "side_models": sorted(self.side_models),
        }


def _strict_oof_boundaries(decision: pd.Series, folds: int) -> list[pd.Timestamp]:
    times = np.sort(pd.unique(decision.to_numpy(dtype="datetime64[ns]")))
    if len(times) < 2:
        return []
    # First chronological block is a burn-in only.  Equal timestamp ranges
    # ensure no same-decision rows cross a fit/validation boundary.
    cuts = np.linspace(0, len(times), int(folds) + 2, dtype=np.int64)[1:-1]
    return [pd.Timestamp(times[cut], tz="UTC") for cut in np.unique(cuts) if 0 < cut < len(times)]


def strict_oof_meta_archetype_features(
    frame: pd.DataFrame,
    *,
    config: StageIIMetaArchetypeConfig,
    causal_feature_cols: Sequence[str],
) -> StageIIMetaArchetypeOOFResult:
    """Create strict chronological OOF Stage-II features.

    Each validation block is transformed only after its side-local path modes,
    membership recogniser and residual priors have been fit on prior-resolved
    rows.  Uncovered burn-in rows retain an explicit ``unknown`` membership;
    callers must not silently treat them as an ordinary conversion mode.
    """

    cfg = config
    features = _validate_causal_features(frame, causal_feature_cols, cfg)
    for column in (cfg.decision_ts_col, cfg.label_available_ts_col, cfg.side_col, cfg.exact_net_col, cfg.base_expected_net_col):
        if column not in frame.columns:
            raise ValueError(f"Missing required Stage-II column: {column}")
    decision = _as_utc(frame[cfg.decision_ts_col], cfg.decision_ts_col)
    available = _as_utc(frame[cfg.label_available_ts_col], cfg.label_available_ts_col)
    if (available <= decision).any():
        raise ValueError("Stage-II labels must resolve strictly after their decision timestamp")
    # Invalid target rows cannot train path clusters.  They may still appear in
    # OOS output, but cannot create future knowledge in a prior model.
    descriptor, outcome_valid = _path_matrix(frame, cfg)
    del descriptor
    base_output = pd.DataFrame(0.0, index=frame.index, columns=stage_ii_feature_names(cfg.components), dtype=np.float32)
    base_output[UNKNOWN_MEMBERSHIP_COLUMN] = np.float32(1.0)
    audit_rows: list[dict[str, Any]] = []
    catalogues: list[pd.DataFrame] = []
    diagnostic_truth = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=membership_feature_names(cfg.components),
        dtype=np.float32,
    )
    boundaries = _strict_oof_boundaries(decision, cfg.oof_folds)
    for fold, start in enumerate(boundaries):
        later = boundaries[fold + 1] if fold + 1 < len(boundaries) else None
        valid_mask = decision.ge(start) if later is None else decision.ge(start) & decision.lt(later)
        train_mask = available.lt(start) & outcome_valid
        train = frame.loc[train_mask]
        valid = frame.loc[valid_mask]
        if len(valid) == 0:
            continue
        if len(train) < int(cfg.min_train_rows):
            audit_rows.append(
                {"fold": fold, "valid_start": start, "valid_end": later, "train_rows": int(len(train)), "valid_rows": int(len(valid)), "status": "insufficient_prior_rows", "train_max_label_available_ts": available.loc[train_mask].max() if train_mask.any() else pd.NaT}
            )
            continue
        state = SideLocalMetaArchetypeState(cfg, features).fit(train)
        safe = valid.drop(columns=list(_outcome_columns(cfg)), errors="ignore")
        transformed = state.transform(safe)
        base_output.loc[valid.index, transformed.columns] = transformed
        truth = state.diagnostic_realised_memberships(valid)
        diagnostic_truth.loc[truth.index, truth.columns] = truth
        truth_values = truth.to_numpy(dtype=np.float32)
        predicted_values = transformed.loc[:, truth.columns].to_numpy(dtype=np.float32)
        diagnostic_mask = np.isfinite(truth_values).all(axis=1)
        if diagnostic_mask.any():
            clipped = np.clip(predicted_values[diagnostic_mask], 1e-8, 1.0)
            cross_entropy = -np.sum(
                truth_values[diagnostic_mask] * np.log(clipped), axis=1
            )
            brier = np.mean(
                (predicted_values[diagnostic_mask] - truth_values[diagnostic_mask]) ** 2,
                axis=1,
            )
            causal_membership_log_loss = float(np.mean(cross_entropy))
            causal_membership_brier = float(np.mean(brier))
        else:
            causal_membership_log_loss = np.nan
            causal_membership_brier = np.nan
        catalogue = state.catalog_.copy()
        if not catalogue.empty:
            catalogue.insert(0, "fold", fold)
            catalogue.insert(1, "valid_start", start)
            catalogues.append(catalogue)
        audit_rows.append(
            {"fold": fold, "valid_start": start, "valid_end": later, "train_rows": int(len(train)), "valid_rows": int(len(valid)), "status": "scored", "train_max_label_available_ts": available.loc[train_mask].max(), "fitted_sides": ",".join(sorted(state.side_models)), "causal_membership_log_loss": causal_membership_log_loss, "causal_membership_brier": causal_membership_brier, "diagnostic_labelled_rows": int(diagnostic_mask.sum())}
        )
    fold_audit = pd.DataFrame(audit_rows)
    catalog = pd.concat(catalogues, ignore_index=True) if catalogues else pd.DataFrame()
    manifest = {
        "schema": "stage_ii_meta_conversion_archetypes_oof_v1",
        "strict_oof": True,
        "prior_resolution_rule": f"{cfg.label_available_ts_col} < validation_decision_ts",
        "side_local_construction": True,
        "soft_memberships_only": True,
        "hard_routing": False,
        "local_trading_experts": False,
        "causal_feature_cols": features,
        "path_descriptor_cols": list(cfg.path_descriptor_cols),
        "diagnostic_truth_memberships": "evaluation_only_not_a_model_feature",
        "oof_folds_requested": int(cfg.oof_folds),
        "scored_rows": int(base_output[f"{META_ARCHETYPE_PREFIX}available"].sum()),
        "unknown_rows": int(base_output[UNKNOWN_MEMBERSHIP_COLUMN].sum()),
    }
    return StageIIMetaArchetypeOOFResult(
        base_output, fold_audit, catalog, diagnostic_truth, manifest
    )
