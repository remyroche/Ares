"""Joint Stage-V grouped-MDA drift/OOD ablation for the native R3 → FQ3 stack.

Stage V is intentionally an *input-context* experiment.  Each arm retrains
the complete same-side chain::

    native R3 simplex -> native P(clear)-P(adverse)
        -> direct fold-local FQ3 correction
        -> reconstructed native score
        -> causal 21-day common-bps map -> one pooled-global rank

There is no standalone common-bps base regression and no independently fitted
meta regressor.  The only promotable result is reconstructed-meta economics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .stage_i_causal_admission import Causal21dAdmissionSpec, apply_causal_21d_side_admission
from .stage_i_production_oos import _selection_metrics
from .stage_i_target_specific_oos import (
    DIRECT_BASE_INPUT_SEMANTICS,
    DIRECT_FQ3_SEMANTICS,
    _fit_direct_correctness,
    _reconstruct_direct_correctness,
)
from .stage_iv_v_orchestration import freeze_stage_v_feature_contract
from .stage_v_drift_ood import STAGE_V_FEATURE_COLUMNS, STAGE_V_SCHEMA, StageVContract, fit_stage_v_drift_ood_state, transform_stage_v_drift_ood_features


STAGE_V_EXPERIMENT_SCHEMA = "stage_v_joint_native_r3_direct_fq3_drift_ood_v2"
STAGE_V_ARMS = ("control", "base_ood", "meta_ood", "both_ood")
_SIDES = ("long", "short")
_FORBIDDEN_FQ3 = ("prequential", "causal_21d", "mapped", "expected_net", "converted_ev", "converted_score", "meta_direct", "meta_p_")
_BASE_ALIASES = {
    "base_raw_score", "base_direct_score", "r3_opportunity_score",
    "r3_p_adverse", "r3_p_weak", "r3_p_clear",
    "base_r3_entropy", "base_r3_top2_margin", "base_r3_max_probability",
    "base_output_entropy", "base_output_top2_margin", "base_output_max_probability",
}
_REQUIRED_META_HANDOFF = (
    "base_raw_score", "r3_p_adverse", "r3_p_weak", "r3_p_clear",
    "base_r3_entropy", "base_r3_top2_margin", "base_r3_max_probability",
)


class StageVExperimentError(ValueError):
    """Raised when the joint native-score Stage-V contract is violated."""


class _StageVContextUnavailable(StageVExperimentError):
    """A requested OOD arm has no positive frozen MDA group in this fold."""


ModelFitter = Callable[[pd.DataFrame, np.ndarray, str, str, str, str], Any]


@dataclass(frozen=True)
class StageVLayerSource:
    """Frozen selected raw contract for one side and one stack layer.

    For ``base`` the ``target_column`` is the native integer R3 class.  For
    ``meta`` it is retained only as source provenance: direct FQ3 labels are
    always built fold-locally from ``exact_net_bps`` and the arm's native base
    handoff, never from this column.
    """

    layer: str
    side: str
    selector: pd.DataFrame
    oos: pd.DataFrame
    raw_feature_names: tuple[str, ...]
    mda_group_audit: pd.DataFrame
    target_column: str
    selector_manifest_sha256: str
    oos_surface_lineage: Mapping[str, Any]
    target_units: str = "native_r3_class"  # base; meta may use provenance-only.

    @property
    def normalized_layer(self) -> str:
        return str(self.layer).lower()

    @property
    def normalized_side(self) -> str:
        return str(self.side).lower()

    def validate(self) -> None:
        if self.normalized_layer not in {"base", "meta"} or self.normalized_side not in _SIDES:
            raise StageVExperimentError("Stage-V source must be a base/meta × long/short cell")
        if not self.raw_feature_names or len(set(self.raw_feature_names)) != len(self.raw_feature_names):
            raise StageVExperimentError("Stage-V selected raw feature contract must be non-empty and unique")
        if self.mda_group_audit.empty:
            raise StageVExperimentError("Stage-V requires frozen grouped-MDA audit evidence")
        if not bool(self.oos_surface_lineage.get("declared", False)):
            raise StageVExperimentError("Stage-V requires declared frozen OOS-surface lineage")
        if len(str(self.selector_manifest_sha256)) != 64:
            raise StageVExperimentError("Stage-V requires selector-manifest SHA-256 lineage")
        if self.normalized_layer == "meta" and any(token in name.lower() for name in self.raw_feature_names for token in _FORBIDDEN_FQ3):
            raise StageVExperimentError("direct FQ3 meta contract contains mapped/expected-net feature")
        for role, frame in (("selector", self.selector), ("oos", self.oos)):
            required = {"candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "exact_gross_bps", *self.raw_feature_names}
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise StageVExperimentError(f"{self.layer}/{self.side} {role} lacks required fields: {missing[:12]}")
            if "symbol" not in frame and "__symbol__" not in frame:
                raise StageVExperimentError(f"{self.layer}/{self.side} {role} requires symbol or __symbol__")
            if frame.candidate_id.isna().any() or frame.candidate_id.duplicated().any():
                raise StageVExperimentError(f"{self.layer}/{self.side} {role} needs unique candidate_id")
            if not frame.side_name.astype(str).str.lower().eq(self.normalized_side).all():
                raise StageVExperimentError(f"{self.layer}/{self.side} {role} mixes sides")
            decision = pd.to_datetime(frame.decision_ts, utc=True, errors="coerce")
            available = pd.to_datetime(frame.label_available_ts, utc=True, errors="coerce")
            if decision.isna().any() or available.isna().any() or (available <= decision).any():
                raise StageVExperimentError(f"{self.layer}/{self.side} {role} violates label timing")
        if self.normalized_layer == "base":
            if self.target_column not in self.selector:
                raise StageVExperimentError("native R3 base target is absent")
            label = pd.to_numeric(self.selector[self.target_column], errors="coerce").to_numpy(float)
            if not np.isin(label[np.isfinite(label)], (0.0, 1.0, 2.0)).all():
                raise StageVExperimentError("Stage-V base target must be native R3 classes 0/1/2")


@dataclass(frozen=True)
class StageVExperimentConfig:
    folds: int = 4
    min_train_rows: int = 500
    max_groups: int = 24
    selection_top_fraction: float = 0.10
    min_selected_rows: int = 50
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec()
    arms: tuple[str, ...] = STAGE_V_ARMS

    def validate(self) -> None:
        if self.folds < 2 or self.min_train_rows < 16:
            raise StageVExperimentError("Stage-V needs >=2 folds and >=16 prior rows")
        if not 1 <= self.max_groups <= 64 or not 0 < self.selection_top_fraction <= 1 or self.min_selected_rows < 1:
            raise StageVExperimentError("Stage-V configuration is out of bounds")
        if tuple(self.arms) != STAGE_V_ARMS or self.admission_spec.window_days != 21:
            raise StageVExperimentError("Stage-V uses fixed joint arms and canonical 21-day mapping")


def default_stage_v_model_fitter(X: pd.DataFrame, y: np.ndarray, _layer: str, _side: str, _arm: str, _phase: str) -> HistGradientBoostingClassifier:
    """Bounded deterministic multiclass fitter; production may inject frozen LGBM."""
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=24,
        l2_regularization=1.0, max_iter=180, random_state=20260803,
    ).fit(X, y)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _normalise(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    out = frame.copy()
    out.side_name = out.side_name.astype(str).str.lower()
    out.decision_ts = pd.to_datetime(out.decision_ts, utc=True, errors="raise")
    out.label_available_ts = pd.to_datetime(out.label_available_ts, utc=True, errors="raise")
    out["symbol"] = out["symbol"].astype(str) if "symbol" in out else out["__symbol__"].astype(str)
    if not out.side_name.eq(side).all():
        raise StageVExperimentError("cross-side source after normalisation")
    return out.sort_values(["decision_ts", "candidate_id"], kind="stable").reset_index(drop=True)


def _fold_starts(frame: pd.DataFrame, folds: int) -> list[pd.Timestamp]:
    values = np.sort(frame.decision_ts.drop_duplicates().to_numpy(dtype="datetime64[ns]"))
    if len(values) < folds + 2:
        return []
    cuts = np.unique(np.linspace(0, len(values), folds + 2, dtype=np.int64)[1:-1])
    return [pd.Timestamp(values[pos], tz="UTC") for pos in cuts if 0 < pos < len(values)]


def _arm_context(arm: str, layer: str) -> tuple[str, ...]:
    if arm == "control" or (arm == "base_ood" and layer == "meta") or (arm == "meta_ood" and layer == "base"):
        return ()
    return STAGE_V_FEATURE_COLUMNS


def _three_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(frame), dtype=float)
    classes = np.asarray(getattr(model, "classes_", ()), dtype=int)
    output = np.zeros((len(frame), 3), dtype=np.float32)
    for pos, label in enumerate(classes):
        if label in (0, 1, 2):
            output[:, int(label)] = raw[:, pos]
    if not np.isfinite(output).all() or (output.sum(axis=1) <= 0).any():
        raise StageVExperimentError("joint model did not produce an R3/FQ3 three-class probability simplex")
    output /= output.sum(axis=1, keepdims=True)
    return output


def _trust(p: np.ndarray) -> dict[str, np.ndarray]:
    ordered = np.sort(p, axis=1)
    entropy = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
    return {
        "base_r3_entropy": entropy, "base_r3_top2_margin": ordered[:, -1] - ordered[:, -2],
        "base_r3_max_probability": ordered[:, -1], "base_output_entropy": entropy,
        "base_output_top2_margin": ordered[:, -1] - ordered[:, -2], "base_output_max_probability": ordered[:, -1],
    }


def _build_meta_design(raw: pd.DataFrame, *, names: Sequence[str], base_score: np.ndarray, base_p: np.ndarray, context: pd.DataFrame) -> pd.DataFrame:
    if any(any(token in str(name).lower() for token in _FORBIDDEN_FQ3) for name in names):
        raise StageVExperimentError("mapped/expected-net feature attempted to enter direct FQ3")
    if len(raw) != len(base_score) or base_p.shape != (len(raw), 3):
        raise StageVExperimentError("same-side base handoff is not row-aligned")
    derived: dict[str, np.ndarray] = {
        "base_raw_score": base_score, "base_direct_score": base_score, "r3_opportunity_score": base_score,
        "r3_p_adverse": base_p[:, 0], "r3_p_weak": base_p[:, 1], "r3_p_clear": base_p[:, 2], **_trust(base_p),
    }
    pieces: dict[str, Any] = {}
    for name in names:
        pieces[str(name)] = derived[str(name)] if str(name) in derived else raw[str(name)].to_numpy()
    # These protected native handoffs are injected even if a faulty selector
    # omitted them.  They replace, never supplement, any raw stale handoff.
    for name in _REQUIRED_META_HANDOFF:
        pieces[name] = derived[name]
    for name in context.columns:
        pieces[str(name)] = context[name].to_numpy()
    design = pd.DataFrame(pieces, index=raw.index)
    if any(any(token in name.lower() for token in _FORBIDDEN_FQ3) for name in design.columns):
        raise StageVExperimentError("FQ3 design contains a forbidden converted-score feature")
    if not np.isfinite(design.apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
        raise StageVExperimentError("FQ3 selected context/base handoff has non-finite rows")
    return design


def _paired_sources(sources: Sequence[StageVLayerSource]) -> dict[str, tuple[StageVLayerSource, StageVLayerSource]]:
    expected = {(layer, side) for layer in ("base", "meta") for side in _SIDES}
    cells = {(source.normalized_layer, source.normalized_side): source for source in sources}
    if set(cells) != expected or len(sources) != 4:
        raise StageVExperimentError("Stage-V requires exactly base/meta × long/short sources")
    pairs: dict[str, tuple[StageVLayerSource, StageVLayerSource]] = {}
    for side in _SIDES:
        base, meta = cells[("base", side)], cells[("meta", side)]
        for left, right, name in ((base.selector, meta.selector, "selector"), (base.oos, meta.oos, "oos")):
            key = ["candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "exact_gross_bps"]
            a, b = _normalise(left, side), _normalise(right, side)
            if not a.loc[:, key].reset_index(drop=True).equals(b.loc[:, key].reset_index(drop=True)):
                raise StageVExperimentError(f"{side} base/meta {name} population or label contract is not identical")
        pairs[side] = (base, meta)
    return pairs


def _fit_context(train: pd.DataFrame, apply: pd.DataFrame, source: StageVLayerSource, config: StageVExperimentConfig, *, requested: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit a state only when this arm actually injects Stage-V fields.

    A control is a true upstream control: it cannot be made unavailable by a
    lack of positive grouped-MDA evidence, and it never pays a drift fit.
    """
    if not requested:
        state = {
            "enabled": False, "reason": "controller_none_no_context_requested",
            "contract": {"side": source.normalized_side, "layer": source.normalized_layer},
        }
        return pd.DataFrame(index=train.index), pd.DataFrame(index=apply.index), state
    state = fit_stage_v_drift_ood_state(train.loc[:, list(source.raw_feature_names)], contract=StageVContract(source.normalized_side, source.normalized_layer), mda_audit=source.mda_group_audit, feature_columns=source.raw_feature_names, max_groups=config.max_groups)
    if not state.get("enabled", False):
        raise _StageVContextUnavailable(f"{source.layer}/{source.side} has no positive frozen MDA group")
    return (
        transform_stage_v_drift_ood_features(train, state, contract=StageVContract(source.normalized_side, source.normalized_layer)),
        transform_stage_v_drift_ood_features(apply, state, contract=StageVContract(source.normalized_side, source.normalized_layer)), state,
    )


def _fit_base(train: pd.DataFrame, apply: pd.DataFrame, *, source: StageVLayerSource, context_train: pd.DataFrame, context_apply: pd.DataFrame, arm: str, fitter: ModelFitter) -> tuple[np.ndarray, np.ndarray, Any, tuple[str, ...]]:
    extras = _arm_context(arm, "base")
    names = tuple((*source.raw_feature_names, *extras))
    x_train = train.loc[:, list(source.raw_feature_names)].copy()
    x_apply = apply.loc[:, list(source.raw_feature_names)].copy()
    for name in extras:
        x_train[name], x_apply[name] = context_train[name].to_numpy(), context_apply[name].to_numpy()
    label = pd.to_numeric(train[source.target_column], errors="coerce").to_numpy(float)
    valid = np.isfinite(label) & np.isin(label, (0.0, 1.0, 2.0))
    if valid.sum() < 3 or len(np.unique(label[valid])) < 3:
        raise StageVExperimentError("R3 fold lacks all three native classes")
    model = fitter(x_train.loc[valid], label[valid].astype(np.int8), "base", source.normalized_side, arm, "native_r3")
    p = _three_probability(model, x_apply)
    return (p[:, 2] - p[:, 0]).astype(np.float32), p, model, names


def _strict_side(base: StageVLayerSource, meta: StageVLayerSource, *, config: StageVExperimentConfig, fitter: ModelFitter) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    b, m = _normalise(base.selector, base.normalized_side), _normalise(meta.selector, meta.normalized_side)
    result = b.loc[:, ["candidate_id", "symbol", "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "exact_gross_bps"]].copy()
    result["base_strict_oof_available"] = False
    result["meta_strict_oof_available"] = False
    result["side"] = base.normalized_side
    states: dict[str, Any] = {arm: {"base": [], "meta": []} for arm in STAGE_V_ARMS}
    audit: list[dict[str, Any]] = []
    starts = _fold_starts(b, config.folds)
    # First build strict native base handoffs for every controller arm.
    base_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm in STAGE_V_ARMS:
        score, p = np.full(len(b), np.nan, np.float32), np.full((len(b), 3), np.nan, np.float32)
        for fold, start in enumerate(starts):
            valid = b.decision_ts.ge(start) if fold + 1 == len(starts) else b.decision_ts.ge(start) & b.decision_ts.lt(starts[fold + 1])
            train = b.label_available_ts.lt(start)
            if int(train.sum()) < config.min_train_rows or not valid.any():
                audit.append({"side": base.normalized_side, "arm": arm, "layer": "base", "fold_id": fold, "status": "insufficient_prior_rows", "train_rows": int(train.sum()), "validation_rows": int(valid.sum())})
                continue
            try:
                ctrain, cvalid, state = _fit_context(
                    b.loc[train], b.loc[valid], base, config,
                    requested=bool(_arm_context(arm, "base")),
                )
            except _StageVContextUnavailable as exc:
                audit.append({"side": base.normalized_side, "arm": arm, "layer": "base", "fold_id": fold, "status": "ood_context_unavailable", "reason": str(exc), "train_rows": int(train.sum()), "validation_rows": int(valid.sum())})
                continue
            s, probability, model, names = _fit_base(b.loc[train], b.loc[valid], source=base, context_train=ctrain, context_apply=cvalid, arm=arm, fitter=fitter)
            score[valid], p[valid] = s, probability
            states[arm]["base"].append({"fold_id": fold, "state": state, "features": list(names), "model": model})
            audit.append({"side": base.normalized_side, "arm": arm, "layer": "base", "fold_id": fold, "status": "scored", "train_rows": int(train.sum()), "validation_rows": int(valid.sum()), "validation_start_utc": str(start), "train_max_label_available_ts": str(b.loc[train, "label_available_ts"].max()), "strict_prior_resolved": True})
        base_cache[arm] = (score, p)
        result[f"{arm}_base_direct_score"] = score
        result[f"{arm}_base_p_adverse"], result[f"{arm}_base_p_weak"], result[f"{arm}_base_p_clear"] = p[:, 0], p[:, 1], p[:, 2]
    # Then FQ3 only trains on previously emitted strict base handoffs.
    for arm in STAGE_V_ARMS:
        base_score, base_p = base_cache[arm]
        direct, correction, mp = np.full(len(b), np.nan, np.float32), np.full(len(b), np.nan, np.float32), np.full((len(b), 3), np.nan, np.float32)
        for fold, start in enumerate(starts):
            valid = b.decision_ts.ge(start) if fold + 1 == len(starts) else b.decision_ts.ge(start) & b.decision_ts.lt(starts[fold + 1])
            train = b.label_available_ts.lt(start) & np.isfinite(base_score)
            score = valid & np.isfinite(base_score)
            if int(train.sum()) < config.min_train_rows or not score.any():
                audit.append({"side": base.normalized_side, "arm": arm, "layer": "meta", "fold_id": fold, "status": "insufficient_prior_strict_base_rows", "train_rows": int(train.sum()), "validation_rows": int(score.sum())})
                continue
            try:
                ctrain, cvalid, state = _fit_context(
                    m.loc[train], m.loc[score], meta, config,
                    requested=bool(_arm_context(arm, "meta")),
                )
            except _StageVContextUnavailable as exc:
                audit.append({"side": base.normalized_side, "arm": arm, "layer": "meta", "fold_id": fold, "status": "ood_context_unavailable", "reason": str(exc), "train_rows": int(train.sum()), "validation_rows": int(score.sum())})
                continue
            design_train = _build_meta_design(m.loc[train], names=meta.raw_feature_names, base_score=base_score[train], base_p=base_p[train], context=ctrain.loc[:, list(_arm_context(arm, "meta"))])
            design_valid = _build_meta_design(m.loc[score], names=meta.raw_feature_names, base_score=base_score[score], base_p=base_p[score], context=cvalid.loc[:, list(_arm_context(arm, "meta"))])
            labels, fq3_state = _fit_direct_correctness(b.loc[train, "exact_net_bps"].to_numpy(float), base_score[train], score_domain=(-1.0, 1.0))
            model = fitter(design_train, labels, "meta", base.normalized_side, arm, "direct_fq3")
            probability = _three_probability(model, design_valid)
            delta, combined = _reconstruct_direct_correctness(probability, base_score[score], fq3_state)
            direct[score], correction[score], mp[score] = combined, delta, probability
            states[arm]["meta"].append({"fold_id": fold, "state": {"drift_ood": state, "fq3": fq3_state.to_dict(), "meta_features": list(design_train.columns)}, "features": list(design_train.columns), "model": model})
            audit.append({"side": base.normalized_side, "arm": arm, "layer": "meta", "fold_id": fold, "status": "scored", "train_rows": int(train.sum()), "validation_rows": int(score.sum()), "validation_start_utc": str(start), "train_max_label_available_ts": str(b.loc[train, "label_available_ts"].max()), "strict_prior_resolved": True, "target_semantics": DIRECT_FQ3_SEMANTICS, "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS})
        result[f"{arm}_meta_direct_score"] = direct
        result[f"{arm}_meta_direct_correction"] = correction
        result[f"{arm}_meta_p_error_tercile_0"], result[f"{arm}_meta_p_error_tercile_1"], result[f"{arm}_meta_p_error_tercile_2"] = mp[:, 0], mp[:, 1], mp[:, 2]
    result["base_strict_oof_available"] = np.isfinite(result[[f"{arm}_base_direct_score" for arm in STAGE_V_ARMS]]).any(axis=1)
    result["meta_strict_oof_available"] = np.isfinite(result[[f"{arm}_meta_direct_score" for arm in STAGE_V_ARMS]]).any(axis=1)
    return result, states, audit


def _frozen_side(base: StageVLayerSource, meta: StageVLayerSource, strict: pd.DataFrame, *, config: StageVExperimentConfig, fitter: ModelFitter) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    b, m, bo, mo = _normalise(base.selector, base.normalized_side), _normalise(meta.selector, meta.normalized_side), _normalise(base.oos, base.normalized_side), _normalise(meta.oos, meta.normalized_side)
    if not bo.decision_ts.min() > b.decision_ts.max():
        raise StageVExperimentError(f"{base.side} OOS must start after frozen selector")
    out = bo.loc[:, ["candidate_id", "symbol", "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "exact_gross_bps"]].copy()
    final: dict[str, Any] = {arm: {"base": None, "meta": None} for arm in STAGE_V_ARMS}
    audit: list[dict[str, Any]] = []
    for arm in STAGE_V_ARMS:
        for suffix in ("base_direct_score", "meta_direct_score", "meta_direct_correction"):
            out[f"{arm}_{suffix}"] = np.nan
        for suffix in ("base_p_adverse", "base_p_weak", "base_p_clear", "meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2"):
            out[f"{arm}_{suffix}"] = np.nan
        # Final base: selector labels only.  Its OOS score is native.
        try:
            bc_train, bc_oos, bstate = _fit_context(
                b, bo, base, config, requested=bool(_arm_context(arm, "base")),
            )
        except _StageVContextUnavailable as exc:
            final[arm]["base"] = {"state": {"enabled": False, "reason": str(exc)}, "features": [], "model": None}
            final[arm]["meta"] = {"state": {"enabled": False, "reason": "base_ood_context_unavailable"}, "features": [], "model": None}
            audit.append({"side": base.normalized_side, "arm": arm, "status": "ood_context_unavailable", "reason": str(exc)})
            continue
        bs, bp, bmodel, bnames = _fit_base(b, bo, source=base, context_train=bc_train, context_apply=bc_oos, arm=arm, fitter=fitter)
        # Final FQ3: train only on strict base OOF handoffs, never in-sample base scores.
        h = strict.loc[strict[f"{arm}_base_direct_score"].notna()].copy()
        h_index = h.index.to_numpy()
        try:
            mc_train, mc_oos, mstate = _fit_context(
                m.loc[h_index], mo, meta, config, requested=bool(_arm_context(arm, "meta")),
            )
        except _StageVContextUnavailable as exc:
            final[arm]["base"] = {"state": bstate, "features": list(bnames), "model": bmodel}
            final[arm]["meta"] = {"state": {"enabled": False, "reason": str(exc)}, "features": [], "model": None}
            audit.append({"side": base.normalized_side, "arm": arm, "status": "ood_context_unavailable", "reason": str(exc)})
            continue
        design_train = _build_meta_design(m.loc[h_index], names=meta.raw_feature_names, base_score=h[f"{arm}_base_direct_score"].to_numpy(float), base_p=h.loc[:, [f"{arm}_base_p_adverse", f"{arm}_base_p_weak", f"{arm}_base_p_clear"]].to_numpy(float), context=mc_train.loc[:, list(_arm_context(arm, "meta"))])
        design_oos = _build_meta_design(mo, names=meta.raw_feature_names, base_score=bs, base_p=bp, context=mc_oos.loc[:, list(_arm_context(arm, "meta"))])
        labels, fq3_state = _fit_direct_correctness(h.exact_net_bps.to_numpy(float), h[f"{arm}_base_direct_score"].to_numpy(float), score_domain=(-1.0, 1.0))
        mmodel = fitter(design_train, labels, "meta", base.normalized_side, arm, "direct_fq3")
        mp = _three_probability(mmodel, design_oos)
        correction, direct = _reconstruct_direct_correctness(mp, bs, fq3_state)
        out[f"{arm}_base_direct_score"], out[f"{arm}_meta_direct_score"], out[f"{arm}_meta_direct_correction"] = bs, direct, correction
        out[f"{arm}_base_p_adverse"], out[f"{arm}_base_p_weak"], out[f"{arm}_base_p_clear"] = bp[:, 0], bp[:, 1], bp[:, 2]
        out[f"{arm}_meta_p_error_tercile_0"], out[f"{arm}_meta_p_error_tercile_1"], out[f"{arm}_meta_p_error_tercile_2"] = mp[:, 0], mp[:, 1], mp[:, 2]
        final[arm]["base"] = {"state": bstate, "features": list(bnames), "model": bmodel}
        final[arm]["meta"] = {"state": {"drift_ood": mstate, "fq3": fq3_state.to_dict(), "meta_features": list(design_train.columns)}, "features": list(design_train.columns), "model": mmodel}
        audit.append({"side": base.normalized_side, "arm": arm, "selector_rows": len(b), "oos_rows": len(bo), "strictly_later_oos": True, "meta_training_rows_from_strict_base_oof": len(h), "meta_target": DIRECT_FQ3_SEMANTICS})
    out["base_strict_oof_available"], out["meta_strict_oof_available"] = True, True
    return out, final, audit


def _key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["candidate_key"] = out.side_name.astype(str) + "::" + out.candidate_id.astype(str)
    if out.candidate_key.duplicated().any():
        raise StageVExperimentError("joint Stage-V ledger has duplicate candidate identities")
    return out


def _metrics(frame: pd.DataFrame, *, population: str, admission: Causal21dAdmissionSpec, include_base_diagnostic: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    mapped_all: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for arm in STAGE_V_ARMS:
        if include_base_diagnostic:
            base = frame.loc[np.isfinite(pd.to_numeric(frame[f"{arm}_base_direct_score"], errors="coerce"))].copy()
            diagnostic = _selection_metrics(base, score_column=f"{arm}_base_direct_score", layer=f"base_diagnostic:{arm}", admission_mode="native_score_diagnostic_no_bps_mapping", requested_population_rows=len(base))
            diagnostic["population"], diagnostic["arm"], diagnostic["promotable"] = population, arm, False
            rows.append(diagnostic)
        meta = frame.loc[np.isfinite(pd.to_numeric(frame[f"{arm}_meta_direct_score"], errors="coerce"))].copy()
        raw = _selection_metrics(meta, score_column=f"{arm}_meta_direct_score", layer=f"joint_meta:{arm}", admission_mode="native_reconstructed_before_21d_mapping", requested_population_rows=len(meta))
        raw["population"], raw["arm"], raw["promotable"] = population, arm, True
        rows.append(raw)
        if meta.empty:
            continue
        mapping_input = meta.rename(columns={f"{arm}_meta_direct_score": "__native_joint_score__", "exact_net_bps": "net_bps"})
        mapped, audit = apply_causal_21d_side_admission(mapping_input, score_column="__native_joint_score__", net_column="net_bps", decision_column="decision_ts", label_available_column="label_available_ts", identity_column="candidate_key", spec=admission)
        mapped = mapped.rename(columns={"net_bps": "exact_net_bps"})
        mapped["mapped_expected_net_bps"] = mapped.causal_21d_side_expected_net_bps
        mapped["arm"], mapped["population"] = arm, population
        mapped_all.append(mapped)
        audit["arm"], audit["population"] = arm, population
        audits.append(audit)
        accepted = mapped.loc[mapped.causal_21d_side_admitted_ge_50bps.astype(bool)]
        final = _selection_metrics(accepted, score_column="mapped_expected_net_bps", layer=f"joint_meta:{arm}", admission_mode="with_side_local_causal_21d_admission_after_reconstruction", requested_population_rows=len(meta))
        final["population"], final["arm"], final["promotable"] = population, arm, True
        rows.append(final)
    return (pd.concat(rows, ignore_index=True, sort=False), pd.concat(mapped_all, ignore_index=True, sort=False), pd.concat(audits, ignore_index=True, sort=False))


def _winner(metrics: pd.DataFrame, config: StageVExperimentConfig) -> dict[str, Any]:
    common = (metrics.population.eq("strict_oof")
              & metrics.layer.astype(str).str.startswith("joint_meta:")
              & metrics.admission_mode.eq("with_side_local_causal_21d_admission_after_reconstruction")
              & np.isclose(pd.to_numeric(metrics.top_fraction, errors="coerce"), config.selection_top_fraction))
    pooled = metrics.loc[common & metrics.row_type.eq("pooled_global") & metrics.scope.eq("pooled_global")].copy()
    output = {
        "selection_population": "strict_oof",
        "selection": "matched joint reconstructed native FQ3 score -> side-local causal 21d common-bps mapping -> pooled-global top-k",
        "base_economics": "diagnostic_only_never_selectable",
        "promotion": "EXPERIMENT_WINNER_ONLY_NOT_POLICY_PROMOTION",
        "ood_advance_gate": {
            "strictly_beats_control_top10_net_bps": True,
            "same_candidate_population_rows": True,
            "minimum_selected_rows": int(config.min_selected_rows),
            "non_negative_worst_month_net_bps": True,
            "non_negative_latest_month_net_bps": True,
        },
    }
    if pooled.empty or "control" not in set(pooled.arm.astype(str)):
        return {**output, "decision": "NO_STAGE_V_OOD_ADVANCE_RETAIN_UPSTREAM", "winner_arm": None, "reason": "no_admitted_control_joint_meta_metrics"}
    control = pooled.loc[pooled.arm.eq("control")].iloc[0]
    candidates: list[dict[str, Any]] = []
    for arm in ("base_ood", "meta_ood", "both_ood"):
        arm_row = pooled.loc[pooled.arm.eq(arm)]
        if arm_row.empty:
            candidates.append({"arm": arm, "advance": False, "reason": "no_admitted_joint_meta_metrics"})
            continue
        row = arm_row.iloc[0]
        # The strict OOF master is arm-matched before mapping.  Persist the
        # equality gate so unequal-support candidates cannot win by attrition.
        same_population = int(row.candidate_rows) == int(control.candidate_rows)
        contributions = metrics.loc[common & metrics.arm.eq(arm) & metrics.row_type.eq("selected_contribution") & metrics.scope.eq("month")].copy()
        if contributions.empty:
            worst_month = latest_month = np.nan
        else:
            worst_month = float(pd.to_numeric(contributions.realised_net_bps_per_trade, errors="coerce").min())
            months = contributions.sort_values("period_key", kind="stable")
            latest_month = float(pd.to_numeric(months.iloc[-1].realised_net_bps_per_trade, errors="coerce"))
        lift = float(row.realised_net_bps_per_trade) - float(control.realised_net_bps_per_trade)
        advance = bool(
            same_population and lift > 0.0 and int(row.selected_rows) >= int(config.min_selected_rows)
            and np.isfinite(worst_month) and worst_month >= 0.0
            and np.isfinite(latest_month) and latest_month >= 0.0
        )
        candidates.append({"arm": arm, "advance": advance, "top10_net_lift_bps": lift, "selected_rows": int(row.selected_rows), "same_candidate_population": same_population, "worst_month_net_bps": worst_month, "latest_month_net_bps": latest_month})
    advancing = [item for item in candidates if item["advance"]]
    if not advancing:
        return {**output, "decision": "NO_STAGE_V_OOD_ADVANCE_RETAIN_UPSTREAM", "winner_arm": None, "control_top10_net_bps": float(control.realised_net_bps_per_trade), "candidates": candidates}
    selected = sorted(advancing, key=lambda item: (-float(item["top10_net_lift_bps"]), item["arm"]))[0]
    return {**output, "decision": "STAGE_V_OOD_ARM_ADVANCES", "winner_arm": selected["arm"], "control_top10_net_bps": float(control.realised_net_bps_per_trade), "selected_net_bps_per_trade": float(control.realised_net_bps_per_trade) + float(selected["top10_net_lift_bps"]), "selected_rows": int(selected["selected_rows"]), "candidates": candidates}


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _ordered_feature_sha(names: Sequence[str]) -> str:
    return sha256(json.dumps(list(map(str, names)), separators=(",", ":")).encode()).hexdigest()


def _none_contract(source: StageVLayerSource) -> dict[str, Any]:
    """A valid no-context contract has no fitted state by construction."""
    raw = tuple(source.raw_feature_names)
    raw_sha = _ordered_feature_sha(raw)
    return {
        "side": source.normalized_side, "layer": source.normalized_layer,
        "controller": "none", "raw_feature_names": list(raw), "context_feature_names": [],
        "state_sha256": _digest({"controller": "none", "side": source.normalized_side, "layer": source.normalized_layer, "state": "not_requested"}),
        "ordered_model_feature_names": list(raw), "source_feature_contract_sha256": raw_sha,
        "model_feature_contract_sha256": raw_sha, "state_fit": "not_requested",
    }


def _write_models(root: Path, states: Mapping[str, Any], *, mode: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for side, arms in states.items():
        for arm, layers in arms.items():
            for layer, records in layers.items():
                items = records if isinstance(records, list) else [records]
                for pos, record in enumerate(items):
                    if record.get("model") is None:
                        continue
                    suffix = f"fold_{pos:02d}" if isinstance(records, list) else "final"
                    state = record["state"]
                    sp = root / "states" / mode / side / arm / layer / f"{suffix}.json"
                    mp = root / "models" / mode / side / arm / layer / f"{suffix}.pkl"
                    _dump(sp, state)
                    mp.parent.mkdir(parents=True, exist_ok=True)
                    with mp.open("wb") as handle:
                        pickle.dump(record["model"], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    hashes[str(sp.relative_to(root))] = _sha(sp)
                    hashes[str(mp.relative_to(root))] = _sha(mp)
    return hashes


def run_stage_v_drift_ood_ablation(*, sources: Sequence[StageVLayerSource], output_dir: str | Path, config: StageVExperimentConfig = StageVExperimentConfig(), fitter: ModelFitter = default_stage_v_model_fitter) -> Mapping[str, Any]:
    """Run matched joint controller arms without touching another artifact."""
    config.validate()
    for source in sources:
        source.validate()
    pairs = _paired_sources(sources)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-V artifact: {output}")
    oof_parts: list[pd.DataFrame] = []
    oos_parts: list[pd.DataFrame] = []
    fold_audit: list[dict[str, Any]] = []
    final_audit: list[dict[str, Any]] = []
    strict_states: dict[str, Any] = {}
    frozen_states: dict[str, Any] = {}
    contracts: dict[str, Any] = {}
    for side, (base, meta) in pairs.items():
        strict, state, audit = _strict_side(base, meta, config=config, fitter=fitter)
        frozen, final, f_audit = _frozen_side(base, meta, strict, config=config, fitter=fitter)
        oof_parts.append(_key(strict)); oos_parts.append(_key(frozen))
        strict_states[side], frozen_states[side] = state, final
        fold_audit.extend(audit); final_audit.extend(f_audit)
        for arm in STAGE_V_ARMS:
            for layer, source in (("base", base), ("meta", meta)):
                final_record = final[arm][layer]
                final_state = final_record["state"]
                key = f"{side}:{arm}:{layer}"
                if not _arm_context(arm, layer):
                    contracts[key] = _none_contract(source)
                elif final_record.get("model") is None:
                    contracts[key] = {
                        **_none_contract(source), "controller": "unavailable",
                        "availability": "no_positive_frozen_mda_group", "reason": final_state.get("reason", "context_unavailable"),
                    }
                else:
                    drift_state = final_state if layer == "base" else final_state["drift_ood"]
                    contracts[key] = freeze_stage_v_feature_contract(contract=StageVContract(side, layer), raw_feature_names=source.raw_feature_names, state=drift_state, controller="grouped_ood").to_dict()
                if layer == "meta":
                    contracts[key]["fq3_features"] = final_record["features"]
                    contracts[key]["forbids_pre_mapped_expected_net"] = True
    oof, oos = pd.concat(oof_parts, ignore_index=True), pd.concat(oos_parts, ignore_index=True)
    oof_metrics, oof_map, oof_audit = _metrics(oof, population="strict_oof", admission=config.admission_spec, include_base_diagnostic=True)
    # For later OOS, only strict OOF history plus prior resolved OOS rows may
    # construct the 21-day map.  Base diagnostics never map.
    combined = pd.concat([oof, oos], ignore_index=True, sort=False)
    if combined.candidate_key.duplicated().any():
        collisions = combined.loc[combined.candidate_key.duplicated(keep=False), "candidate_key"].astype(str).head(4).tolist()
        raise StageVExperimentError(
            "strict-OOF/later-OOS canonical identity collision; causal-map identities must be unique side-qualified IDs: "
            f"{collisions}"
        )
    _, combined_map, combined_audit = _metrics(combined, population="history_plus_frozen_oos", admission=config.admission_spec, include_base_diagnostic=False)
    public_rows: list[pd.DataFrame] = []
    for arm in STAGE_V_ARMS:
        raw = _selection_metrics(oos, score_column=f"{arm}_meta_direct_score", layer=f"joint_meta:{arm}", admission_mode="native_reconstructed_before_21d_mapping", requested_population_rows=len(oos))
        raw["population"], raw["arm"], raw["promotable"] = "frozen_oos", arm, True
        public_rows.append(raw)
        mapped = combined_map.loc[(combined_map.arm == arm) & combined_map.candidate_key.isin(set(oos.candidate_key))]
        accepted = mapped.loc[mapped.causal_21d_side_admitted_ge_50bps.astype(bool)]
        admitted = _selection_metrics(accepted, score_column="mapped_expected_net_bps", layer=f"joint_meta:{arm}", admission_mode="with_side_local_causal_21d_admission_after_reconstruction", requested_population_rows=len(oos))
        admitted["population"], admitted["arm"], admitted["promotable"] = "frozen_oos", arm, True
        public_rows.append(admitted)
    metrics = pd.concat([oof_metrics, *public_rows], ignore_index=True, sort=False)
    winner = _winner(oof_metrics, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)); artifact = temp_parent / output.name
    try:
        artifact.mkdir()
        oof.to_parquet(artifact / "joint_strict_oof_predictions.parquet", index=False, compression="zstd")
        oos.to_parquet(artifact / "joint_frozen_oos_predictions.parquet", index=False, compression="zstd")
        metrics.to_parquet(artifact / "per_side_month_joint_meta_21d_metrics.parquet", index=False, compression="zstd")
        pd.DataFrame(fold_audit).to_parquet(artifact / "strict_oof_fold_audit.parquet", index=False, compression="zstd")
        pd.DataFrame(final_audit).to_parquet(artifact / "frozen_oos_fit_audit.parquet", index=False, compression="zstd")
        oof_map.to_parquet(artifact / "joint_strict_oof_with_causal_21d_admission.parquet", index=False, compression="zstd")
        combined_map.loc[combined_map.candidate_key.isin(set(oos.candidate_key))].to_parquet(artifact / "joint_frozen_oos_with_causal_21d_admission.parquet", index=False, compression="zstd")
        pd.concat([oof_audit, combined_audit], ignore_index=True).to_parquet(artifact / "causal_21d_admission_audit.parquet", index=False, compression="zstd")
        _dump(artifact / "joint_feature_contracts.json", contracts); _dump(artifact / "winner.json", winner)
        files = {**_write_models(artifact, strict_states, mode="strict_oof"), **_write_models(artifact, frozen_states, mode="frozen_oos")}
        lineage = {f"{source.normalized_layer}:{source.normalized_side}": {"selector_manifest_sha256": source.selector_manifest_sha256, "raw_feature_names": list(source.raw_feature_names), "mda_group_audit_sha256": _digest(source.mda_group_audit.to_dict(orient="records")), "oos_surface_lineage": dict(source.oos_surface_lineage), "target_column": source.target_column} for source in sources}
        manifest = {"schema": STAGE_V_EXPERIMENT_SCHEMA, "status": "complete", "stage_v_context_schema": STAGE_V_SCHEMA, "config": {**asdict(config), "admission_spec": asdict(config.admission_spec)}, "arms": list(STAGE_V_ARMS), "architecture": "native_same_side_R3_base_to_direct_fold_local_FQ3_then_reconstructed_native_score_then_causal_21d_common_bps_mapping", "direct_fq3": {"semantics": DIRECT_FQ3_SEMANTICS, "base_input": DIRECT_BASE_INPUT_SEMANTICS, "forbidden_pre_map_features": list(_FORBIDDEN_FQ3)}, "ranking": "pooled global only after reconstructed meta score is causally mapped to common bps; never per timestamp", "base_economics": "diagnostic only; never selected or promoted", "strictness": {"base_and_meta_fold_training": "label_available_ts < validation decision start; meta training also requires strict base OOF handoff", "final_meta_training": "strict base OOF selector handoff only", "frozen_oos": "selector ends strictly before later OOS", "oos_mapping": "strict OOF history plus prior-resolved OOS only"}, "source_lineage": lineage, "feature_contracts": contracts, "state_files_sha256": files, "winner": winner, "promotion": "experiment winner only; independent replay still required", "files": {}}
        for path in artifact.rglob("*"):
            if path.is_file() and path.name != "run_manifest.json": manifest["files"][str(path.relative_to(artifact))] = _sha(path)
        _dump(artifact / "run_manifest.json", manifest)
        os.replace(artifact, output)
        return manifest
    except Exception:
        shutil.rmtree(temp_parent, ignore_errors=True)
        raise
    finally:
        if temp_parent.exists(): temp_parent.rmdir()


__all__ = ["STAGE_V_EXPERIMENT_SCHEMA", "STAGE_V_ARMS", "StageVExperimentError", "StageVLayerSource", "StageVExperimentConfig", "ModelFitter", "default_stage_v_model_fitter", "run_stage_v_drift_ood_ablation"]
