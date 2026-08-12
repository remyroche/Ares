"""Runnable Stage-II bridge for the frozen direct-base/FQ3 stack.

Stage II is allowed to add *causally recognised* path-archetype context to the
meta layer.  It is not allowed to turn the direct base score into bps before
the meta fit.  This module is deliberately separate from the older
``stage_ii_execution`` Huber-residual route, whose bps residual target has
different semantics.

The bridge has two entry points:

``run_stage_ii_direct_fq3_archetype_funnel``
    strict-OOF development comparison of no-archetype versus soft-archetype
    context.  The meta target is the fold-local three-class FQ3 correction.

``score_frozen_stage_ii_direct_fq3``
    one-shot later-period scoring with a frozen configuration.  It emits a
    Stage-III-ready ledger only after the direct meta score has been mapped by
    the causal side-local 21-day common-bps map.

There are no side or regime experts, no hard routing, and no timestamp-local
ranking.  The archetype residual prior is intentionally diagnostic-only here:
it is in bps and therefore cannot be fed into the native-score FQ3 model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .lgbm_pipeline import _fit_lgbm_model
from .stage_i_causal_admission import Causal21dAdmissionSpec, apply_causal_21d_side_admission
from .stage_i_strict_oof import _multiclass_probabilities, _validation_blocks
from .stage_i_target_specific_oos import (
    DIRECT_BASE_INPUT_SEMANTICS,
    DIRECT_FQ3_SEMANTICS,
    _direct_trust,
    _fit_direct_correctness,
    _reconstruct_direct_correctness,
)
from .stage_ii_meta_archetypes import (
    META_ARCHETYPE_PREFIX,
    StageIIMetaArchetypeConfig,
    SideLocalMetaArchetypeState,
    membership_feature_names,
    strict_oof_meta_archetype_features,
)


SCHEMA = "stage_ii_direct_fq3_archetype_bridge_v1"
JOINT_EXPECTED_NET_COLUMN = "prequential_joint_expected_net_bps"
JOINT_MAPPING_SEMANTICS = "direct_fq3_reconstructed_causal_21d_common_bps_v1"
LEGACY_BASE_ALIAS_SEMANTICS = "deprecated_compatibility_alias_of_prequential_joint_expected_net_bps"
_SIDES = ("long", "short")
_ARMS = ("none", "soft_memberships")
_TRUST = ("base_output_entropy", "base_output_top2_margin", "base_output_max_probability")
_BASE_STATE_RE = re.compile(r"^base_state_p(\d+)$")
_SUPPORTED_BASE_STATE_WIDTHS = frozenset((2, 3, 5))
_HANDOFF_IDENTITY = ("candidate_id", "side_name", "symbol", "signal_close_ts")
_LEGACY_R3_STATE_COLUMNS = ("r3_p_adverse", "r3_p_weak", "r3_p_clear")


class StageIIDirectFQ3Error(ValueError):
    """The direct native-score Stage-II contract was violated."""


def _canonical_input_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Return exact Stage-I identity with both historical alias spellings.

    Stage-I input panels commonly use ``__symbol__``/``__ts__`` while the
    direct-FQ3 ledger uses ``symbol``/``signal_close_ts``.  Treat those as
    aliases *only* after proving that any simultaneously supplied pair agrees.
    This deliberately avoids a positional or candidate-id-only merge.
    """
    work = frame.copy()
    if "candidate_id" not in work or "side_name" not in work:
        raise StageIIDirectFQ3Error(f"{source} lacks candidate_id/side_name identity")
    for canonical, alias, temporal in (
        ("symbol", "__symbol__", False),
        ("signal_close_ts", "__ts__", True),
    ):
        has_canonical, has_alias = canonical in work, alias in work
        if not has_canonical and not has_alias:
            raise StageIIDirectFQ3Error(f"{source} lacks {canonical!r} (or alias {alias!r})")
        if temporal:
            canonical_value = pd.to_datetime(work[canonical], utc=True, errors="coerce") if has_canonical else None
            alias_value = pd.to_datetime(work[alias], utc=True, errors="coerce") if has_alias else None
            if (canonical_value is not None and canonical_value.isna().any()) or (alias_value is not None and alias_value.isna().any()):
                raise StageIIDirectFQ3Error(f"{source} has invalid {canonical!r} identity timestamps")
            if canonical_value is not None and alias_value is not None and not canonical_value.equals(alias_value):
                raise StageIIDirectFQ3Error(f"{source} has conflicting {canonical!r}/{alias!r} aliases")
            value = canonical_value if canonical_value is not None else alias_value
        else:
            canonical_value = work[canonical].astype("string").str.strip() if has_canonical else None
            alias_value = work[alias].astype("string").str.strip() if has_alias else None
            if (canonical_value is not None and (canonical_value.isna().any() or canonical_value.eq("").any())) or (
                alias_value is not None and (alias_value.isna().any() or alias_value.eq("").any())
            ):
                raise StageIIDirectFQ3Error(f"{source} has invalid {canonical!r} identity values")
            if canonical_value is not None and alias_value is not None and not canonical_value.equals(alias_value):
                raise StageIIDirectFQ3Error(f"{source} has conflicting {canonical!r}/{alias!r} aliases")
            value = canonical_value if canonical_value is not None else alias_value
        work[canonical] = value
        work[alias] = value
    work["candidate_id"] = work.candidate_id.astype("string").str.strip()
    work["side_name"] = work.side_name.astype(str).str.lower().str.strip()
    if work.candidate_id.isna().any() or work.candidate_id.eq("").any() or not work.side_name.isin(_SIDES).all():
        raise StageIIDirectFQ3Error(f"{source} has invalid candidate_id/side_name identity values")
    if work.duplicated(list(_HANDOFF_IDENTITY)).any():
        raise StageIIDirectFQ3Error(f"{source} has duplicate exact Stage-I identities")
    return work


def _content_sha256(frame: pd.DataFrame) -> str:
    """Stable in-memory content digest used when an input has no file path."""
    canonical = frame.copy()
    for name in canonical.columns:
        if pd.api.types.is_datetime64_any_dtype(canonical[name]):
            canonical[name] = pd.to_datetime(canonical[name], utc=True).astype("string")
    values = pd.util.hash_pandas_object(canonical, index=False, categorize=True).to_numpy(np.uint64)
    names = "\x1f".join(map(str, canonical.columns)).encode("utf-8")
    return sha256(names + values.tobytes()).hexdigest()


def _state_contract_from_handoff(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Canonicalise native Stage-I state probabilities without changing width."""
    work = frame.copy()
    native = _base_state_columns(work) if any(_BASE_STATE_RE.fullmatch(str(name)) for name in work.columns) else ()
    if not native and all(name in work for name in _LEGACY_R3_STATE_COLUMNS):
        for index, name in enumerate(_LEGACY_R3_STATE_COLUMNS):
            work[f"base_state_p{index}"] = pd.to_numeric(work[name], errors="coerce")
        native = tuple(f"base_state_p{index}" for index in range(3))
    if not native:
        raise StageIIDirectFQ3Error("joint OOF ledger lacks a native 2-, 3-, or 5-state base handoff")
    # _base_state_columns performs the contiguous-width validation after a
    # legacy R3 alias has been normalised.
    return work, _base_state_columns(work)


def _ledger_column(work: pd.DataFrame, canonical: str, aliases: Sequence[str]) -> pd.Series:
    """Read one ledger field with a strict, explicit compatibility alias."""
    names = (canonical, *tuple(aliases))
    present = [name for name in names if name in work]
    if not present:
        raise StageIIDirectFQ3Error(f"joint OOF ledger lacks required field {canonical!r}")
    value = work[present[0]]
    for name in present[1:]:
        left = pd.to_numeric(value, errors="coerce").to_numpy(float)
        right = pd.to_numeric(work[name], errors="coerce").to_numpy(float)
        if not np.allclose(left, right, equal_nan=False):
            raise StageIIDirectFQ3Error(f"joint OOF ledger has conflicting aliases for {canonical!r}")
    return value


def materialize_stage_ii_direct_fq3_handoff(
    output_directory: str | Path,
    *,
    frozen_stage_i_input_panel: pd.DataFrame,
    joint_oof_ledger: pd.DataFrame,
    selected_causal_feature_cols: Sequence[str],
    selected_meta_feature_cols: Sequence[str],
    frozen_input_panel_path: str | Path | None = None,
    joint_oof_ledger_path: str | Path | None = None,
) -> Path:
    """Publish the identity-bound Stage-I input needed by the Stage-II bridge.

    Only selected source-causal fields are read from the frozen Stage-I panel.
    Native base states, the direct score, labels and strict-OOF evidence come
    from the joint OOF ledger.  Trust is *always* derived afresh from the
    ledger's native simplex; an input-panel value with the same name can never
    become a stale or leaked model handoff.  The materialized ledger supports
    the scalar (2 state), R3 (3 state) and ordinal (5 state) contracts without
    relabelling their state meanings.
    """
    causal = tuple(dict.fromkeys(map(str, selected_causal_feature_cols)))
    meta = tuple(dict.fromkeys(map(str, selected_meta_feature_cols)))
    if not causal or not meta:
        raise StageIIDirectFQ3Error("Stage-II handoff needs non-empty selected causal and meta feature contracts")
    panel = _canonical_input_identity(frozen_stage_i_input_panel, source="frozen Stage-I input panel")
    ledger = _canonical_input_identity(joint_oof_ledger, source="joint OOF ledger")
    panel_index = pd.MultiIndex.from_frame(panel.loc[:, list(_HANDOFF_IDENTITY)])
    ledger_index = pd.MultiIndex.from_frame(ledger.loc[:, list(_HANDOFF_IDENTITY)])
    if not panel_index.is_unique or not ledger_index.is_unique:  # defensive; aliases were already checked
        raise StageIIDirectFQ3Error("Stage-II handoff identity must be one-to-one")
    panel_only, ledger_only = panel_index.difference(ledger_index), ledger_index.difference(panel_index)
    if len(panel_only) or len(ledger_only):
        raise StageIIDirectFQ3Error(
            "frozen Stage-I input panel and joint OOF ledger have non-identical exact identities "
            f"(panel_only={len(panel_only)}, ledger_only={len(ledger_only)})"
        )
    ledger, state_columns = _state_contract_from_handoff(ledger)
    # These values are label/OOF evidence, not source-panel features.  The
    # aliases cover the Stage-I target-specific writer while preserving one
    # canonical Stage-II schema.
    output = ledger.loc[:, list(_HANDOFF_IDENTITY)].copy()
    output["__symbol__"] = output.symbol.astype("string")
    output["__ts__"] = pd.to_datetime(output.signal_close_ts, utc=True)
    for name, aliases in (
        ("decision_ts", ()),
        ("label_available_ts", ()),
        ("exact_gross_bps", ("gross_bps",)),
        ("exact_net_bps", ("net_bps",)),
        ("base_direct_score", ("base_raw_score", "r3_opportunity_score")),
    ):
        output[name] = _ledger_column(ledger, name, aliases).to_numpy()
    if "base_strict_oof_available" in ledger:
        output["base_strict_oof_available"] = _bool(ledger.base_strict_oof_available, name="base_strict_oof_available")
    elif "base_oof_fold_id" in ledger:
        output["base_strict_oof_available"] = pd.to_numeric(ledger.base_oof_fold_id, errors="coerce").to_numpy(float) >= 0
    else:
        raise StageIIDirectFQ3Error("joint OOF ledger lacks strict base-OOF availability evidence")
    for name in state_columns:
        output[name] = pd.to_numeric(ledger[name], errors="coerce").to_numpy()
    simplex = output.loc[:, list(state_columns)].to_numpy(float)
    trust = _direct_trust(simplex)
    for name in _TRUST:
        output[name] = trust[name].to_numpy(np.float32)

    generated = {"base_direct_score", "base_raw_score", *state_columns, *_TRUST}
    selected_source = tuple(dict.fromkeys((*causal, *meta)))
    panel_features = tuple(name for name in selected_source if name not in generated)
    missing = sorted(set(panel_features).difference(panel.columns))
    if missing:
        raise StageIIDirectFQ3Error(f"frozen Stage-I panel lacks selected causal/meta fields: {missing[:12]}")
    # Reindex by the exact four-field identity, retaining the ledger order used
    # by its strict OOF lineage.  No positional join is ever permitted.
    panel_values = panel.set_index(list(_HANDOFF_IDENTITY)).loc[ledger_index, list(panel_features)].reset_index(drop=True)
    for name in panel_features:
        output[name] = panel_values[name].to_numpy()

    # A direct base handoff may have been selected as ``base_raw_score``.  The
    # validator constructs the same protected alias, but persisting it here
    # makes the handoff self-contained for the Stage-II bridge and its audit.
    output["base_raw_score"] = pd.to_numeric(output.base_direct_score, errors="coerce").to_numpy()
    # Preserve selected order in the manifest; validation below is the final
    # finite/entry/label/cost contract gate before publication.
    required_meta = tuple(dict.fromkeys((*_base_features(state_columns), *meta)))
    if missing := sorted(set(required_meta).difference(output.columns)):
        raise StageIIDirectFQ3Error(f"materialized handoff lacks selected direct/meta fields: {missing[:12]}")
    if output.duplicated(list(_HANDOFF_IDENTITY)).any():
        raise StageIIDirectFQ3Error("materialized Stage-II handoff has duplicate exact identity")

    root = Path(output_directory).resolve()
    if root.exists() or not root.parent.is_dir():
        raise StageIIDirectFQ3Error("Stage-II handoff output must be a new directory under an existing parent")
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        handoff_path = temporary / "stage_ii_direct_fq3_input.parquet"
        output.to_parquet(handoff_path, index=False, compression="zstd")
        panel_source = Path(frozen_input_panel_path).resolve() if frozen_input_panel_path is not None else None
        ledger_source = Path(joint_oof_ledger_path).resolve() if joint_oof_ledger_path is not None else None
        if panel_source is not None and not panel_source.is_file():
            raise StageIIDirectFQ3Error("declared frozen Stage-I panel path is not a file")
        if ledger_source is not None and not ledger_source.is_file():
            raise StageIIDirectFQ3Error("declared joint OOF ledger path is not a file")
        manifest = {
            "schema": "stage_ii_direct_fq3_handoff_v1",
            "status": "complete",
            "rows": int(len(output)),
            "identity": list(_HANDOFF_IDENTITY),
            "identity_join": "exact_one_to_one_candidate_id_side_symbol_signal_close_ts",
            "identity_aliases": {"__symbol__": "symbol", "__ts__": "signal_close_ts"},
            "base_state_columns": list(state_columns),
            "base_state_width": int(len(state_columns)),
            "base_state_contract": "native_simplex_preserved_without_conversion",
            "selected_causal_feature_cols": list(causal),
            "selected_meta_feature_cols": list(meta),
            "panel_carried_feature_cols": list(panel_features),
            "derived_trust_feature_cols": list(_TRUST),
            "trust_lineage": "derived_from_joint_oof_native_base_simplex_not_input_panel",
            "joint_oof_lineage": "base_score_states_labels_and_strict_oof_evidence_from_joint_ledger",
            "frozen_input_panel": {
                "path": str(panel_source) if panel_source is not None else None,
                "sha256": _file_sha256(panel_source) if panel_source is not None else _content_sha256(panel),
            },
            "joint_oof_ledger": {
                "path": str(ledger_source) if ledger_source is not None else None,
                "sha256": _file_sha256(ledger_source) if ledger_source is not None else _content_sha256(ledger),
            },
            "handoff_content_sha256": _content_sha256(output),
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        checksums = {name: _file_sha256(temporary / name) for name in (handoff_path.name, manifest_path.name)}
        (temporary / "checksums.json").write_text(json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return root


@dataclass(frozen=True)
class StageIIDirectFQ3Spec:
    """Frozen native-score meta and reporting contract.

    ``meta_feature_cols`` must be selected under the meta feature key.  The
    bridge adds its own required direct base/state/trust handoff; callers may
    not substitute a mapped bps field for it.
    """

    meta_feature_cols: tuple[str, ...]
    model_params: Mapping[str, Any]
    score_domain: tuple[float, float] = (-1.0, 1.0)
    n_validation_folds: int = 4
    min_train_rows: int = 500
    top_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    admission_spec: Causal21dAdmissionSpec = Causal21dAdmissionSpec()
    require_admission_for_selection: bool = True

    def validate(self) -> None:
        names = tuple(dict.fromkeys(map(str, self.meta_feature_cols)))
        if not names or names != tuple(self.meta_feature_cols):
            raise StageIIDirectFQ3Error("meta_feature_cols must be a unique non-empty ordered tuple")
        if "prequential_base_expected_net_bps" in names or any("expected_net" in x or "mapped" in x for x in names):
            raise StageIIDirectFQ3Error("direct FQ3 meta may not receive mapped/common-bps features")
        if not isinstance(self.model_params, Mapping):
            raise StageIIDirectFQ3Error("model_params must be a frozen mapping")
        lower, upper = (float(x) for x in self.score_domain)
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise StageIIDirectFQ3Error("score_domain must be finite and ordered")
        if int(self.n_validation_folds) < 1 or int(self.min_train_rows) < 3:
            raise StageIIDirectFQ3Error("strict OOF requires positive folds and >=3 prior rows")
        if self.admission_spec.window_days != 21:
            raise StageIIDirectFQ3Error("Stage II direct bridge requires the canonical 21-day admission map")
        if not self.top_fractions or any(not 0 < float(x) <= 1 for x in self.top_fractions):
            raise StageIIDirectFQ3Error("top fractions must lie in (0,1]")


@dataclass(frozen=True)
class StageIIDirectFQ3Candidate:
    candidate_id: str
    archetype_config: StageIIMetaArchetypeConfig
    causal_feature_cols: tuple[str, ...]


@dataclass(frozen=True)
class StageIIDirectFQ3ArmResult:
    candidate_id: str
    arm: str
    feature_names: tuple[str, ...]
    oof_predictions: pd.DataFrame
    metrics: pd.DataFrame
    contributions: pd.DataFrame
    admission_audit: pd.DataFrame
    fold_audit: pd.DataFrame


@dataclass(frozen=True)
class StageIIDirectFQ3Result:
    candidate_audit: pd.DataFrame
    arms: tuple[StageIIDirectFQ3ArmResult, ...]
    selected_candidate_id: str | None
    selected_arm: str | None
    selected_features: tuple[str, ...]
    selected_oof_predictions: pd.DataFrame | None
    archetype_oof_features: pd.DataFrame | None
    manifest: Mapping[str, Any]


def _utc(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        raise StageIIDirectFQ3Error(f"missing required timestamp {name!r}")
    value = pd.to_datetime(frame[name], utc=True, errors="coerce")
    if value.isna().any():
        raise StageIIDirectFQ3Error(f"{name!r} contains invalid timestamps")
    return value


def _bool(series: pd.Series, *, name: str) -> np.ndarray:
    result: list[bool] = []
    for value in series.to_numpy(dtype=object):
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            result.append(bool(value))
        else:
            raise StageIIDirectFQ3Error(f"{name} must use canonical booleans/0/1")
    return np.asarray(result, dtype=bool)


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        raise StageIIDirectFQ3Error(f"missing required field {name!r}")
    value = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
    if not np.isfinite(value).all():
        raise StageIIDirectFQ3Error(f"{name!r} must be finite")
    return value


def _base_state_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the frozen native base simplex columns in canonical order.

    Stage-I uses a two-state scalar target, three-state R3 target, or
    five-state ordinal target.  They all enter the direct FQ3 correction in
    their native probability form: neither the dimensionality nor a bps map
    is inferred from the score.  Reject sparse/unknown contracts rather than
    silently dropping a state column.
    """
    indexed: list[tuple[int, str]] = []
    for name in frame.columns:
        match = _BASE_STATE_RE.fullmatch(str(name))
        if match:
            indexed.append((int(match.group(1)), str(name)))
    indexed.sort(key=lambda item: item[0])
    expected = list(range(len(indexed)))
    actual = [index for index, _ in indexed]
    if actual != expected or len(indexed) not in _SUPPORTED_BASE_STATE_WIDTHS:
        available = [name for _, name in indexed]
        raise StageIIDirectFQ3Error(
            "direct base state handoff must be a contiguous 2-, 3-, or 5-state "
            f"simplex (found {available})"
        )
    return tuple(name for _, name in indexed)


def _base_features(state_columns: Sequence[str]) -> tuple[str, ...]:
    return ("base_raw_score", *tuple(state_columns), *_TRUST)


def _side_candidate_key(frame: pd.DataFrame) -> pd.Series:
    """Canonical map/join identity; raw candidate ids are not cross-side keys."""
    if "candidate_id" not in frame or "side_name" not in frame:
        raise StageIIDirectFQ3Error("side-qualified identity needs candidate_id and side_name")
    side = frame.side_name.astype(str).str.lower().str.strip()
    candidate = frame.candidate_id.astype("string").str.strip()
    if side.isin(_SIDES).eq(False).any() or candidate.isna().any() or candidate.eq("").any():
        raise StageIIDirectFQ3Error("side-qualified identity contains an invalid side or candidate id")
    return (side + "::" + candidate).astype(str)


def validate_stage_ii_direct_fq3_ledger(frame: pd.DataFrame, *, spec: StageIIDirectFQ3Spec) -> pd.DataFrame:
    """Validate and canonicalise the Stage-I direct joint handoff.

    A bps map is specifically prohibited in the *meta input* contract.  The
    exact net label remains available for fitting and later causal mapping.
    """
    spec.validate()
    state_columns = _base_state_columns(frame)
    required = {
        "candidate_id", "symbol", "side_name", "signal_close_ts", "decision_ts", "label_available_ts",
        "exact_gross_bps", "exact_net_bps", "base_direct_score", "base_strict_oof_available",
        *state_columns, *_TRUST, *spec.meta_feature_cols,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise StageIIDirectFQ3Error(f"direct Stage-I handoff lacks fields: {missing[:12]}")
    if "prequential_base_expected_net_bps" in frame.columns:
        # It may be present in a wide source ledger, but must not appear in the
        # declared model feature contract.  This catches an accidental legacy
        # bridge while retaining benign output-only source columns.
        if "prequential_base_expected_net_bps" in spec.meta_feature_cols:
            raise StageIIDirectFQ3Error("mapped base bps was placed in the FQ3 meta contract")
    work = frame.copy()
    # Stage-I emits the frozen native value under ``base_direct_score``.  The
    # FQ3 fitter uses ``base_raw_score`` as its protected generic handoff name;
    # manufacture that alias here and reject a conflicting source alias.
    if "base_raw_score" in work and not np.allclose(
        pd.to_numeric(work.base_raw_score, errors="coerce").to_numpy(float),
        pd.to_numeric(work.base_direct_score, errors="coerce").to_numpy(float), equal_nan=False,
    ):
        raise StageIIDirectFQ3Error("base_raw_score conflicts with the frozen native base_direct_score")
    work["base_raw_score"] = pd.to_numeric(work.base_direct_score, errors="coerce")
    work["decision_ts"] = _utc(work, "decision_ts")
    work["signal_close_ts"] = _utc(work, "signal_close_ts")
    work["label_available_ts"] = _utc(work, "label_available_ts")
    if not np.allclose((work.decision_ts - work.signal_close_ts).dt.total_seconds() / 3600.0, 1.0):
        raise StageIIDirectFQ3Error("Stage-II direct input must enter one hour after signal close")
    if not np.allclose((work.label_available_ts - work.signal_close_ts).dt.total_seconds() / 3600.0, 13.0):
        raise StageIIDirectFQ3Error("Stage-II direct input labels must be available close+13h")
    work["side_name"] = work.side_name.astype(str).str.lower()
    if not work.side_name.isin(_SIDES).all():
        raise StageIIDirectFQ3Error("Stage-II direct input requires long and short rows only")
    if work.duplicated(["candidate_id", "symbol", "decision_ts", "side_name"]).any():
        raise StageIIDirectFQ3Error("direct input identity is duplicated")
    work["__stage_ii_side_candidate_key__"] = _side_candidate_key(work)
    if work["__stage_ii_side_candidate_key__"].duplicated().any():
        raise StageIIDirectFQ3Error(
            "candidate_id must be unique within side; raw ids may overlap only across sides"
        )
    gross, net = _numeric(work, "exact_gross_bps"), _numeric(work, "exact_net_bps")
    if not np.allclose(gross - 100.0, net, rtol=0.0, atol=1e-3):
        raise StageIIDirectFQ3Error("direct input must apply the 100bps cost exactly once")
    raw = _numeric(work, "base_raw_score")
    lower, upper = spec.score_domain
    if ((raw < lower - 1e-6) | (raw > upper + 1e-6)).any():
        raise StageIIDirectFQ3Error("base_raw_score is outside its declared native score domain")
    states = work.loc[:, list(state_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(states).all() or (states < 0).any() or not np.allclose(states.sum(axis=1), 1.0, atol=1e-6):
        raise StageIIDirectFQ3Error("direct base state handoff must be a finite native simplex")
    if not _bool(work.base_strict_oof_available, name="base_strict_oof_available").all():
        raise StageIIDirectFQ3Error("every Stage-II direct row must have a strict same-side base OOF score")
    model_features = tuple(dict.fromkeys((*_base_features(state_columns), *spec.meta_feature_cols)))
    values = work.loc[:, list(model_features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise StageIIDirectFQ3Error("all selected direct meta features must be finite")
    return work.sort_values(["decision_ts", "side_name", "candidate_id"], kind="stable").reset_index(drop=True)


def _direct_fit(frame: pd.DataFrame, labels: np.ndarray, weight: np.ndarray | None, *, params: Mapping[str, Any]) -> Any:
    return _fit_lgbm_model(
        frame, labels, weight, classifier=True,
        params={**dict(params), "objective": "multiclass", "num_class": 3},
        objective_mode="stage_ii_direct_fq3_meta_archetype",
    )


def _strict_direct_predictions(
    frame: pd.DataFrame, *, feature_names: Sequence[str], spec: StageIIDirectFQ3Spec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the direct FQ3 correction side-by-side from prior-resolved labels."""
    output = frame.loc[:, ["candidate_id", "symbol", "signal_close_ts", "decision_ts", "label_available_ts", "side_name", "exact_gross_bps", "exact_net_bps", "base_raw_score"]].copy()
    n = len(output)
    probability = np.full((n, 3), np.nan, dtype=np.float32)
    correction = np.full(n, np.nan, dtype=np.float32)
    direct = np.full(n, np.nan, dtype=np.float32)
    fold_id = np.full(n, -1, dtype=np.int32)
    audit: list[dict[str, Any]] = []
    decision, available = output.decision_ts, output.label_available_ts
    next_fold = 0
    for side in _SIDES:
        positions = np.flatnonzero(output.side_name.eq(side).to_numpy())
        if not len(positions):
            continue
        blocks = _validation_blocks(decision.iloc[positions].reset_index(drop=True), available.iloc[positions].reset_index(drop=True), n_folds=spec.n_validation_folds, min_train_rows=spec.min_train_rows)
        for local in blocks:
            valid_idx = positions[np.asarray(local, dtype=np.int32)]
            start = decision.iloc[valid_idx].min()
            train_idx = np.flatnonzero(output.side_name.eq(side).to_numpy() & available.lt(start).to_numpy())
            if len(train_idx) < spec.min_train_rows:
                audit.append({"side_name": side, "fold_id": next_fold, "status": "insufficient_prior_rows", "train_rows": len(train_idx), "validation_rows": len(valid_idx), "validation_start_ts": start})
                next_fold += 1
                continue
            labels, state = _fit_direct_correctness(
                output.exact_net_bps.iloc[train_idx].to_numpy(float), output.base_raw_score.iloc[train_idx].to_numpy(float), score_domain=spec.score_domain,
            )
            model = _direct_fit(frame.iloc[train_idx].loc[:, list(feature_names)], labels, None, params=spec.model_params)
            p = _multiclass_probabilities(model, frame.iloc[valid_idx].loc[:, list(feature_names)])
            delta, combined = _reconstruct_direct_correctness(p, output.base_raw_score.iloc[valid_idx].to_numpy(float), state)
            probability[valid_idx], correction[valid_idx], direct[valid_idx], fold_id[valid_idx] = p, delta, combined, next_fold
            audit.append({
                "side_name": side, "fold_id": next_fold, "status": "scored", "train_rows": len(train_idx), "validation_rows": len(valid_idx),
                "validation_start_ts": start, "train_max_label_available_ts": available.iloc[train_idx].max(),
                "strict_prior_resolved": bool(available.iloc[train_idx].lt(start).all()),
                "target_semantics": DIRECT_FQ3_SEMANTICS, "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
                "q33": state.thresholds[0], "q67": state.thresholds[1], "class_support": list(state.class_support),
            })
            next_fold += 1
    output["meta_p_error_tercile_0"] = probability[:, 0]
    output["meta_p_error_tercile_1"] = probability[:, 1]
    output["meta_p_error_tercile_2"] = probability[:, 2]
    output["meta_direct_correction"] = correction
    output["meta_direct_score"] = direct
    output["meta_strict_oof_available"] = np.isfinite(direct)
    output["meta_oof_fold_id"] = fold_id
    return output, pd.DataFrame(audit)


def _mapped_metrics(frame: pd.DataFrame, *, arm: str, spec: StageIIDirectFQ3Spec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.loc[frame.meta_strict_oof_available].copy()
    work["__stage_ii_side_candidate_key__"] = _side_candidate_key(work)
    work["net_bps"] = work.exact_net_bps.to_numpy(float)
    mapped, map_audit = apply_causal_21d_side_admission(
        work, score_column="meta_direct_score", net_column="net_bps", decision_column="decision_ts",
        label_available_column="label_available_ts", identity_column="__stage_ii_side_candidate_key__", spec=spec.admission_spec,
    )
    rows: list[dict[str, Any]] = []
    attribution: list[dict[str, Any]] = []
    populations = (
        ("without_21d_admission", mapped, "meta_direct_score"),
        ("with_21d_side_local_admission", mapped.loc[mapped.causal_21d_side_admitted_ge_50bps & mapped.causal_21d_side_expected_net_bps.notna()], "causal_21d_side_expected_net_bps"),
    )
    for scope, population, score in populations:
        ordered = population.sort_values([score, "__stage_ii_side_candidate_key__"], ascending=[False, True], kind="stable")
        for fraction in spec.top_fractions:
            selected = ordered.head(min(len(ordered), max(1, int(np.ceil(fraction * len(mapped))))))
            common = {"arm": arm, "admission_scope": scope, "top_fraction": float(fraction), "ranking_basis": "pooled_global_after_causal_common_bps_mapping_never_per_timestamp", "eligible_rows": len(population), "selected_rows": len(selected)}
            rows.append({**common, "gross_bps_per_trade": float(selected.exact_gross_bps.mean()) if len(selected) else np.nan, "net_bps_per_trade": float(selected.exact_net_bps.mean()) if len(selected) else np.nan})
            if len(selected):
                selection = selected.assign(month=selected.decision_ts.dt.strftime("%Y-%m"))
                for (side, month), chunk in selection.groupby(["side_name", "month"], observed=True):
                    attribution.append({**common, "side_name": side, "month": month, "net_bps_per_trade": float(chunk.exact_net_bps.mean()), "selected_rows": len(chunk)})
    return pd.DataFrame(rows), pd.DataFrame(attribution), map_audit


def _candidate_quality(features: pd.DataFrame, *, candidate_id: str) -> dict[str, Any]:
    available = pd.to_numeric(features[f"{META_ARCHETYPE_PREFIX}available"], errors="coerce").eq(1.0)
    return {"candidate_id": candidate_id, "available_rows": int(available.sum()), "available_fraction": float(available.mean())}


def run_stage_ii_direct_fq3_archetype_funnel(
    frame: pd.DataFrame, *, spec: StageIIDirectFQ3Spec, candidates: Sequence[StageIIDirectFQ3Candidate],
) -> StageIIDirectFQ3Result:
    """Run the bounded native-score Stage-II archetype addition.

    The two arms use the same strict-OOF rows per candidate.  Candidate choice
    is intentionally shallow: adequate causal coverage first, then the
    admitted pooled-global top-10 net result.  This avoids silently turning
    archetype discovery into a large model/geometry search.
    """
    work = validate_stage_ii_direct_fq3_ledger(frame, spec=spec)
    state_columns = _base_state_columns(work)
    direct_base_features = tuple(dict.fromkeys((*_base_features(state_columns), *spec.meta_feature_cols)))
    choices = tuple(candidates)
    if not choices or len(choices) > 8 or len({x.candidate_id for x in choices}) != len(choices):
        raise StageIIDirectFQ3Error("declare one to eight uniquely named archetype candidates")
    all_arms: list[StageIIDirectFQ3ArmResult] = []
    audit_rows: list[dict[str, Any]] = []
    archetypes_by_id: dict[str, pd.DataFrame] = {}
    for candidate in choices:
        if not str(candidate.candidate_id).strip():
            raise StageIIDirectFQ3Error("archetype candidate id must be non-empty")
        # Archetype discovery needs a bps reference only to construct realised
        # train-side path modes.  It is an outcome descriptor, never an FQ3
        # model feature.  A literal zero reference preserves native-score
        # semantics and prevents a mapped base score from entering Stage II.
        archetype_input = work.copy()
        reference_name = "__stage_ii_path_reference_zero_bps__"
        archetype_input[reference_name] = 0.0
        config = candidate.archetype_config
        if config.base_expected_net_col != reference_name:
            config = StageIIMetaArchetypeConfig(**{**config.__dict__, "base_expected_net_col": reference_name})
        result = strict_oof_meta_archetype_features(archetype_input, config=config, causal_feature_cols=candidate.causal_feature_cols)
        archetypes = result.features.reset_index(drop=True)
        archetypes_by_id[candidate.candidate_id] = archetypes
        quality = _candidate_quality(archetypes, candidate_id=candidate.candidate_id)
        available = pd.to_numeric(archetypes[f"{META_ARCHETYPE_PREFIX}available"], errors="coerce").eq(1.0).to_numpy()
        # Every control uses this candidate's same available rows.  Unknown
        # burn-in is never made into an ordinary archetype class.
        common = work.loc[available].reset_index(drop=True)
        arch = archetypes.loc[available].reset_index(drop=True)
        if len(common) < spec.min_train_rows:
            audit_rows.append({**quality, "disposition": "diagnostic_insufficient_oof_support"})
            continue
        for arm in _ARMS:
            additions: tuple[str, ...] = () if arm == "none" else tuple([
                *membership_feature_names(config.components), f"{META_ARCHETYPE_PREFIX}prob__unknown",
                f"{META_ARCHETYPE_PREFIX}entropy", f"{META_ARCHETYPE_PREFIX}confidence",
                f"{META_ARCHETYPE_PREFIX}support_log1p", f"{META_ARCHETYPE_PREFIX}available",
            ])
            features = tuple(dict.fromkeys((*direct_base_features, *additions)))
            design = pd.concat([
                common.loc[:, list(direct_base_features)].reset_index(drop=True),
                arch.loc[:, list(additions)].reset_index(drop=True),
            ], axis=1)
            # Keep identity/economic fields outside the explicit model feature
            # list.  The strict routine uses them only for chronology/labels.
            model_input = common.loc[:, [
                "candidate_id", "symbol", "signal_close_ts", "decision_ts", "label_available_ts",
                "side_name", "exact_gross_bps", "exact_net_bps", "base_raw_score",
            ]].copy()
            for name in features:
                if name != "base_raw_score":
                    model_input[name] = design[name].to_numpy()
            prediction, folds = _strict_direct_predictions(model_input, feature_names=features, spec=spec)
            metrics, contributions, map_audit = _mapped_metrics(prediction, arm=arm, spec=spec)
            all_arms.append(StageIIDirectFQ3ArmResult(candidate.candidate_id, arm, features, prediction, metrics, contributions, map_audit, folds))
        audit_rows.append({**quality, "disposition": "evaluated", "archetype_manifest": result.manifest})
    audit = pd.DataFrame(audit_rows)
    if not all_arms:
        return StageIIDirectFQ3Result(
            audit, (), None, None, (), None, None,
            {
                "schema": SCHEMA,
                "decision": "NO_STAGE_II_DIRECT_FQ3_ARM_ADVANCES",
                "reason": "no candidate had sufficient strict OOF support",
                "base_state_columns": list(state_columns),
            },
        )
    # A Stage-II addition is promoted only by its *increment* against the
    # matching no-archetype direct-FQ3 control on the exact same candidate
    # population.  Selecting ``none`` would merely rediscover Stage I, so it
    # is an explicit no-advance outcome.  Different archetype candidates can
    # have different availability burn-ins; cross-candidate choice therefore
    # uses this robust within-candidate delta, never their raw tail levels.
    by_candidate = {(item.candidate_id, item.arm): item for item in all_arms}

    def _tail(result: StageIIDirectFQ3ArmResult) -> float:
        row = result.metrics.loc[
            result.metrics.admission_scope.eq("with_21d_side_local_admission")
            & np.isclose(result.metrics.top_fraction, 0.10)
        ]
        return float(row.net_bps_per_trade.iloc[0]) if len(row) == 1 else float("nan")

    def _month_net(result: StageIIDirectFQ3ArmResult) -> pd.Series:
        rows = result.contributions.loc[
            result.contributions.admission_scope.eq("with_21d_side_local_admission")
            & np.isclose(result.contributions.top_fraction, 0.10)
        ].copy()
        if rows.empty:
            return pd.Series(dtype=float)
        weighted = rows.net_bps_per_trade.to_numpy(float) * rows.selected_rows.to_numpy(float)
        table = pd.DataFrame({"month": rows.month.astype(str), "weighted": weighted, "rows": rows.selected_rows.to_numpy(float)})
        grouped = table.groupby("month", sort=True).sum(numeric_only=True)
        return grouped.weighted / grouped.rows

    promotions: list[tuple[tuple[float, float, float, str], StageIIDirectFQ3ArmResult]] = []
    delta_audit: dict[str, dict[str, Any]] = {}
    for candidate in choices:
        control, soft = by_candidate.get((candidate.candidate_id, "none")), by_candidate.get((candidate.candidate_id, "soft_memberships"))
        if control is None or soft is None:
            delta_audit[candidate.candidate_id] = {"delta_disposition": "missing_matched_control"}
            continue
        control_tail, soft_tail = _tail(control), _tail(soft)
        control_month, soft_month = _month_net(control), _month_net(soft)
        common_months = control_month.index.intersection(soft_month.index)
        delta = soft_tail - control_tail if np.isfinite(control_tail) and np.isfinite(soft_tail) else np.nan
        month_delta = (soft_month.loc[common_months] - control_month.loc[common_months]) if len(common_months) else pd.Series(dtype=float)
        worst_delta = float(month_delta.min()) if len(month_delta) else np.nan
        latest_delta = float(month_delta.loc[month_delta.index.max()]) if len(month_delta) else np.nan
        admitted = np.isfinite(delta) and (not spec.require_admission_for_selection or (np.isfinite(control_tail) and np.isfinite(soft_tail)))
        advances = bool(admitted and delta > 0.0 and np.isfinite(worst_delta) and worst_delta >= 0.0 and np.isfinite(latest_delta) and latest_delta >= 0.0)
        delta_audit[candidate.candidate_id] = {
            "matching_control_rows": len(control.oof_predictions), "matching_soft_rows": len(soft.oof_predictions),
            "control_top10_net_bps": control_tail, "soft_top10_net_bps": soft_tail,
            "aggregate_top10_delta_bps": delta, "worst_month_top10_delta_bps": worst_delta,
            "latest_month_top10_delta_bps": latest_delta, "paired_month_count": len(month_delta),
            "delta_disposition": "advances" if advances else "does_not_beat_matched_control",
        }
        if advances:
            promotions.append(((-delta, -worst_delta, -latest_delta, candidate.candidate_id), soft))
    if not audit.empty:
        for index, row in audit.iterrows():
            details = delta_audit.get(str(row.candidate_id), {})
            for name, value in details.items():
                audit.loc[index, name] = value
            if row.get("disposition") == "evaluated":
                audit.loc[index, "disposition"] = details.get("delta_disposition", "does_not_beat_matched_control")
    if not promotions:
        return StageIIDirectFQ3Result(
            audit, tuple(all_arms), None, None, (), None, None,
            {
                "schema": SCHEMA,
                "decision": "NO_STAGE_II_ARCHETYPE_ADVANCES",
                "reason": "no soft-archetype arm cleared its matching robust delta gate",
                "selection": "within_candidate_matched_none_control_then_cross_candidate_robust_delta",
                "base_state_columns": list(state_columns),
            },
        )
    selected = sorted(promotions, key=lambda item: item[0])[0][1]
    selected_arch = archetypes_by_id.get(selected.candidate_id)
    return StageIIDirectFQ3Result(
        audit, tuple(all_arms), selected.candidate_id, selected.arm, selected.feature_names,
        selected.oof_predictions, selected_arch,
        {"schema": SCHEMA, "decision": "STAGE_II_DIRECT_FQ3_ARCHETYPE_ARM_SELECTED", "selected_candidate_id": selected.candidate_id, "selected_arm": selected.arm,
         "meta_target": DIRECT_FQ3_SEMANTICS, "meta_input": DIRECT_BASE_INPUT_SEMANTICS,
         "archetype_prior": "diagnostic_only_not_an_FQ3_input", "mapping": "causal_side_local_21d_common_bps_after_native_direct_reconstruction", "ranking": "pooled_global_only_never_per_timestamp", "selection": "soft_archetype_must_beat_matching_none_on_identical_rows_then_cross_candidate_robust_delta", "hard_routing": False, "local_experts": False,
         "base_state_columns": list(state_columns)},
    )


def score_frozen_stage_ii_direct_fq3(
    *, training: pd.DataFrame, mapping_reference: pd.DataFrame, evaluation: pd.DataFrame,
    spec: StageIIDirectFQ3Spec, candidate: StageIIDirectFQ3Candidate, selected_arm: str,
) -> pd.DataFrame:
    """Score one later period for the frozen direct-FQ3/archetype choice.

    ``mapping_reference`` must contain strict-OOF direct meta scores produced
    by the selected development arm.  This prohibits an in-sample training
    score from becoming the historical source for the causal bps map.
    """
    if selected_arm not in _ARMS:
        raise StageIIDirectFQ3Error("frozen Stage-II direct scorer accepts only declared direct arms")
    train = validate_stage_ii_direct_fq3_ledger(training, spec=spec)
    test = validate_stage_ii_direct_fq3_ledger(evaluation, spec=spec)
    train_state_columns = _base_state_columns(train)
    test_state_columns = _base_state_columns(test)
    if train_state_columns != test_state_columns:
        raise StageIIDirectFQ3Error(
            "frozen Stage-II training/evaluation base state contracts differ"
        )
    if set(train.__stage_ii_side_candidate_key__).intersection(test.__stage_ii_side_candidate_key__):
        raise StageIIDirectFQ3Error("frozen Stage-II train/evaluation identities overlap")
    start = test.decision_ts.min()
    if not train.label_available_ts.lt(start).all():
        raise StageIIDirectFQ3Error("frozen Stage-II training labels are not resolved before evaluation")
    reference_required = {"candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "meta_direct_score", "meta_strict_oof_available"}
    if missing := sorted(reference_required.difference(mapping_reference.columns)):
        raise StageIIDirectFQ3Error(f"mapping reference lacks strict-OOF direct meta evidence: {missing}")
    reference = mapping_reference.copy()
    reference["decision_ts"] = _utc(reference, "decision_ts")
    reference["label_available_ts"] = _utc(reference, "label_available_ts")
    reference["side_name"] = reference.side_name.astype(str).str.lower()
    reference["__stage_ii_side_candidate_key__"] = _side_candidate_key(reference)
    if reference["__stage_ii_side_candidate_key__"].duplicated().any():
        raise StageIIDirectFQ3Error("mapping reference contains duplicate side-qualified candidate ids")
    if not _bool(reference.meta_strict_oof_available, name="meta_strict_oof_available").all() or not reference.label_available_ts.lt(start).all():
        raise StageIIDirectFQ3Error("mapping reference must be fully strict-OOF and resolved before evaluation")
    causal = tuple(candidate.causal_feature_cols)
    path_reference = "__stage_ii_path_reference_zero_bps__"
    train_state = train.copy()
    train_state[path_reference] = 0.0
    config = candidate.archetype_config
    if config.base_expected_net_col != path_reference:
        config = StageIIMetaArchetypeConfig(**{**config.__dict__, "base_expected_net_col": path_reference})
    state = SideLocalMetaArchetypeState(config, causal).fit(train_state)
    safe_train = train_state.drop(columns=[config.exact_net_col, *config.path_descriptor_cols], errors="ignore")
    safe_test = test.drop(columns=[config.exact_net_col, *config.path_descriptor_cols], errors="ignore")
    train_arch, test_arch = state.transform(safe_train), state.transform(safe_test)
    additions: tuple[str, ...] = () if selected_arm == "none" else tuple([
        *membership_feature_names(config.components), f"{META_ARCHETYPE_PREFIX}prob__unknown",
        f"{META_ARCHETYPE_PREFIX}entropy", f"{META_ARCHETYPE_PREFIX}confidence",
        f"{META_ARCHETYPE_PREFIX}support_log1p", f"{META_ARCHETYPE_PREFIX}available",
    ])
    base_features = tuple(dict.fromkeys((*_base_features(train_state_columns), *spec.meta_feature_cols)))
    feature_names = tuple((*base_features, *additions))
    train_design = pd.concat([train.loc[:, list(base_features)].reset_index(drop=True), train_arch.loc[:, list(additions)].reset_index(drop=True)], axis=1)
    test_design = pd.concat([test.loc[:, list(base_features)].reset_index(drop=True), test_arch.loc[:, list(additions)].reset_index(drop=True)], axis=1)
    probability = np.full((len(test), 3), np.nan, dtype=np.float32)
    correction = np.full(len(test), np.nan, dtype=np.float32)
    direct = np.full(len(test), np.nan, dtype=np.float32)
    for side in _SIDES:
        train_idx = np.flatnonzero(train.side_name.eq(side).to_numpy())
        test_idx = np.flatnonzero(test.side_name.eq(side).to_numpy())
        if not len(test_idx):
            continue
        if len(train_idx) < spec.min_train_rows:
            raise StageIIDirectFQ3Error(f"frozen {side} direct FQ3 model lacks prior training support")
        labels, state_fq3 = _fit_direct_correctness(train.exact_net_bps.iloc[train_idx].to_numpy(float), train.base_raw_score.iloc[train_idx].to_numpy(float), score_domain=spec.score_domain)
        model = _direct_fit(train_design.iloc[train_idx], labels, None, params=spec.model_params)
        p = _multiclass_probabilities(model, test_design.iloc[test_idx])
        delta, combined = _reconstruct_direct_correctness(p, test.base_raw_score.iloc[test_idx].to_numpy(float), state_fq3)
        probability[test_idx], correction[test_idx], direct[test_idx] = p, delta, combined
    if not np.isfinite(direct).all():
        raise StageIIDirectFQ3Error("frozen direct FQ3 scorer did not score every evaluation row")
    out = test.copy().reset_index(drop=True)
    out["total_cost_bps"] = 100.0
    for name in test_arch:
        out[name] = test_arch[name].to_numpy()
    out["meta_p_error_tercile_0"], out["meta_p_error_tercile_1"], out["meta_p_error_tercile_2"] = probability[:, 0], probability[:, 1], probability[:, 2]
    out["meta_direct_correction"], out["meta_direct_score"] = correction, direct
    out["meta_strict_oof_available"] = True
    out["meta_is_strict_oof"] = True
    out["meta_source_side"] = out.side_name.astype(str)
    out["meta_score_semantics"] = DIRECT_FQ3_SEMANTICS
    # Map the reconstructed native score only after the FQ3 model has run.
    reference = reference.loc[:, ["candidate_id", "side_name", "decision_ts", "label_available_ts", "exact_net_bps", "meta_direct_score", "__stage_ii_side_candidate_key__"]].copy()
    reference["signal_close_ts"] = reference.decision_ts - pd.Timedelta(hours=1)
    reference["exact_gross_bps"] = reference.exact_net_bps.to_numpy(float) + 100.0
    reference["net_bps"] = reference.exact_net_bps.to_numpy(float)
    out["__stage_ii_side_candidate_key__"] = _side_candidate_key(out)
    map_input = pd.concat([reference, out.assign(net_bps=out.exact_net_bps)], ignore_index=True, sort=False)
    mapped, audit = apply_causal_21d_side_admission(map_input, score_column="meta_direct_score", net_column="net_bps", decision_column="decision_ts", label_available_column="label_available_ts", identity_column="__stage_ii_side_candidate_key__", spec=spec.admission_spec)
    mapped = mapped.loc[mapped["__stage_ii_side_candidate_key__"].isin(out["__stage_ii_side_candidate_key__"])].set_index("__stage_ii_side_candidate_key__").reindex(out["__stage_ii_side_candidate_key__"]).reset_index()
    # This bps value is the *joint* direct-base + FQ3-correction score.  It
    # must never be presented as a base-only expected value to a later layer.
    joint_expected = mapped.causal_21d_side_expected_net_bps.to_numpy()
    out[JOINT_EXPECTED_NET_COLUMN] = joint_expected
    out["meta_causal_21d_expected_net_bps"] = joint_expected
    out["joint_expected_net_bps_semantics"] = JOINT_MAPPING_SEMANTICS
    out["joint_map_is_prequential"] = True
    out["joint_map_source_side"] = out.side_name.astype(str)
    out["causal_21d_admitted"] = mapped.causal_21d_side_admitted_ge_50bps.to_numpy(bool)
    out["causal_21d_admission_is_prequential"] = True
    out["causal_21d_admission_source_side"] = out.side_name.astype(str)
    # A no-support early row is impossible here because reference precedes the
    # test; keep the exact latest reference cutoff for Stage-III validation.
    max_available = reference.label_available_ts.max()
    out["causal_21d_admission_max_label_available_ts"] = max_available
    out["causal_21d_admission_window_days"] = 21
    out["joint_map_max_label_available_ts"] = max_available
    # Compatibility only for consumers which have not yet moved to the
    # canonical joint column.  Direct-FQ3 Stage III explicitly rejects this
    # alias as its upstream score column.
    out["prequential_base_expected_net_bps"] = joint_expected
    out["prequential_base_expected_net_bps_semantics"] = LEGACY_BASE_ALIAS_SEMANTICS
    out["base_map_is_prequential"] = True
    out["base_map_source_side"] = out.side_name.astype(str)
    out["base_map_max_label_available_ts"] = max_available
    # The Stage-III direct-R3 reader retains these aliases for its historical
    # three-state contract.  A two- or five-state Stage-I winner is still a
    # valid Stage-II input, but must not be relabelled as R3 semantics.
    if len(train_state_columns) == 3:
        out["r3_is_strict_oof"] = True
        out["r3_source_side"] = out.side_name.astype(str)
        out["r3_fit_end_ts"] = train.label_available_ts.max()
        out["r3_score_semantics"] = "same_side_direct_strict_oof_probabilities_without_conversion"
        out["r3_p_adverse"], out["r3_p_weak"], out["r3_p_clear"] = (
            out[train_state_columns[0]], out[train_state_columns[1]], out[train_state_columns[2]]
        )
    out["stage_ii_meta_mapping_audit_rows"] = len(audit)
    return out


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_stage_ii_direct_fq3_locked_oos_bundle(
    output_directory: str | Path, *, ledger: pd.DataFrame, winner_bundle: str | Path,
) -> Path:
    """Publish a checksummed Stage-II direct-FQ3 OOS ledger for Stage III.

    The publication intentionally has the same small, immutable source shape
    accepted by the Stage-III CLI.  It is a writer only: configuration choice,
    HPO, feature selection and model fitting must have happened earlier.
    """
    from .stage_ii_production_oos import load_stage_ii_winner_bundle

    winner_root = Path(winner_bundle).resolve()
    winner = load_stage_ii_winner_bundle(winner_root)
    required = {
        "candidate_id", "symbol", "signal_close_ts", "decision_ts", "label_available_ts", "side_name",
        "exact_gross_bps", "exact_net_bps", "total_cost_bps", JOINT_EXPECTED_NET_COLUMN,
        "meta_causal_21d_expected_net_bps", "joint_expected_net_bps_semantics",
        "joint_map_is_prequential", "joint_map_source_side", "joint_map_max_label_available_ts",
        "prequential_base_expected_net_bps", "prequential_base_expected_net_bps_semantics",
        "r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_is_strict_oof", "r3_source_side",
        "r3_fit_end_ts", "r3_score_semantics", "base_map_is_prequential", "base_map_source_side",
        "base_map_max_label_available_ts", "meta_is_strict_oof", "meta_source_side", "meta_score_semantics",
        "meta_direct_score", "meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2",
        "causal_21d_admitted", "causal_21d_admission_is_prequential", "causal_21d_admission_source_side",
        "causal_21d_admission_max_label_available_ts", "causal_21d_admission_window_days",
    }
    if missing := sorted(required.difference(ledger.columns)):
        raise StageIIDirectFQ3Error(f"Stage-II direct locked OOS ledger lacks fields: {missing[:12]}")
    # Reuse the direct input validator for all common identity/timing/base
    # evidence, then validate the post-FQ3 output that Stage III consumes.
    work = ledger.copy()
    for name in ("decision_ts", "signal_close_ts", "label_available_ts"):
        work[name] = _utc(work, name)
    if work.duplicated(["candidate_id", "symbol", "decision_ts", "side_name"]).any():
        raise StageIIDirectFQ3Error("Stage-II direct OOS ledger has duplicate identity")
    if not _bool(work.meta_is_strict_oof, name="meta_is_strict_oof").all() or not work.meta_source_side.astype(str).str.lower().eq(work.side_name.astype(str).str.lower()).all():
        raise StageIIDirectFQ3Error("Stage-II direct OOS meta correction is not strict same-side evidence")
    if not work.meta_score_semantics.astype(str).eq(DIRECT_FQ3_SEMANTICS).all():
        raise StageIIDirectFQ3Error("Stage-II direct OOS ledger has wrong FQ3 semantics")
    p = work.loc[:, ["meta_p_error_tercile_0", "meta_p_error_tercile_1", "meta_p_error_tercile_2"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise StageIIDirectFQ3Error("Stage-II direct OOS FQ3 probabilities must be a simplex")
    expected = pd.to_numeric(work[JOINT_EXPECTED_NET_COLUMN], errors="coerce")
    if not expected.notna().all():
        raise StageIIDirectFQ3Error("Stage-III bridge requires a causal common-bps map for every locked OOS row")
    if not work.joint_expected_net_bps_semantics.astype(str).eq(JOINT_MAPPING_SEMANTICS).all():
        raise StageIIDirectFQ3Error("Stage-II direct OOS ledger has wrong joint common-bps semantics")
    if not _bool(work.joint_map_is_prequential, name="joint_map_is_prequential").all() or not work.joint_map_source_side.astype(str).str.lower().eq(work.side_name.astype(str).str.lower()).all():
        raise StageIIDirectFQ3Error("Stage-II direct OOS joint map is not prequential and same-side")
    if not (_utc(work, "joint_map_max_label_available_ts") < _utc(work, "decision_ts")).all():
        raise StageIIDirectFQ3Error("Stage-II direct OOS joint map uses current/future resolved labels")
    if not np.allclose(expected.to_numpy(float), pd.to_numeric(work.meta_causal_21d_expected_net_bps, errors="coerce").to_numpy(float), equal_nan=False):
        raise StageIIDirectFQ3Error("Stage-II direct OOS joint map aliases disagree")
    if not work.prequential_base_expected_net_bps_semantics.astype(str).eq(LEGACY_BASE_ALIAS_SEMANTICS).all():
        raise StageIIDirectFQ3Error("legacy base expected-net field must be explicitly marked deprecated")
    root = Path(output_directory).resolve()
    if root.exists() or not root.parent.is_dir():
        raise StageIIDirectFQ3Error("locked OOS output must be a new directory under an existing parent")
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        oos = temporary / "locked_oos_ledger.parquet"
        work.to_parquet(oos, index=False, compression="zstd")
        content = pd.util.hash_pandas_object(work.sort_values(["decision_ts", "candidate_id"], kind="stable"), index=False, categorize=True).to_numpy(np.uint64)
        run_manifest = {
            "schema": "stage_ii_direct_fq3_locked_oos_v1", "status": "complete",
            "winner_manifest_sha256": _file_sha256(winner_root / "winner_manifest.json"),
            "stage_i_base_winner_artifact_sha256": winner.stage_i_base_winner_artifact_sha256,
            "stage_i_base_oof_ledger_sha256": winner.stage_i_base_oof_ledger_sha256,
            "oos_content_sha256": sha256(content.tobytes()).hexdigest(),
            "meta_target": DIRECT_FQ3_SEMANTICS, "meta_input": DIRECT_BASE_INPUT_SEMANTICS,
            "mapping": "causal_side_local_21d_common_bps_after_direct_reconstruction",
            "canonical_upstream_score_column": JOINT_EXPECTED_NET_COLUMN,
            "canonical_upstream_score_semantics": JOINT_MAPPING_SEMANTICS,
            "legacy_base_alias": "prequential_base_expected_net_bps",
            "legacy_base_alias_semantics": LEGACY_BASE_ALIAS_SEMANTICS,
            "ranking": "pooled_global_only_never_per_timestamp", "selection_forbidden": True,
            "reselection_forbidden": True, "hpo_forbidden": True, "hard_routing": False,
            "local_experts": False,
        }
        manifest_path = temporary / "run_manifest.json"
        manifest_path.write_text(json.dumps(run_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        checksums = {name: _file_sha256(temporary / name) for name in (oos.name, manifest_path.name)}
        (temporary / "checksums.json").write_text(json.dumps(checksums, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return root


__all__ = [
    "SCHEMA", "JOINT_EXPECTED_NET_COLUMN", "JOINT_MAPPING_SEMANTICS", "StageIIDirectFQ3Error", "StageIIDirectFQ3Spec", "StageIIDirectFQ3Candidate",
    "StageIIDirectFQ3ArmResult", "StageIIDirectFQ3Result", "validate_stage_ii_direct_fq3_ledger",
    "materialize_stage_ii_direct_fq3_handoff",
    "run_stage_ii_direct_fq3_archetype_funnel", "score_frozen_stage_ii_direct_fq3",
    "publish_stage_ii_direct_fq3_locked_oos_bundle",
]
