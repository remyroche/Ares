"""Training-only outcome context for Stage-I MDA archetype scoring.

These labels describe resolved outcomes and path states.  They are permitted
only inside feature-selection diagnostics; this module never adds a column to
the model matrix or to an inference contract.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


MDA_SUPPORT_MODES = ("full", "target-only")


def restrict_stage_i_mda_training_support(
    training_support: dict[str, Any], *, mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restrict realised-path MDA support for a predeclared control arm.

    The target vector is supplied independently to every selector.  Therefore
    ``target-only`` preserves only side identity and supervised-row validity;
    it cannot expose event, realised economics, path state, or an outcome
    archetype to feature selection.  This lets an otherwise-identical arm
    measure whether those training-only aids are genuinely incremental.
    """
    if mode not in MDA_SUPPORT_MODES:
        raise ValueError(f"unknown MDA support mode: {mode}")
    context = training_support.get("label_context")
    if not isinstance(context, dict):
        raise ValueError("MDA training support lacks label_context")
    if mode == "full":
        return context, {
            "mode": mode,
            "realised_path_support_available": True,
            "archetype_conditioned_enabled": True,
            "removed_label_context_fields": [],
        }
    required = ("side_name", "valid_resolved_support")
    missing = [field for field in required if field not in context]
    if missing:
        raise ValueError(f"target-only MDA support lacks {missing}")
    reduced = {field: context[field] for field in required}
    return reduced, {
        "mode": mode,
        "realised_path_support_available": False,
        "archetype_conditioned_enabled": False,
        "removed_label_context_fields": sorted(
            str(field) for field in context if field not in reduced
        ),
    }


_IDENTITY = ("candidate_id", "__ts__", "__symbol__")


def _label_hash(labels: np.ndarray) -> str:
    values = pd.Series(np.asarray(labels, dtype=object).astype(str))
    return hashlib.sha256(
        pd.util.hash_pandas_object(values, index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    ).hexdigest()


def _identity_hash(ledger: pd.DataFrame, identity: Sequence[str]) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(ledger.loc[:, list(identity)], index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    ).hexdigest()


def _broad_time_era(timestamps: pd.Series) -> np.ndarray:
    """Three fixed-in-reference broad eras; used only for audit slicing."""
    n = len(timestamps)
    values = pd.to_datetime(timestamps, utc=True, errors="coerce")
    if values.isna().any() or n < 3:
        return np.full(n, "era_unknown", dtype=object)
    # Rank bins are deterministic for the frozen training reference and avoid
    # inventing a market-state or latent/AE/GMM feature.
    ranks = values.rank(method="first", pct=True).to_numpy(dtype=np.float32)
    return np.where(
        ranks <= 1.0 / 3.0,
        "era_early",
        np.where(ranks <= 2.0 / 3.0, "era_middle", "era_late"),
    ).astype(object)


def _cross_tab(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_order: Sequence[str],
    right_order: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Return a complete, serialisable support cross-tab (including zeroes)."""
    table = pd.crosstab(
        pd.Series(left, dtype="object"),
        pd.Series(right, dtype="object"),
        dropna=False,
    )
    return {
        str(left_key): {
            str(right_key): int(
                table.reindex(
                    index=[left_key], columns=[right_key], fill_value=0
                ).iat[0, 0]
            )
            for right_key in right_order
        }
        for left_key in left_order
    }


def _truthy_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Interpret optional validity flags conservatively for training support."""
    values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce")
    textual = values.astype("string").str.strip().str.lower()
    return (
        numeric.ge(0.5).fillna(False)
        | textual.isin(("true", "t", "yes", "y", "1"))
    ).to_numpy(dtype=bool)


def build_stage_i_mda_training_support(
    ledger: pd.DataFrame,
    *,
    side: str,
    identity_columns: Sequence[str] = _IDENTITY,
    decision_timestamps: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Create strictly training-only support and archetype labels.

    The label is intentionally compact: ``side × broad era × economic/path
    state``.  It uses only resolved ledger outcomes (R3 class, robust-clear
    softness, first-touch event, and exact-net bin), and is passed separately
    from ``X``.  This permits MDA to test feature robustness by economically
    meaningful slices without creating an inference-time "god feature".
    """
    required = {
        *identity_columns,
        "side_name",
        "__ts__",
        "r3_class",
        "robust_clear_soft_b25_t50",
        "t2_tp6_sl4_event",
        "exact_net_bps",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(
            "Stage-I MDA training support lacks required resolved ledger fields: "
            f"{missing}"
        )
    work = ledger.reset_index(drop=True)
    n = len(work)
    normalized_side = str(side).strip().lower()
    observed_side = work["side_name"].astype(str).str.lower()
    if (
        normalized_side not in {"long", "short"}
        or not observed_side.eq(normalized_side).all()
    ):
        raise ValueError(
            "Stage-I MDA training support must be exactly one declared side"
        )
    r3 = pd.to_numeric(work["r3_class"], errors="coerce").to_numpy(np.float32)
    soft = pd.to_numeric(
        work["robust_clear_soft_b25_t50"], errors="coerce"
    ).to_numpy(np.float32)
    event = pd.to_numeric(
        work["t2_tp6_sl4_event"], errors="coerce"
    ).to_numpy(np.float32)
    net = pd.to_numeric(work["exact_net_bps"], errors="coerce").to_numpy(np.float32)

    # The first-touch event is canonical for path identity.  R3 is deliberately
    # retained as selector-ledger provenance, not used as an alternate route to
    # a clear state: R3 can be clear when a timeout had a favourable intra-path
    # excursion, which must never create a robust-clear support label.
    event_upper = event == 0.0
    event_lower = event == 1.0
    event_timeout = event == 2.0
    primitive_resolved = (
        np.isfinite(r3)
        & np.isin(r3, (0.0, 1.0, 2.0))
        & np.isfinite(soft)
        & (soft >= 0.0)
        & (soft <= 1.0)
        & np.isfinite(event)
        & np.isin(event, (0.0, 1.0, 2.0))
        & np.isfinite(net)
    )
    explicit_valid = (
        _truthy_column(work, "label_valid")
        if "label_valid" in work.columns
        else np.ones(n, dtype=bool)
    )
    explicit_invalid = (
        _truthy_column(work, "target_invalid")
        if "target_invalid" in work.columns
        else np.zeros(n, dtype=bool)
    )
    valid_resolved = primitive_resolved & explicit_valid & ~explicit_invalid

    event_state = np.select(
        [event_upper, event_lower, event_timeout],
        ["upper", "lower", "timeout"],
        default="invalid_unresolved",
    ).astype(object)
    r3_state = np.select(
        [r3 == 0.0, r3 == 1.0, r3 == 2.0],
        ["adverse", "weak", "clear"],
        default="invalid_unresolved",
    ).astype(object)
    net_bin = np.select(
        [net <= -100.0, net <= 0.0, net <= 50.0, net > 50.0],
        ["severe_loss", "loss", "net_0_to_50", "net_gt_50"],
        default="invalid_unresolved",
    ).astype(object)
    net_bin[~valid_resolved] = "invalid_unresolved"

    # These assignments are mutually exclusive by construction.  Invalid or
    # unresolved rows remain in a fail-closed state for auditability and cannot
    # be relabelled as timeout, adverse, or clear through R3 provenance.
    path_state = np.full(n, "invalid_unresolved", dtype=object)
    path_state[valid_resolved & event_lower] = "adverse"
    path_state[valid_resolved & event_timeout] = "weak_timeout"
    upper_valid = valid_resolved & event_upper
    path_state[upper_valid & (net > 50.0) & (soft >= 0.50)] = "robust_clear"
    path_state[upper_valid & ((net <= 50.0) | (soft < 0.50))] = "marginal_clear"
    if decision_timestamps is None:
        era_timestamps = work["__ts__"]
        era_timestamp_semantics = "legacy_signal_close"
    else:
        era_timestamps = pd.Series(decision_timestamps).reset_index(drop=True)
        if len(era_timestamps) != n:
            raise ValueError("Stage-I MDA decision timestamps must be row-aligned")
        era_timestamp_semantics = "decision_ts"
    era = _broad_time_era(era_timestamps)
    archetypes = np.asarray(
        [
            f"{normalized_side}|{era_i}|{state_i}"
            for era_i, state_i in zip(era, path_state)
        ],
        dtype=object,
    )
    counts = pd.Series(archetypes).value_counts(dropna=False)
    state_order = (
        "robust_clear",
        "marginal_clear",
        "adverse",
        "weak_timeout",
        "invalid_unresolved",
    )
    event_order = ("upper", "lower", "timeout", "invalid_unresolved")
    r3_order = ("adverse", "weak", "clear", "invalid_unresolved")
    net_order = (
        "severe_loss",
        "loss",
        "net_0_to_50",
        "net_gt_50",
        "invalid_unresolved",
    )
    contradictory_event_state = (
        ((path_state == "robust_clear") & ~event_upper)
        | ((path_state == "marginal_clear") & ~event_upper)
        | ((path_state == "adverse") & ~event_lower)
        | ((path_state == "weak_timeout") & ~event_timeout)
        | ((path_state == "invalid_unresolved") & valid_resolved)
    )
    audit = {
        "schema": "stage_i_mda_training_support_v2",
        "source": "training_only:selector_ledger_r3_event_robust_clear_exact_net",
        "training_only": True,
        "inference_feature": False,
        "side": normalized_side,
        "rows": int(n),
        "identity_sha256": _identity_hash(work, identity_columns),
        "archetype_label_sha256": _label_hash(archetypes),
        "archetype_count": int(len(counts)),
        "archetype_support": {str(key): int(value) for key, value in counts.items()},
        "canonical_state_contract": {
            "state_name": "path_economic_state",
            "event_codes": {"upper": 0, "lower": 1, "timeout": 2},
            "states": {
                "robust_clear": (
                    "upper event AND exact_net_bps > 50 AND "
                    "robust_clear_soft_b25_t50 >= 0.50"
                ),
                "marginal_clear": (
                    "upper event AND (exact_net_bps <= 50 OR "
                    "robust_clear_soft_b25_t50 < 0.50)"
                ),
                "adverse": "lower event",
                "weak_timeout": "timeout event",
                "invalid_unresolved": (
                    "missing, out-of-domain, or explicitly invalid selector outcome"
                ),
            },
            "r3_class_role": (
                "audited selector-ledger provenance only; never an alternate "
                "path-state route"
            ),
            "invalid_rows": "fail_closed_invalid_unresolved",
        },
        "path_state_support": {
            state: int(np.sum(path_state == state)) for state in state_order
        },
        "event_support": {
            state: int(np.sum(event_state == state)) for state in event_order
        },
        "valid_support": {
            "valid_resolved": int(np.sum(valid_resolved)),
            "invalid_unresolved": int(np.sum(~valid_resolved)),
        },
        "economic_bin_support": {
            state: int(np.sum(net_bin == state)) for state in net_order
        },
        "cross_tabs": {
            "event_by_path_economic_state": _cross_tab(
                event_state, path_state, left_order=event_order, right_order=state_order
            ),
            "r3_class_by_path_economic_state": _cross_tab(
                r3_state, path_state, left_order=r3_order, right_order=state_order
            ),
            "event_by_r3_class": _cross_tab(
                event_state, r3_state, left_order=event_order, right_order=r3_order
            ),
            "path_economic_state_by_net_bin": _cross_tab(
                path_state, net_bin, left_order=state_order, right_order=net_order
            ),
        },
        "invariants": {
            "rows_accounted_for": bool(len(path_state) == n),
            "event_support_matches_rows": bool(
                sum(np.sum(event_state == state) for state in event_order) == n
            ),
            "valid_support_matches_rows": bool(
                np.sum(valid_resolved) + np.sum(~valid_resolved) == n
            ),
            "contradictory_event_state_rows": int(np.sum(contradictory_event_state)),
            "zero_contradictory_event_state_rows": bool(
                not np.any(contradictory_event_state)
            ),
            "valid_state_support": int(np.sum(path_state != "invalid_unresolved")),
            "invalid_rows_fail_closed": bool(
                np.all(path_state[~valid_resolved] == "invalid_unresolved")
            ),
        },
        "broad_era_support": {
            str(key): int(value) for key, value in pd.Series(era).value_counts().items()
        },
        "era_timestamp_semantics": era_timestamp_semantics,
    }
    # `feature_selection_archetype` is consumed by selector diagnostics only.
    # The other fields make support provenance explicit for the established
    # outcome/path archetype utilities, while never becoming feature columns.
    label_context: dict[str, Any] = {
        "feature_selection_archetype": archetypes,
        "side_name": observed_side.to_numpy(dtype=object),
        "r3_class": r3,
        "robust_clear_soft": np.where(valid_resolved, soft, np.nan).astype(
            np.float32
        ),
        "event_upper": (valid_resolved & event_upper).astype(np.float32),
        "event_lower": (valid_resolved & event_lower).astype(np.float32),
        "event_timeout": (valid_resolved & event_timeout).astype(np.float32),
        "is_timeout": (valid_resolved & event_timeout).astype(np.float32),
        "exit_code": np.where(
            ~valid_resolved,
            "unresolved",
            np.where(
                event_upper,
                "take_profit",
                np.where(event_lower, "stop_loss", "timeout"),
            ),
        ),
        "exact_net_bps": np.where(valid_resolved, net, np.nan).astype(np.float32),
        "y_ret": np.where(valid_resolved, net, np.nan).astype(np.float32),
        "economic_bin": net_bin,
        "path_economic_state": path_state,
        "valid_resolved_support": valid_resolved.astype(np.float32),
    }
    return {
        "label_context": label_context,
        "archetype_labels": archetypes,
        "audit": audit,
    }


def mda_reference_support_audit(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact serialisable support provenance for reference diagnostics."""
    raw = reference.get("archetype_label_audit")
    return dict(raw) if isinstance(raw, Mapping) else {}
