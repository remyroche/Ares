#!/usr/bin/env python3
"""Materialise strict prior-rule soft memberships on later OOF environments.

Rules for an evaluation environment are learned only from discovery folds that
precede it.  Each rule is evaluated in its own source fold's robust-standard
coordinate system, using the saved earlier-only centre/scale lineage.  The
resulting memberships overlap and never use realised event/outcome columns as
inputs; event is retained solely for later conditional-effect audits.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.align_transport_supervised_archetypes import build_consensus  # noqa: E402

ARTIFACT = ROOT / "data_perp/artifacts/transport_supervised_archetypes_20260803_v1"
LEDGER = ROOT / "data_perp/artifacts/tp6_m6_shared_regime_residual_features_20260809_v3/shared_regime_residual_ledger.parquet"
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3/parts/*.parquet"


def _folds(timestamp: pd.Series) -> pd.Series:
    values = pd.Index(timestamp.drop_duplicates().sort_values())
    lookup = {item: min(4, int(5 * position / max(len(values), 1))) for position, item in enumerate(values)}
    return timestamp.map(lookup).astype(np.int8)


def _flatten(definitions: dict) -> list[dict]:
    return [entry for side in definitions.get("definitions", {}).values() for head in side.values() for entry in head]


def _membership(frame: pd.DataFrame, definition: dict, scaler: pd.DataFrame) -> tuple[np.ndarray, float]:
    log_terms = []
    coverage = []
    source_fold = max(definition["fold_recurrence"]["recurring_folds"])
    for condition in definition["conditions"]:
        field = condition["feature"]
        lookup = scaler.loc[
            scaler.fold.eq(source_fold)
            & scaler.side_name.eq(definition["side_name"])
            & scaler["head"].eq(definition["event_head"])
            & scaler.feature.eq(field)
        ]
        if len(lookup) != 1:
            raise ValueError(f"missing/ambiguous prior scaler for {definition['archetype_id']} {field}")
        centre, scale = lookup.iloc[0][["center", "scale"]].astype(float)
        raw = pd.to_numeric(frame[field], errors="coerce").to_numpy(float)
        valid = np.isfinite(raw)
        coverage.append(valid.mean())
        z = np.where(valid, (raw - centre) / max(scale, 1e-12), 0.)
        signed = int(condition["direction"]) * (z - float(condition["threshold_robust_standard_units"]))
        temperature = float(condition["sigmoid_temperature_robust_standard_units"])
        probability = 1. / (1. + np.exp(-np.clip(signed / temperature, -30., 30.)))
        log_terms.append(np.log(np.clip(probability, 1e-12, 1.)))
    return np.exp(np.mean(np.vstack(log_terms), axis=0)).astype(np.float32), float(np.mean(coverage))


def run() -> None:
    candidates = pd.read_parquet(ARTIFACT / "archetype_rule_candidates.parquet")
    scaler = pd.read_parquet(ARTIFACT / "archetype_feature_scalers.parquet")
    # Folds 3 and 4 are scoreable from at least one fully prior discovery
    # fold.  Fold 2 has no earlier rule catalogue and deliberately has no
    # membership rows rather than a future-informed surrogate.
    definitions_by_fold = {}
    required_fields: set[str] = set()
    for evaluation_fold in (3, 4):
        _alignment, payload = build_consensus(
            candidates.loc[candidates.fold.lt(evaluation_fold)],
            maximum_definitions_per_group=3,
        )
        definitions = _flatten(payload)
        definitions_by_fold[evaluation_fold] = definitions
        required_fields.update(condition["feature"] for definition in definitions for condition in definition["conditions"])
    # This is the stable nested-evaluation contract: catalogue D2 is learned
    # before fold 2, then held fixed while fold 3 trains a residual classifier
    # and fold 4 evaluates it.  It is intentionally separate from the rolling
    # representation above, whose fold-specific catalogue is useful for
    # support/effect audit but not a common train/test feature schema.
    _frozen_alignment, frozen_payload = build_consensus(
        candidates.loc[candidates.fold.eq(2)], maximum_definitions_per_group=3,
    )
    frozen_definitions = _flatten(frozen_payload)
    required_fields.update(condition["feature"] for definition in frozen_definitions for condition in definition["conditions"])
    con = duckdb.connect(config={"threads": "2", "memory_limit": "512MB", "temp_directory": "/tmp"})
    raw_fields = ", ".join(f'p."{field}"' for field in sorted(required_fields))
    query = f'''SELECT l.candidate_id,l.__ts__,l.side_name,l.event,l.net_bps,l.gross_bps,l.p_adverse,l.p_weak,l.p_clear,l.prequential_base_expected_net_bps,
    p."atr_1h",p."decision_price",p."assumed_round_trip_cost_bps",{raw_fields}
    FROM read_parquet('{LEDGER.as_posix()}') l JOIN read_parquet('{PANEL.as_posix()}') p USING(candidate_id)
    WHERE l.shared_regime_contract_complete AND l.prequential_base_expected_net_bps IS NOT NULL AND abs(hash(l.candidate_id)) % 5 = 0'''
    frame = con.execute(query).fetchdf()
    con.close()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["fold"] = _folds(frame["__ts__"])
    atr_bps = np.abs(frame.atr_1h.to_numpy(float)) / np.maximum(np.abs(frame.decision_price.to_numpy(float)), 1e-12) * 1e4
    frame["cost_to_atr"] = np.clip(frame.assumed_round_trip_cost_bps.to_numpy(float) / np.maximum(atr_bps, 1e-6), 0., 100.)
    rows, dictionary, coverage_rows = [], [], []
    for evaluation_fold, definitions in definitions_by_fold.items():
        work = frame.loc[frame.fold.eq(evaluation_fold)].copy()
        for definition_version, definition_set in ((f"rolling_f{evaluation_fold}", definitions), ("frozen_d2", frozen_definitions)):
          for definition in definition_set:
            name = (f"oof_f{evaluation_fold}__{definition['archetype_id']}" if definition_version.startswith("rolling") else f"frozen_d2__{definition['archetype_id']}")
            values, coverage = _membership(work, definition, scaler)
            # A rule discovered inside a side/event conditional head remains
            # an inference-safe context field for *all* candidates of that
            # side. It does not see the realised event; that is audit-only.
            eligible_side = work.side_name.eq(definition["side_name"]).to_numpy()
            work[name] = np.where(eligible_side, values, 0.).astype(np.float32)
            if not any(item["membership_column"] == name for item in dictionary):
                dictionary.append({"evaluation_fold": evaluation_fold, "definition_version": definition_version, "membership_column": name, "source_discovery_folds": definition["fold_recurrence"]["recurring_folds"], **definition})
            coverage_rows.append({"evaluation_fold": evaluation_fold, "membership_column": name, "side_name": definition["side_name"], "event_head": definition["event_head"], "input_coverage": coverage, "rows": len(work), "effective_support": float(np.square(work[name].sum()) / max(np.square(work[name]).sum(), 1e-12))})
        rows.append(work)
    output = pd.concat(rows, ignore_index=True)
    membership_columns = [item["membership_column"] for item in dictionary]
    keep = ["candidate_id", "__ts__", "fold", "side_name", "event", "net_bps", "gross_bps", "p_adverse", "p_weak", "p_clear", "prequential_base_expected_net_bps", "cost_to_atr", *membership_columns]
    output.loc[:, keep].to_parquet(ARTIFACT / "archetype_soft_memberships_oof.parquet", index=False)
    pd.DataFrame(dictionary).to_parquet(ARTIFACT / "archetype_oof_membership_dictionary.parquet", index=False)
    pd.DataFrame(coverage_rows).to_parquet(ARTIFACT / "archetype_oof_membership_coverage.parquet", index=False)
    support_rows, effect_rows = [], []
    dictionary_by_column = {entry["membership_column"]: entry for entry in dictionary}
    for name in membership_columns:
        definition = dictionary_by_column[name]
        local = output.loc[output.fold.eq(definition["evaluation_fold"]) & output.side_name.eq(definition["side_name"])].copy()
        if local.empty:
            continue
        local["month"] = local.__ts__.dt.to_period("M").astype(str)
        for month, part in local.groupby("month", observed=True):
            value = part[name].to_numpy(float)
            support_rows.append({"evaluation_fold": definition["evaluation_fold"], "membership_column": name, "side_name": definition["side_name"], "event_head": definition["event_head"], "environment": month, "rows": len(part), "mean_membership": float(value.mean()), "effective_support": float(np.square(value.sum()) / max(np.square(value).sum(), 1e-12)), "support_at_05": int((value >= .5).sum()), "support_share_at_05": float((value >= .5).mean())})
        target_event = 1 if definition["event_head"] == "clear" else 0
        local = local.loc[local.event.eq(target_event)].copy()
        if len(local) < 100:
            continue
        local["base_bin"] = pd.qcut(local.prequential_base_expected_net_bps.rank(method="first"), 10, labels=False, duplicates="drop")
        local["clear_bin"] = pd.qcut(local.p_clear.rank(method="first"), 5, labels=False, duplicates="drop")
        local["adverse_bin"] = pd.qcut(local.p_adverse.rank(method="first"), 5, labels=False, duplicates="drop")
        local["cost_bin"] = pd.qcut(local.cost_to_atr.rank(method="first"), 5, labels=False, duplicates="drop")
        local["membership_bin"] = pd.qcut(local[name].rank(method="first"), 5, labels=False, duplicates="drop")
        differences, matched_rows = [], 0
        for _key, part in local.groupby(["base_bin", "clear_bin", "adverse_bin", "cost_bin"], observed=True):
            high, low = part.loc[part.membership_bin.eq(4)], part.loc[part.membership_bin.eq(0)]
            if len(high) >= 3 and len(low) >= 3:
                matched_rows += len(high) + len(low)
                differences.append((len(high) + len(low), float(high.net_bps.mean() - low.net_bps.mean()), float(high.gross_bps.mean() - low.gross_bps.mean())))
        if differences:
            weight = np.asarray([item[0] for item in differences], dtype=float)
            effect_rows.append({"evaluation_fold": definition["evaluation_fold"], "membership_column": name, "side_name": definition["side_name"], "event_head": definition["event_head"], "rows_event_conditioned": len(local), "matched_rows": matched_rows, "matched_strata": len(differences), "conditional_net_effect_high_minus_low_bps": float(np.average([item[1] for item in differences], weights=weight)), "conditional_gross_effect_high_minus_low_bps": float(np.average([item[2] for item in differences], weights=weight)), "matching": "fold × side × realised event head × base decile × p_clear quintile × p_adverse quintile × cost_to_atr quintile"})
    pd.DataFrame(support_rows).to_parquet(ARTIFACT / "archetype_support_by_environment.parquet", index=False)
    pd.DataFrame(effect_rows).to_parquet(ARTIFACT / "archetype_conditional_effects.parquet", index=False)
    (ARTIFACT / "archetype_oof_membership_manifest.json").write_text(json.dumps({"schema": "transport_archetype_oof_membership_v1", "row_proxy": "deterministic 20% candidate sample", "evaluation_folds": [3, 4], "no_fold_2_memberships": "no earlier discovery rule catalogue exists", "rule_definition": "strictly earlier discovery folds only", "normalisation": "source-fold earlier-only centre/IQR scaler", "membership": "independent geometric mean of directional sigmoid conditions; never a simplex", "event_usage": "realised event retained only for later effect audit; never an input", "definitions": len(dictionary), "promotion_status": "OOF_MEMBERSHIPS_MATERIALISED_PENDING_SUPPORT_EFFECT_TRANSPORT_AND_MDA_GATES"}, indent=2) + "\n")


if __name__ == "__main__":
    run()
