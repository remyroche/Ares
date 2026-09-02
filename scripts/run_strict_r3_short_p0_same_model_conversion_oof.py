#!/usr/bin/env python3
"""Produce strict-OOS short P0/F90 conversion scores with same-model reserves.

This is the short counterpart of the long conversion stage.  For each held
month it uses the already sealed P0/F90 base artifact for that month to score
both the prior 28 calendar days and the held month.  Promoted residual heads,
when present, are fitted only on rows resolved before that reserve.  When the
strict ensemble gate rejects every head, an explicitly requested base-only
fallback retains P0/F90 and frozen Geometry/K9 but declares BCF unavailable.
A correctness ranker is then fitted on earlier scored rows and the held
model's 28-day reserve supplies the final-score CDF domain.  Thus no
held-window percentile, post-date prediction, or future label can enter
``final_score``.

The HPO/CMI selection period is deliberately retained as development-only.
The emitted ``selection_eligible`` flag is false before ``--selection-end``;
downstream promotion and economics must filter it true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import Booster

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    BASE_BLEND_WEIGHT,
    CONSENSUS_BLEND_WEIGHT,
    CORRECTNESS_FLOOR,
    CORRECTNESS_SPAN,
    _aggregate_state_fields,
    _fit_correctness,
    _fit_severe_diagnostic,
    _numeric_matrix,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    ScoreReference,
    load_geometry_bundle,
)
from scripts.run_strict_r3_short_p0_consensus_oof import (  # noqa: E402
    SIDE,
    _head_specs,
    _load_ledger,
    _month_range,
    _residual_grade,
    _valid_base_rows,
    _valid_residual_rows,
    _with_geometry,
)
from extreme_price_movements.strict_r3_canonical_current import _fit_consensus_head  # noqa: E402


REFERENCE_DAYS = 28
ROUTE_FRACTION = 0.30


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    targets = [path] if path.is_file() else sorted(value for value in path.rglob("*") if value.is_file())
    for target in targets:
        digest.update(str(target.relative_to(path) if path.is_dir() else target.name).encode())
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _base_artifact(roots: list[Path], month: pd.Timestamp) -> Path:
    paths = [root / "fold_models" / f"month={month:%Y-%m}" for root in roots]
    found = [path for path in paths if path.is_dir()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected exactly one P0/F90 base artifact for {month:%Y-%m}; found={found}"
        )
    required = ("base_model.txt", "base_preprocess.json", "prior42_base_scores.npy", "policy_net_map.joblib")
    missing = [name for name in required if not (found[0] / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete P0/F90 artifact {found[0]}: {missing}")
    return found[0]


def _score_base_same_model(frame: pd.DataFrame, artifact: Path) -> pd.DataFrame:
    preprocess = json.loads((artifact / "base_preprocess.json").read_text())
    fields = tuple(map(str, preprocess["ordered_fields"]))
    medians = np.asarray([float(preprocess["training_medians"][field]) for field in fields], dtype=np.float32)
    if len(fields) != 90 or len(set(fields)) != 90:
        raise ValueError("P0/F90 base artifact does not expose the frozen F90 contract")
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise KeyError(f"P0/F90 reference frame lacks base fields: {missing[:10]}")
    model = Booster(model_file=str(artifact / "base_model.txt"))
    reference = ScoreReference.fit(
        np.load(artifact / "prior42_base_scores.npy"),
        source=f"{artifact.name}_same_base_model_prior42",
    )
    policy_map = joblib.load(artifact / "policy_net_map.joblib")
    output = frame.copy()
    eligible = output["base_feature_eligible"].fillna(False).astype(bool).to_numpy()
    raw = np.full(len(output), np.nan, dtype=np.float32)
    rank = np.full(len(output), np.nan, dtype=np.float32)
    anchor = np.full(len(output), np.nan, dtype=np.float32)
    if eligible.any():
        matrix = _numeric_matrix(output.loc[eligible], fields, medians)
        current = np.asarray(model.predict(matrix), dtype=np.float32)
        raw[eligible] = current
        rank[eligible] = reference.cdf(current).astype(np.float32)
        anchor[eligible] = np.asarray(policy_map.predict(rank[eligible]), dtype=np.float32)
    output["base_score"] = raw
    output["base_rank42"] = rank
    output["base_anchor_bps"] = anchor
    return output


def _route_mask(frame: pd.DataFrame) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "base_rank42"]].copy()
    work["__position__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", "base_rank42", "candidate_id"],
        ascending=[True, False, True], kind="stable", na_position="last",
    )
    rank = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy()
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy()
    work["__route__"] = rank < np.maximum(1, np.ceil(count * ROUTE_FRACTION).astype(int))
    return work.sort_values("__position__", kind="stable")["__route__"].to_numpy(bool)


def _metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = frame.loc[
        frame["selection_eligible"].fillna(False)
        & frame["policy_path_valid"].fillna(False)
        & frame["final_score"].notna()
        & frame["policy_net_bps"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame(rows)
    valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
    for month, group in [("pooled", valid), *valid.groupby("month", sort=True)]:
        for tail in (0.01, 0.02, 0.05):
            threshold = group["final_score"].quantile(1.0 - tail, interpolation="higher")
            chosen = group.loc[group["final_score"].ge(threshold)]
            rows.append({
                "month": month, "tail": tail, "rows": int(len(chosen)),
                "net_bps_per_trade": float(chosen["policy_net_bps"].mean()),
                "total_net_bps": float(chosen["policy_net_bps"].sum()),
            })
    return pd.DataFrame(rows)


def _selected_columns(
    selector_manifest: Path | None,
    contract: dict[str, Any],
    *,
    allow_base_only: bool = False,
) -> tuple[str, ...]:
    if selector_manifest is None:
        if allow_base_only and not contract.get("heads"):
            return ()
        raise ValueError("selector OOF manifest is required when residual heads are present")
    manifest = json.loads(selector_manifest.read_text())
    selected = tuple(map(str, manifest.get("ensemble_selection", {}).get("selected_rank_columns", ())))
    available = {f"head__{head['name']}__rank" for head in contract["heads"]}
    if allow_base_only and not available and not selected:
        return ()
    if not selected or not set(selected).issubset(available):
        raise ValueError("selector OOF manifest is not compatible with the promoted short head contract")
    return selected


def run(
    *, ledger_roots: list[Path], contract_path: Path, selector_manifest: Path | None,
    geometry_dir: Path, start: pd.Timestamp, end: pd.Timestamp,
    selection_end: pd.Timestamp, history_start: pd.Timestamp, out: Path,
    allow_base_only: bool = False,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    contract = json.loads(contract_path.read_text())
    contract_sha256 = _sha(contract_path)
    no_head_contract = (
        contract.get("schema") == "strict_r3_short_p0_cmi_consensus_v2"
        and contract.get("side") == SIDE
        and contract.get("heads") == []
    )
    # `_head_specs` deliberately rejects an empty v2 contract for a genuine
    # consensus fit.  Conversion is the one valid consumer that can continue
    # after that strict gate, but only through this explicit base-only mode.
    heads = () if allow_base_only and no_head_contract else _head_specs(contract)
    if not heads and not allow_base_only:
        raise ValueError(
            "same-model conversion requires at least one promoted residual head; "
            "pass --allow-base-only only for the explicit fail-closed base-only fallback"
        )
    selected_columns = _selected_columns(
        selector_manifest, contract, allow_base_only=allow_base_only,
    )
    geometry = load_geometry_bundle(geometry_dir)
    if geometry.bundle_sha256 != contract["geometry"]["bundle_sha256"]:
        raise ValueError("same-model conversion geometry differs from frozen short head contract")
    ledger = pd.concat(
        [_load_ledger(root, minimum_month=history_start) for root in ledger_roots],
        ignore_index=True,
    )
    if ledger["candidate_id"].duplicated().any():
        raise ValueError("same-model conversion ledger roots overlap")
    for column in ("__decision_ts__", "policy_label_available_at", "h12_label_available_ts"):
        ledger[column] = pd.to_datetime(ledger[column], utc=True, errors="raise")
    if not ledger["side_name"].astype(str).str.lower().eq(SIDE).all():
        raise ValueError("same-model conversion source is not short-local")
    required = sorted({field for head in heads for field in head.fields})
    ledger = _with_geometry(ledger, geometry, required)
    ledger["policy_net_bps"] = pd.to_numeric(ledger["p0_canonical_net_bps"], errors="coerce")
    ledger["policy_label_available_ts"] = ledger["policy_label_available_at"]
    ledger["policy_residual_bps"] = (
        pd.to_numeric(ledger["p0_canonical_net_bps"], errors="coerce")
        - pd.to_numeric(ledger["prequential_base_anchor_bps"], errors="coerce")
    )
    aggregate = _aggregate_state_fields(geometry.transform(ledger.iloc[:1]))
    # Geometry was already added above; enforce that raw K9 membership fields
    # never reach correctness/trust through the aggregate contract.
    if any(field.startswith("k09__cluster_") for field in aggregate):
        raise AssertionError("raw K9 membership survived the aggregate filter")
    correct_fields = (
        "base_score", "base_anchor_bps", "base_rank42",
        "conditional_consensus_rank", "upstream", *aggregate,
    )
    mode = {str(head["name"]): str(head["weight_mode"]) for head in contract["heads"]}
    all_head_columns = tuple(f"head__{head.name}__rank" for head in heads)
    all_ordinary_columns = tuple(
        column for column in all_head_columns
        if mode[column.removeprefix("head__").removesuffix("__rank")] == "ordinary"
    )
    ordinary_selected = tuple(
        column for column in selected_columns
        if mode[column.removeprefix("head__").removesuffix("__rank")] == "ordinary"
    )
    ordinary_columns = ordinary_selected or selected_columns
    # BCF is a distinct all-promoted-head family.  A one-head model has no
    # agreement geometry, so it is explicitly unavailable rather than being
    # dressed up as a BCF ensemble.
    bcf_enabled = len(all_head_columns) >= 2 and bool(all_ordinary_columns)
    bcf_fields = (
        "base_score", "base_anchor_bps", "base_rank42",
        "bcf_consensus_rank", "bcf_upstream", *aggregate,
    )
    out.mkdir(parents=True)
    history_parts: list[pd.DataFrame] = []
    held_parts: list[pd.DataFrame] = []
    reference_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for fold_index, held_start in enumerate(_month_range(start, end)):
        held_end = held_start + pd.offsets.MonthBegin(1)
        reserve_start = held_start - pd.Timedelta(days=REFERENCE_DAYS)
        artifact = _base_artifact(ledger_roots, held_start)
        artifact_sha256 = _sha(artifact)
        producer_hash = hashlib.sha256(
            (
                "short_p0_same_model_conversion_v1|"
                + artifact_sha256 + "|" + contract_sha256 + "|"
                + geometry.bundle_sha256 + "|" + ",".join(selected_columns)
            ).encode()
        ).hexdigest()
        train = ledger.loc[
            ledger["__decision_ts__"].lt(reserve_start)
            & _valid_residual_rows(ledger, reserve_start)
        ].copy()
        reference = ledger.loc[
            ledger["__decision_ts__"].ge(reserve_start)
            & ledger["__decision_ts__"].lt(held_start)
            & _valid_base_rows(ledger)
        ].copy()
        held = ledger.loc[
            ledger["__decision_ts__"].ge(held_start)
            & ledger["__decision_ts__"].lt(held_end)
            & _valid_base_rows(ledger)
        ].copy()
        if len(train) < 1_000 or reference.empty or held.empty:
            audits.append({"month": held_start.strftime("%Y-%m"), "status": "skipped_insufficient_base_or_residual_support", "train_rows": len(train), "reference_rows": len(reference), "held_rows": len(held)})
            continue
        pair = pd.concat([reference.assign(__role__="reference"), held.assign(__role__="held")], ignore_index=True)
        pair = _score_base_same_model(pair, artifact)
        for head_index, spec in enumerate(heads):
            grade = _residual_grade(train["policy_residual_bps"], spec.target_edges_bps)
            fitted = _fit_consensus_head(train, grade, spec, seed=20260821 + fold_index * 100 + head_index)
            raw, rank = fitted.predict_rank(pair)
            pair[f"head__{spec.name}__raw"] = raw
            pair[f"head__{spec.name}__rank"] = rank
            model_dir = out / "fold_models" / f"month={held_start:%Y-%m}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted, model_dir / f"{spec.name}.joblib", compress=3)
        if heads:
            pair["bcf_consensus_rank"] = np.nanmedian(
                pair.loc[:, all_head_columns].to_numpy(float), axis=1,
            ).astype(np.float32)
            if all_ordinary_columns:
                pair["bcf_ordinary_shadow_consensus_rank"] = np.nanmedian(
                    pair.loc[:, all_ordinary_columns].to_numpy(float), axis=1,
                ).astype(np.float32)
            else:
                pair["bcf_ordinary_shadow_consensus_rank"] = np.nan
            pair["bcf_upstream"] = (
                BASE_BLEND_WEIGHT * pair["base_rank42"].to_numpy(float)
                + CONSENSUS_BLEND_WEIGHT * pair["bcf_consensus_rank"].to_numpy(float)
            ).astype(np.float32)
            pair["conditional_consensus_rank"] = np.nanmedian(
                pair.loc[:, selected_columns].to_numpy(float), axis=1,
            ).astype(np.float32)
            pair["ordinary_shadow_consensus_rank"] = np.nanmedian(
                pair.loc[:, ordinary_columns].to_numpy(float), axis=1,
            ).astype(np.float32)
            pair["ordinary_shadow_fallback_to_selected"] = not bool(ordinary_selected)
            pair["upstream"] = (
                BASE_BLEND_WEIGHT * pair["base_rank42"].to_numpy(float)
                + CONSENSUS_BLEND_WEIGHT * pair["conditional_consensus_rank"].to_numpy(float)
            ).astype(np.float32)
            pair["ordinary_shadow_upstream"] = (
                BASE_BLEND_WEIGHT * pair["base_rank42"].to_numpy(float)
                + CONSENSUS_BLEND_WEIGHT * pair["ordinary_shadow_consensus_rank"].to_numpy(float)
            ).astype(np.float32)
        else:
            # The strict development gate rejected every residual head.  Keep
            # the downstream clock and Geometry/K9 state available, but make
            # the absence of a consensus explicit: P0/F90 is the complete
            # upstream score and BCF is unavailable rather than fabricated.
            pair["conditional_consensus_rank"] = pair["base_rank42"].to_numpy(np.float32)
            pair["ordinary_shadow_consensus_rank"] = pair["base_rank42"].to_numpy(np.float32)
            pair["ordinary_shadow_fallback_to_selected"] = False
            pair["upstream"] = pair["base_rank42"].to_numpy(np.float32)
            pair["ordinary_shadow_upstream"] = pair["base_rank42"].to_numpy(np.float32)
            pair["bcf_consensus_rank"] = np.nan
            pair["bcf_ordinary_shadow_consensus_rank"] = np.nan
            pair["bcf_upstream"] = np.nan
        pair["base_route_timestamp_top30"] = _route_mask(pair)
        pair["base_route_fraction"] = ROUTE_FRACTION
        resolved_history = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
        resolved_history = resolved_history.loc[
            resolved_history["policy_label_available_ts"].lt(reserve_start)
            & resolved_history["policy_path_valid"].fillna(False)
            & resolved_history["policy_net_bps"].notna()
        ].copy() if not resolved_history.empty else resolved_history
        status = "complete"
        if len(resolved_history) < 1_000:
            status = "scored_no_correctness_history"
            pair["correctness_raw"] = np.nan
            pair["correctness_rank"] = np.nan
            pair["correctness_gate_active"] = False
            pair["raw_correctness_demote"] = np.nan
            pair["final_score"] = np.nan
            pair["severe200_probability_shadow"] = np.nan
        else:
            correctness = _fit_correctness(resolved_history, correct_fields)
            raw = np.full(len(pair), np.nan, dtype=float)
            routed = pair["base_route_timestamp_top30"].to_numpy(bool)
            if routed.any():
                raw[routed] = correctness.model.predict(
                    _numeric_matrix(pair.loc[routed], correctness.fields, correctness.medians),
                )
            pair["correctness_raw"] = raw
            pair["correctness_rank"] = np.where(
                routed, correctness.score_reference.cdf(raw), np.nan,
            )
            gate = routed & pair["upstream"].ge(correctness.training_score_floor).to_numpy(bool)
            pair["correctness_gate_active"] = gate
            multiplier = CORRECTNESS_FLOOR + CORRECTNESS_SPAN * pair["correctness_rank"].to_numpy(float)
            pair["raw_correctness_demote"] = np.where(
                routed,
                pair["upstream"].to_numpy(float) * np.where(gate, multiplier, 1.0),
                np.nan,
            )
            reference_values = pair.loc[pair["__role__"].eq("reference"), "raw_correctness_demote"].dropna().to_numpy(float)
            if len(reference_values) < 2:
                status = "scored_no_complete_same_model_reference"
                pair["final_score"] = np.nan
            else:
                cdf = ScoreReference.fit(reference_values, source=f"{held_start:%Y-%m}_short_same_model_prior28_correctness")
                pair["final_score"] = cdf.cdf(pair["raw_correctness_demote"].to_numpy(float))
            severe = _fit_severe_diagnostic(resolved_history, correct_fields, cutoff=reserve_start)
            probability = np.full(len(pair), np.nan, dtype=float)
            if severe.model is not None and routed.any():
                probability[routed] = severe.model.predict_proba(
                    _numeric_matrix(pair.loc[routed], severe.fields, severe.medians),
                )[:, 1]
            pair["severe200_probability_shadow"] = probability
        pair["severe_affects_final_score"] = False
        pair["bcf_severe200_probability"] = np.nan
        pair["bcf_raw_severe"] = np.nan
        pair["bcf_final_score"] = np.nan
        pair["bcf_score_available"] = False
        if bcf_enabled and len(resolved_history) >= 1_000:
            bcf_severe = _fit_severe_diagnostic(
                resolved_history, bcf_fields, cutoff=reserve_start,
            )
            if bcf_severe.model is not None:
                bcf_probability = bcf_severe.model.predict_proba(
                    _numeric_matrix(pair, bcf_severe.fields, bcf_severe.medians),
                )[:, 1]
                pair["bcf_severe200_probability"] = bcf_probability
                pair["bcf_raw_severe"] = (
                    pair["bcf_upstream"].to_numpy(float)
                    * (1.0 - 0.5 * bcf_probability)
                )
                bcf_reference = pair.loc[
                    pair["__role__"].eq("reference"), "bcf_raw_severe"
                ].dropna().to_numpy(float)
                if len(bcf_reference) >= 2:
                    bcf_cdf = ScoreReference.fit(
                        bcf_reference,
                        source=f"{held_start:%Y-%m}_short_bcf_same_model_prior28_raw_severe",
                    )
                    pair["bcf_final_score"] = bcf_cdf.cdf(
                        pair["bcf_raw_severe"].to_numpy(float),
                    )
                    pair["bcf_score_available"] = pair["bcf_final_score"].notna()
        pair["conversion_reference_days"] = REFERENCE_DAYS
        pair["conversion_base_artifact"] = str(artifact)
        pair["conversion_base_artifact_sha256"] = artifact_sha256
        pair["conversion_held_month"] = held_start.strftime("%Y-%m")
        # These identifiers are carried on both roles.  The reserve role is
        # explicitly out of every active supervised fit: P0 was fitted before
        # its 42d reserve; residual/correctness/Severe fits stop before the
        # 28d conversion reserve.  A later map may therefore use exactly this
        # producer's reserve from its first held hour without a vintage bridge.
        pair["conversion_bundle_sha256"] = producer_hash
        pair["upstream_bundle_sha256"] = artifact_sha256
        # The current selected-consensus score remains one normalized semantic
        # family across monthly refits.  Exact-reserve maps still segregate
        # producers via ``conversion_bundle_sha256`` below; conflating the two
        # would incorrectly fragment the frozen MC1 recent-global shift.
        pair["ev_score_family_id"] = (
            f"short_p0_current_selected_v1:{geometry.bundle_sha256}"
        )
        pair["calibration_activation_ts"] = held_start
        pair["calibration_reference_oos_to_all_active_fits"] = pair["__role__"].eq("reference")
        pair["geometry_bundle_sha256"] = geometry.bundle_sha256
        pair["selection_eligible"] = pair["__decision_ts__"].ge(selection_end)
        pair["stack_status"] = status
        pair["policy_label_available_ts"] = pd.to_datetime(pair["policy_label_available_ts"], utc=True)
        history_parts.append(pair.loc[pair["__role__"].eq("held")].drop(columns="__role__").copy())
        held_parts.append(pair.loc[pair["__role__"].eq("held")].drop(columns="__role__").copy())
        # The prior-28d score reserve has a deliberately different identity
        # namespace: a candidate can be re-scored by a later monthly bundle.
        # It is persisted separately rather than merged into the held OOS
        # ledger, preventing duplicate candidate IDs or cross-producer maps.
        reference_parts.append(pair.loc[pair["__role__"].eq("reference")].copy())
        audits.append({
            "month": held_start.strftime("%Y-%m"), "status": status,
            "train_rows": len(train), "reference_rows": len(reference), "held_rows": len(held),
            "correctness_history_rows": len(resolved_history),
            "same_model_base_reference": True,
            "same_model_correctness_reference": status == "complete",
            "selection_eligible": bool(held_start >= selection_end),
        })
    if not held_parts:
        raise ValueError("short same-model conversion produced no held rows")
    output = pd.concat(held_parts, ignore_index=True)
    output.to_parquet(out / "short_same_model_conversion_oof_predictions.parquet", index=False, compression="zstd")
    references = pd.concat(reference_parts, ignore_index=True)
    references.to_parquet(
        out / "short_same_model_conversion_reference_scores.parquet",
        index=False, compression="zstd",
    )
    policy_outcomes = ledger.loc[:, [
        "candidate_id", "side_name", "policy_path_valid", "p0_canonical_net_bps",
        "policy_label_available_at",
    ]].rename(columns={
        "p0_canonical_net_bps": "policy_net_bps",
        "policy_label_available_at": "policy_label_available_ts",
    })
    if policy_outcomes["candidate_id"].duplicated().any():
        raise AssertionError("short conversion policy outcome source must retain unique identities")
    policy_outcomes.to_parquet(
        out / "short_policy_outcomes_source.parquet",
        index=False, compression="zstd",
    )
    bcf_rank_contract = {
        "schema": "strict_r3_short_bcf_promoted_head_contract_v1",
        "side": SIDE,
        "rank_fields": list(all_head_columns),
        "ordinary_rank_fields": list(all_ordinary_columns),
        "source": (
            "all promoted HPO heads; never inferred from held rows"
            if heads else "unavailable: no short residual head passed development gate"
        ),
        "enabled": bool(bcf_enabled),
    }
    (out / "short_bcf_promoted_head_contract.json").write_text(
        json.dumps(bcf_rank_contract, indent=2) + "\n",
    )
    bcf_columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "base_score", "base_rank42", "base_anchor_bps",
        "bcf_consensus_rank", "bcf_ordinary_shadow_consensus_rank", "bcf_upstream",
        "bcf_severe200_probability", "bcf_raw_severe", "bcf_final_score",
        "bcf_score_available", "base_route_timestamp_top30", "selection_eligible",
        "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
        "conversion_base_artifact", "conversion_base_artifact_sha256",
        "conversion_held_month", "conversion_bundle_sha256", "upstream_bundle_sha256",
        "calibration_activation_ts", "calibration_reference_oos_to_all_active_fits",
        "stack_is_prequential",
        *all_head_columns,
    ]
    bcf_output = output.loc[:, bcf_columns].rename(columns={
        "bcf_consensus_rank": "consensus_rank",
        "bcf_ordinary_shadow_consensus_rank": "ordinary_shadow_consensus_rank",
        "bcf_upstream": "upstream",
        "bcf_severe200_probability": "severe200_probability",
        "bcf_raw_severe": "raw_severe",
        "bcf_final_score": "final_score",
    })
    bcf_reference = references.loc[:, bcf_columns].rename(columns={
        "bcf_consensus_rank": "consensus_rank",
        "bcf_ordinary_shadow_consensus_rank": "ordinary_shadow_consensus_rank",
        "bcf_upstream": "upstream",
        "bcf_severe200_probability": "severe200_probability",
        "bcf_raw_severe": "raw_severe",
        "bcf_final_score": "final_score",
    })
    for frame in (bcf_output, bcf_reference):
        frame["ev_score_family_id"] = (
            "short_p0_bcf_all_promoted_v1:" + geometry.bundle_sha256
        )
    bcf_output.to_parquet(
        out / "short_bcf_score_family_oof_predictions.parquet",
        index=False, compression="zstd",
    )
    bcf_reference.to_parquet(
        out / "short_bcf_score_family_reference_scores.parquet",
        index=False, compression="zstd",
    )
    pd.DataFrame(audits).to_parquet(out / "conversion_fold_audit.parquet", index=False, compression="zstd")
    _metric_rows(output).to_parquet(out / "conversion_tail_metrics.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_p0_same_model_conversion_oof_v1",
        "status": "complete", "side": SIDE,
        "held_window": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "selection_end": selection_end.isoformat(),
        "selection_eligible_only_for_promotion": True,
        "base": "saved held-month P0/F90 artifact scores both prior-28d reserve and held rows",
        "same_model_reference": "short_same_model_conversion_reference_scores.parquet; candidate IDs may recur only across distinct conversion_held_month producers",
        "consensus": (
            "promoted HPO heads fit on rows resolved before the prior-28d reserve"
            if heads else "explicit base-only fallback; no residual head passed development gate"
        ),
        "correctness": "top-30% base-routed LambdaRank correctness demotion fit only on earlier OOS scored rows",
        "final_score": "same-model prior-28d CDF of correctness-demoted upstream",
        "severe": "short H12 severe diagnostic only; does not affect final_score",
        "bcf": {
            "enabled": bool(bcf_enabled),
            "definition": (
                "all-promoted-head median upstream times causal Severe-200 demotion, "
                "normalized by same-model prior-28d raw-Severe CDF"
            ),
            "head_contract": "short_bcf_promoted_head_contract.json",
            "unavailable_reason": (
                None if bcf_enabled else "fewer than two promoted heads or no ordinary promoted head"
            ),
        },
        "geometry": {"bundle_sha256": geometry.bundle_sha256, "monthly_refit": False, "raw_k9_memberships": False},
        "source_hashes": {
            "hpo_contract": _sha(contract_path),
            "selector_manifest": (
                _sha(selector_manifest) if selector_manifest is not None else None
            ),
            "geometry": _sha(geometry_dir / "run_manifest.json"),
            "ledgers": {str(root): _sha(root / "run_manifest.json") for root in ledger_roots},
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--selector-oof-manifest", type=Path)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--history-start", default="2024-04-01T00:00:00Z")
    parser.add_argument("--start", default="2025-04-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-08-01T00:00:00Z")
    parser.add_argument("--selection-end", default="2025-07-01T00:00:00Z")
    parser.add_argument(
        "--allow-base-only", action="store_true",
        help="Permit the explicit base-only fallback when the frozen HPO contract has no promoted heads.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        ledger_roots=args.ledger_root, contract_path=args.contract,
        selector_manifest=args.selector_oof_manifest, geometry_dir=args.geometry_dir,
        history_start=_utc(args.history_start), start=_utc(args.start),
        end=_utc(args.end_exclusive), selection_end=_utc(args.selection_end), out=args.out,
        allow_base_only=bool(args.allow_base_only),
    ))


if __name__ == "__main__":
    main()
