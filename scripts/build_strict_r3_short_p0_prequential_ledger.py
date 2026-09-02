#!/usr/bin/env python3
"""Build the strict-prequential P0/F90 short base ledger.

Each held month is scored by a P0/F90 LambdaRank model fit only before its
same-model 42-day reserve.  The reserve provides both the rank domain and an
out-of-fit policy-net map.  Candidate rows are target-free; outcomes are
joined only after the base score is produced.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    BASE_TRAIN_CAP,
    ScoreReference,
    fit_policy_net_map,
)
from scripts.run_short_policy_conversion_funnel import (  # noqa: E402
    PolicySpec,
    _fit,
    _query_order,
    _targets,
    _valid_policy,
)


SIDE = "short"
REFERENCE_DAYS = 42
# This is a point-in-time eligibility gate, not an imputation rule.  A row
# without enough of the frozen F90 inputs is retained in the target-free
# population and explicitly rejected before the base model; it must never be
# silently converted into an all-median prediction.  The same predicate is
# used for fit, same-model rank reference, map support, and held scoring.
BASE_FEATURE_MIN_FRACTION = 0.90
P0_SPEC = PolicySpec(
    "P1_policy_bps", "Frozen P0/F90 policy-bps relevance.", "policy_bps",
    truncation=32, gain_family="linear", query_hours=1,
    weight_kind="uniform",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(value for value in path.rglob("*") if value.is_file())
    for value in paths:
        digest.update(str(value.relative_to(path) if path.is_dir() else value.name).encode())
        with value.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _months(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[pd.Timestamp]:
    values = list(pd.date_range(start.normalize().replace(day=1), end_exclusive, freq="MS", inclusive="left"))
    if not values:
        raise ValueError("ledger requires at least one held month")
    return values


def _selection_fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    size = int(payload.get("recommended_feature_size_development_only", 90))
    fields = [str(value) for value in payload.get("feature_sets", {}).get(str(size), [])]
    if len(fields) != 90 or len(fields) != len(set(fields)):
        raise ValueError("P0/F90 selection must resolve to exactly 90 unique fields")
    return fields


def _candidate_month(paths: list[Path], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "entry_executable", "eligibility_reason",
    ]
    frames = [pd.read_parquet(path, columns=columns, filters=[("__ts__", ">=", start), ("__ts__", "<", end)]) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    for column in ("__ts__", "__decision_ts__"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if result.candidate_id.duplicated().any():
        raise ValueError(f"candidate source overlap for {start:%Y-%m}")
    if not result.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("P0 ledger candidate source contains a non-short row")
    if result.entry_executable.isna().any():
        raise ValueError("P0 ledger candidate source has null entry eligibility")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _feature_month(paths: list[Path], fields: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = ["__ts__", "__symbol__", *fields]
    pieces: list[pd.DataFrame] = []
    for precedence, path in enumerate(paths):
        current = pd.read_parquet(path, columns=columns, filters=[("__ts__", ">=", start), ("__ts__", "<", end)])
        current["__ts__"] = pd.to_datetime(current["__ts__"], utc=True, errors="raise")
        current["__feature_source_precedence__"] = precedence
        pieces.append(current)
    result = pd.concat(pieces, ignore_index=True)
    return (
        result.sort_values(["__ts__", "__symbol__", "__feature_source_precedence__"], kind="stable")
        .drop_duplicates(["__ts__", "__symbol__"], keep="last")
        .drop(columns="__feature_source_precedence__")
        .reset_index(drop=True)
    )


def _part(root: Path, month: pd.Timestamp) -> Path | None:
    path = root / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
    return path if path.exists() else None


def _policy_month(paths: list[Path], month: pd.Timestamp) -> pd.DataFrame:
    candidates = [path for root in paths if (path := _part(root, month)) is not None]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one short policy partition for {month:%Y-%m}; found={candidates}")
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "__label_available_at__", "label_valid", "target_invalid",
        "policy_path_valid", "policy_label_available_at",
        "p0_canonical_gross_bps", "p0_canonical_net_bps",
    ]
    result = pd.read_parquet(candidates[0], columns=columns)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__", "policy_label_available_at"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    if result.candidate_id.duplicated().any() or not result.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError(f"invalid policy identities for {month:%Y-%m}")
    return result


def _h12_month(paths: list[Path], month: pd.Timestamp) -> pd.DataFrame:
    candidates = [path for root in paths if (path := _part(root, month)) is not None]
    if not candidates:
        return pd.DataFrame(columns=[
            "candidate_id", "h12_label_available_ts", "h12_label_valid",
            "h12_tp6_sl4_gross_bps", "h12_tp6_sl4_net_bps",
        ])
    if len(candidates) != 1:
        raise ValueError(f"exact H12 label source overlap for {month:%Y-%m}")
    columns = [
        "candidate_id", "__label_available_at__", "label_valid", "target_invalid",
        "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
    ]
    result = pd.read_parquet(candidates[0], columns=columns)
    result["h12_label_available_ts"] = pd.to_datetime(result.pop("__label_available_at__"), utc=True, errors="raise")
    valid = result.pop("label_valid").fillna(False).astype(bool) & ~result.pop("target_invalid").fillna(True).astype(bool)
    result["h12_label_valid"] = valid
    result = result.rename(columns={
        "t4_tp6_sl4_gross_bps": "h12_tp6_sl4_gross_bps",
        "t4_tp6_sl4_net_bps": "h12_tp6_sl4_net_bps",
    })
    for column in ("h12_tp6_sl4_gross_bps", "h12_tp6_sl4_net_bps"):
        result[column] = pd.to_numeric(result[column], errors="coerce").where(valid)
    if result.candidate_id.duplicated().any():
        raise ValueError(f"duplicate exact H12 identities for {month:%Y-%m}")
    return result


def _merge_month(
    *, candidates: pd.DataFrame, features: pd.DataFrame, policy: pd.DataFrame,
    h12: pd.DataFrame, fields: list[str],
) -> pd.DataFrame:
    result = candidates.merge(features, on=["__ts__", "__symbol__"], how="left", validate="one_to_one")
    if len(result) != len(candidates):
        raise AssertionError("feature join changed target-free candidate identities")
    result = result.merge(
        policy,
        on=["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"],
        how="left", validate="one_to_one",
    )
    if len(result) != len(candidates):
        raise AssertionError("policy join changed target-free candidate identities")
    result = result.merge(h12, on="candidate_id", how="left", validate="one_to_one")
    # The frozen P0 helper always reads its H12 diagnostic columns even when
    # the selected P1 policy target does not use them.  Keep the source-panel
    # canonical names and provide read-only aliases at this adapter boundary.
    result["t4_tp6_sl4_gross_bps"] = result["h12_tp6_sl4_gross_bps"]
    result["t4_tp6_sl4_net_bps"] = result["h12_tp6_sl4_net_bps"]
    result["geometry_definition_population_complete"] = True
    values = result[fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result["base_feature_available_fraction"] = values.notna().mean(axis=1).astype(np.float32)
    result["base_feature_eligible"] = (
        result.entry_executable.fillna(False).astype(bool)
        & result.base_feature_available_fraction.ge(BASE_FEATURE_MIN_FRACTION)
    )
    result["base_feature_rejection_reason"] = np.where(
        result.entry_executable.fillna(False).astype(bool)
        & ~result.base_feature_eligible,
        f"frozen_f90_fraction_below_{BASE_FEATURE_MIN_FRACTION:.2f}",
        None,
    )
    return result


def _base_fit_rows(frame: pd.DataFrame, reserve_start: pd.Timestamp) -> pd.DataFrame:
    usable = (
        frame.base_feature_eligible.fillna(False).astype(bool)
        & frame.policy_label_available_at.lt(reserve_start)
        & frame.policy_path_valid.fillna(False).astype(bool)
        & pd.to_numeric(frame.p0_canonical_net_bps, errors="coerce").notna()
    )
    return frame.loc[usable].sort_values(["policy_label_available_at", "candidate_id"], kind="stable").tail(BASE_TRAIN_CAP).copy()


def _assert_train_coverage(frame: pd.DataFrame, fields: list[str]) -> dict[str, float]:
    population = frame.loc[frame.base_feature_eligible.fillna(False).astype(bool), fields]
    finite = population.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().mean()
    if (finite < .90).any():
        failed = finite.loc[finite < .90].sort_values().head(10).to_dict()
        raise ValueError(f"frozen F90 coverage gate failed: {failed}")
    return {str(key): float(value) for key, value in finite.items()}


def _base_score_eligible(frame: pd.DataFrame) -> pd.Series:
    """Return the target-free feature gate shared by base fit and scoring."""
    return frame.base_feature_eligible.fillna(False).astype(bool)


def _model_artifacts(
    *, directory: Path, model: Any, audit: dict[str, Any],
    reference_score: np.ndarray, policy_map: Any, fields: list[str], medians: pd.Series,
) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    model.booster_.save_model(str(directory / "base_model.txt"))
    joblib.dump(policy_map, directory / "policy_net_map.joblib", compress=3)
    np.save(directory / "prior42_base_scores.npy", reference_score.astype(np.float32))
    (directory / "base_preprocess.json").write_text(json.dumps({
        "ordered_fields": fields,
        "training_medians": {field: float(medians.loc[field]) for field in fields},
    }, indent=2) + "\n")
    (directory / "base_fit_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")


def run(
    *, candidates: list[Path], features: list[Path], policy_roots: list[Path],
    h12_roots: list[Path], selection: Path, first_held_month: pd.Timestamp,
    end_exclusive: pd.Timestamp, reference_days: int, training_start: pd.Timestamp,
    out_dir: Path,
) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {out_dir}")
    if int(reference_days) < 7:
        raise ValueError("reference_days must be at least seven")
    fields = _selection_fields(selection)
    starts = _months(first_held_month, end_exclusive)
    out_dir.mkdir(parents=True)
    audits: list[dict[str, Any]] = []

    def load_month(month: pd.Timestamp) -> pd.DataFrame:
        end = month + pd.offsets.MonthBegin(1)
        return _merge_month(
            candidates=_candidate_month(candidates, month, end),
            features=_feature_month(features, fields, month, end),
            policy=_policy_month(policy_roots, month),
            h12=_h12_month(h12_roots, month), fields=fields,
        )

    def load_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        for month in pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left"):
            block = load_month(month)
            pieces.append(block.loc[block.__decision_ts__.ge(start) & block.__decision_ts__.lt(end)].copy())
        result = pd.concat(pieces, ignore_index=True)
        if result.candidate_id.duplicated().any():
            raise ValueError("range materialisation duplicated target-free candidate identities")
        return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)

    def latest_fit(reserve_start: pd.Timestamp) -> pd.DataFrame:
        """Select exactly the latest eligible 240k rows without a giant panel."""
        needed = int(BASE_TRAIN_CAP)
        selected: list[pd.DataFrame] = []
        available = 0
        months = list(pd.date_range(training_start.normalize().replace(day=1), reserve_start, freq="MS", inclusive="left"))
        for month in reversed(months):
            block = load_month(month)
            block = block.loc[block.__decision_ts__.lt(reserve_start)]
            eligible = _base_fit_rows(block, reserve_start)
            if eligible.empty:
                continue
            remaining = max(needed - available, 0)
            if remaining:
                eligible = eligible.tail(remaining).copy()
                selected.append(eligible)
                available += len(eligible)
            if available >= needed:
                break
        if not selected:
            return pd.DataFrame()
        result = pd.concat(selected, ignore_index=True)
        return result.sort_values(["policy_label_available_at", "candidate_id"], kind="stable").tail(needed).copy()

    for start in starts:
        end = start + pd.offsets.MonthBegin(1)
        reserve_start = start - pd.Timedelta(days=int(reference_days))
        if reserve_start <= training_start:
            raise ValueError("P0 reserve begins before the declared compatible feature history")
        held = load_range(start, end)
        reference = load_range(reserve_start, start)
        fit = latest_fit(reserve_start)
        coverage = _assert_train_coverage(fit, fields) if not fit.empty else {}
        reference_eligible = _base_score_eligible(reference)
        held_eligible = _base_score_eligible(held)
        status = "complete"
        if held.empty or int(reference_eligible.sum()) < 2 or len(fit) < 1_000:
            status = "skipped_insufficient_support"
            audits.append({
                "held_month": start.strftime("%Y-%m"), "status": status,
                "held_rows": len(held), "reference_rows": int(reference_eligible.sum()),
                "held_base_feature_eligible_rows": int(held_eligible.sum()), "fit_rows": len(fit),
            })
            continue
        # Deliberately score only feature-eligible rows.  The target-free held
        # ledger retains every candidate and records why an unavailable row
        # was not scored, but neither a held-window rank nor an all-median
        # base prediction is produced for it.
        prediction_input = pd.concat(
            [reference.loc[reference_eligible], held.loc[held_eligible]],
            ignore_index=True,
        )
        scores, model, fit_audit = _fit(fit, prediction_input, fields, P0_SPEC, train_end=reserve_start)
        target = _targets(fit, P0_SPEC)
        ordered, _, _ = _query_order(fit, target)
        medians = (
            ordered.loc[:, fields].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).median()
        )
        if medians.isna().any():
            raise AssertionError("P0 fit produced a feature without a training median")
        ref_count = int(reference_eligible.sum())
        ref_score, held_score = scores[:ref_count], scores[ref_count:]
        reference_score = ScoreReference.fit(
            ref_score, source=f"{start:%Y-%m}_same_model_prior{reference_days}_p0_f90",
        )
        map_valid = (
            _valid_policy(reference.loc[reference_eligible]).astype(bool)
            & reference.loc[reference_eligible].policy_label_available_at.lt(start).to_numpy()
        )
        if int(map_valid.sum()) < 100:
            status = "skipped_insufficient_map_support"
            audits.append({
                "held_month": start.strftime("%Y-%m"), "status": status,
                "held_rows": len(held), "reference_rows": int(reference_eligible.sum()),
                "held_base_feature_eligible_rows": int(held_eligible.sum()), "fit_rows": len(fit),
                "map_rows": int(map_valid.sum()),
            })
            continue
        policy_map = fit_policy_net_map(
            reference_score.cdf(ref_score[map_valid]),
            pd.to_numeric(
                reference.loc[reference_eligible].loc[map_valid, "p0_canonical_net_bps"],
                errors="raise",
            ),
        )
        held["prequential_base_score"] = np.nan
        held["prequential_base_rank42"] = np.nan
        held["prequential_base_anchor_bps"] = np.nan
        held.loc[held_eligible, "prequential_base_score"] = held_score
        held.loc[held_eligible, "prequential_base_rank42"] = reference_score.cdf(held_score).astype(np.float32)
        held.loc[held_eligible, "prequential_base_anchor_bps"] = policy_map.predict(
            held.loc[held_eligible, "prequential_base_rank42"]
        ).astype(np.float32)
        held["stack_is_prequential"] = True
        held["held_month"] = start.strftime("%Y-%m")
        held["base_reference_days"] = int(reference_days)
        held["base_reference_rows"] = int(reference_eligible.sum())
        held["base_map_rows"] = int(map_valid.sum())
        month_dir = out_dir / "ledger" / f"month={start:%Y-%m}"
        month_dir.mkdir(parents=True)
        held.to_parquet(month_dir / "prequential_base_ledger.parquet", index=False, compression="zstd")
        artifact_dir = out_dir / "fold_models" / f"month={start:%Y-%m}"
        _model_artifacts(
            directory=artifact_dir, model=model,
            audit={
                **fit_audit, "held_month": start.isoformat(), "reference_start": reserve_start.isoformat(),
                "reference_end_exclusive": start.isoformat(), "fit_rows": len(fit),
                "reference_rows": int(reference_eligible.sum()),
                "held_base_feature_eligible_rows": int(held_eligible.sum()),
                "map_rows": int(map_valid.sum()),
                "feature_coverage": coverage, "same_model_reference": True,
                "fit_excludes_reference_reserve": True,
            }, reference_score=ref_score, policy_map=policy_map, fields=fields, medians=medians,
        )
        audits.append({
            "held_month": start.strftime("%Y-%m"), "status": status, "held_rows": len(held),
            "reference_rows": int(reference_eligible.sum()),
            "held_base_feature_eligible_rows": int(held_eligible.sum()),
            "fit_rows": len(fit), "map_rows": int(map_valid.sum()),
            "reference_start": reserve_start, "reference_end_exclusive": start,
            "same_model_reference": True, "fit_excludes_reference_reserve": True,
            "base_feature_coverage_min": float(min(coverage.values())),
        })
    audit = pd.DataFrame(audits)
    audit.to_parquet(out_dir / "prequential_ledger_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_p0_f90_prequential_ledger_v1",
        "side": SIDE,
        "status": "complete",
        "first_held_month": starts[0].isoformat(),
        "end_exclusive": end_exclusive.isoformat(),
        "training_start": training_start.isoformat(),
        "reference_days": int(reference_days),
        "base_contract": "P0/F90 P1_policy_bps; exact timestamp×short LambdaRank K32 linear uniform",
        "training": "policy labels valid/resolved before the 42-day reserve; latest 240,000 rows cap; target-free frozen-F90 feature eligibility >=0.90",
        "rank_reference": "same fitted base scores every feature-eligible target-free candidate in preceding 42 days; no held-window percentiles",
        "policy_map": "20-bin trimmed monotonic map fitted from same-model reserve scores and only reserve outcomes resolved before held start",
        "base_feature_eligibility": {
            "minimum_available_fraction": BASE_FEATURE_MIN_FRACTION,
            "unavailable_rows": "retained with a target-free rejection reason and never scored or used for fit/map/rank support",
        },
        "outcomes_after_scoring": True,
        "source_hashes": {
            "candidates": {str(path): _sha256(path) for path in candidates},
            "features": {str(path): _sha256(path) for path in features},
            "policy_roots": {str(path): _sha256(path) for path in policy_roots},
            "h12_roots": {str(path): _sha256(path) for path in h12_roots},
            "selection": _sha256(selection),
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--feature", type=Path, action="append", required=True)
    parser.add_argument("--policy-root", type=Path, action="append", required=True)
    parser.add_argument("--h12-root", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--first-held-month", default="2024-04-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2025-01-01T00:00:00Z")
    parser.add_argument("--training-start", default="2024-01-01T00:00:00Z")
    parser.add_argument("--reference-days", type=int, default=REFERENCE_DAYS)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        candidates=args.candidate, features=args.feature, policy_roots=args.policy_root,
        h12_roots=args.h12_root, selection=args.selection,
        first_held_month=_utc(args.first_held_month), end_exclusive=_utc(args.end_exclusive),
        reference_days=args.reference_days, training_start=_utc(args.training_start), out_dir=args.out_dir,
    ))


if __name__ == "__main__":
    main()
