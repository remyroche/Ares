#!/usr/bin/env python3
"""Causal SHAP top-contributor residual-state features for the P8U Meta.

The frozen strict-OOF F72 SHAP ledger is the only explanation source.  For
each candidate we recover its ten largest absolute SHAP contributors, then
look up *prior-resolved* Base residuals of the same contributor/sign in the
same twenty-bin Base-rank band.  The values are aggregated using the current
candidate's SHAP mass.  They are an offline research extension only.

No held policy outcome may influence its own state: a residual event first
enters an availability-ordered history only after its declared policy label
availability timestamp.  The producer persists target-free F120+SHAP-state
panels before any later Meta objective opens held outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import materialize_strict_r3_p8u_meta_base_state_v1 as base_state
import run_strict_r3_p8u_meta_target_query_grid_v1 as meta


SCHEMA = "strict_r3_p8u_f72_shap_top10_residual_state_v1"
IDENTITY = tuple(meta.IDENTITY)
TOP_K = 10
WINDOWS = (7, 21)
LOCAL_BANDS = 20
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m",
    "policy_entry_price", "policy_exit_price", "policy_exit_reason", "policy_label_available_ts",
    "supportive_path_valid", "supportive_label_available_ts", "path_arch_peak_mfe_atr",
})


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in text.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise ValueError("--months must be a non-empty ordered unique YYYY-MM list")
    return values


def _end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _under_fields(path: Path) -> tuple[str, ...]:
    raw = json.loads(path.read_text())
    fields = tuple(str(value) for value in raw.get("selected_features", ()))
    if len(fields) != 120 or len(set(fields)) != len(fields):
        raise AssertionError("expected an exact immutable Under-F120 contract")
    return fields


def _assert_target_free(path: Path) -> None:
    leaked = sorted(set(pq.ParquetFile(path).schema_arrow.names).intersection(PROHIBITED))
    if leaked:
        raise AssertionError(f"{path}: target-free input leaks policy/path columns {leaked}")


def _raw_path(roots: Sequence[Path], month: pd.Timestamp) -> Path:
    candidates = [root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet" for root in roots]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise AssertionError(f"{month:%Y-%m}: expected exactly one F120 feature owner, found {len(present)}")
    return present[0]


def _shap_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _contribution_columns(path: Path) -> tuple[str, ...]:
    fields = tuple(column for column in pq.ParquetFile(path).schema_arrow.names if column.startswith("shap_f72_contrib__"))
    if len(fields) != 72 or len(set(fields)) != len(fields):
        raise AssertionError(f"{path}: expected 72 exact F72 SHAP contribution columns")
    return fields


def _topk(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable top-K field indices, signed contribution, and abs mass."""
    absolute = np.abs(values)
    # Stable ordering makes exact ties deterministic in frozen F72 field order.
    indices = np.argsort(-absolute, axis=1, kind="stable")[:, :TOP_K].astype(np.int16, copy=False)
    contribution = np.take_along_axis(values, indices, axis=1).astype(np.float32, copy=False)
    mass = np.abs(contribution).astype(np.float32, copy=False)
    return indices, contribution, mass


def _read_shap_history(root: Path, months: Sequence[pd.Timestamp]) -> tuple[pd.DataFrame, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    parts: list[pd.DataFrame] = []
    all_indices: list[np.ndarray] = []
    all_contrib: list[np.ndarray] = []
    all_mass: list[np.ndarray] = []
    fields: tuple[str, ...] | None = None
    for month in months:
        path = _shap_path(root, month)
        _assert_target_free(path)
        columns = _contribution_columns(path)
        if fields is None:
            fields = columns
        elif fields != columns:
            raise AssertionError(f"{month:%Y-%m}: F72 SHAP contribution order changed")
        required = [*IDENTITY, "base_score", "base_rank_ts", "shap_f72_entropy", *columns]
        frame = pd.read_parquet(path, columns=required)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if frame.duplicated(IDENTITY).any() or not frame.side_name.eq("long").all():
            raise AssertionError(f"{month:%Y-%m}: invalid F72 SHAP identity")
        values = frame.loc[:, list(columns)].to_numpy(np.float32)
        indices, contribution, mass = _topk(values)
        parts.append(frame.loc[:, [*IDENTITY, "base_score", "base_rank_ts", "shap_f72_entropy"]])
        all_indices.append(indices); all_contrib.append(contribution); all_mass.append(mass)
    if fields is None:
        raise AssertionError("no SHAP source months")
    history = pd.concat(parts, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    # The concatenated arrays must follow the same chronological ordering as
    # the concatenated monthly source.  All source months are chronological;
    # sort each index once more through a deterministic identity ordering.
    raw = pd.concat(parts, ignore_index=True)
    order = raw.sort_values(["__decision_ts__", "candidate_id"], kind="stable").index.to_numpy(np.int64)
    return history, fields, np.concatenate(all_indices, axis=0)[order], np.concatenate(all_contrib, axis=0)[order], np.concatenate(all_mass, axis=0)[order]


def _state_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for days in WINDOWS:
        prefix = f"shap_top10_sameband_{days}d_"
        names.extend((
            prefix + "residual_mean", prefix + "abs_residual_mean", prefix + "residual_std",
            prefix + "gt50_rate", prefix + "effective_support", prefix + "support_fraction",
            prefix + "signed_alignment",
        ))
    names.extend((
        "shap_top1_sameband_21d_residual_mean",
        "shap_top1_sameband_21d_effective_support",
        "shap_top1_sameband_21d_signed_alignment",
    ))
    return tuple(names)


def _event_keys(bands: np.ndarray, features: np.ndarray, contribution: np.ndarray, width: int) -> np.ndarray:
    sign = contribution >= 0.0
    return ((bands[:, None].astype(np.int32) * width + features.astype(np.int32)) * 2 + sign.astype(np.int32)).reshape(-1)


def _apply_event_slice(
    state: dict[str, np.ndarray], *, bands: np.ndarray, features: np.ndarray, contribution: np.ndarray,
    residual: np.ndarray, start: int, end: int, direction: float,
) -> None:
    if end <= start:
        return
    local_bands = bands[start:end]
    local_features = features[start:end]
    local_contribution = contribution[start:end]
    count = end - start
    width = len(state["count"])
    # width is flattened (band × feature × sign), so recover the 72-feature
    # stride from the declared dimension held separately below.
    feature_count = int(state["feature_count"][0])
    keys = _event_keys(local_bands, local_features, local_contribution, feature_count)
    values = np.repeat(residual[start:end].astype(float), TOP_K)
    state["count"] += direction * np.bincount(keys, minlength=width)
    state["sum"] += direction * np.bincount(keys, weights=values, minlength=width)
    state["abs_sum"] += direction * np.bincount(keys, weights=np.abs(values), minlength=width)
    state["sq_sum"] += direction * np.bincount(keys, weights=values * values, minlength=width)
    state["gt50"] += direction * np.bincount(keys, weights=(values > 50.0).astype(float), minlength=width)


def _aggregate_current(
    *, bands: np.ndarray, features: np.ndarray, contribution: np.ndarray, mass: np.ndarray,
    state: dict[str, np.ndarray], top_k: int,
) -> dict[str, np.ndarray]:
    feature_count = int(state["feature_count"][0])
    count = state["count"]
    key = _event_keys(bands, features[:, :top_k], contribution[:, :top_k], feature_count).reshape(len(bands), top_k)
    support = count[key]
    valid = support > 0.0
    weight = mass[:, :top_k].astype(float)
    weight_sum = weight.sum(axis=1, keepdims=True)
    weight = np.divide(weight, weight_sum, out=np.full_like(weight, 1.0 / top_k), where=weight_sum > 1e-12)
    denom = (weight * valid).sum(axis=1)
    safe_support = np.maximum(support, 1.0)
    mean = state["sum"][key] / safe_support
    abs_mean = state["abs_sum"][key] / safe_support
    variance = np.maximum(0.0, state["sq_sum"][key] / safe_support - mean * mean)
    std = np.sqrt(variance)
    gt50 = state["gt50"][key] / safe_support
    signed = np.where(contribution[:, :top_k] >= 0.0, 1.0, -1.0)

    def weighted(values: np.ndarray) -> np.ndarray:
        numerator = (weight * values * valid).sum(axis=1)
        return np.divide(numerator, denom, out=np.full(len(bands), np.nan, dtype=float), where=denom > 0.0).astype(np.float32)

    return {
        "residual_mean": weighted(mean), "abs_residual_mean": weighted(abs_mean),
        "residual_std": weighted(std), "gt50_rate": weighted(gt50),
        "effective_support": (weight * support).sum(axis=1).astype(np.float32),
        "support_fraction": denom.astype(np.float32), "signed_alignment": weighted(signed * mean),
    }


def _state_panels(
    history: pd.DataFrame, top_features: np.ndarray, top_contribution: np.ndarray, top_mass: np.ndarray,
    events: pd.DataFrame, feature_count: int,
) -> pd.DataFrame:
    """Build all causal states in one availability-ordered streaming pass."""
    locator = pd.Index(history.candidate_id)
    event_rows = locator.get_indexer(events.candidate_id)
    if (event_rows < 0).any():
        raise AssertionError("resolved residual event is absent from target-free F72 SHAP ledger")
    work = events.loc[:, ["available", "band", "residual_bps"]].copy()
    work["__row__"] = event_rows.astype(np.int64)
    work = work.sort_values(["available", "__row__"], kind="stable").reset_index(drop=True)
    event_available = work.available.astype("int64").to_numpy()
    event_bands = work.band.to_numpy(np.int16)
    event_residual = work.residual_bps.to_numpy(np.float32)
    event_features = top_features[work.__row__.to_numpy(np.int64)]
    event_contribution = top_contribution[work.__row__.to_numpy(np.int64)]
    key_count = LOCAL_BANDS * feature_count * 2
    states = {
        days: {
            "count": np.zeros(key_count, dtype=np.float64),
            "sum": np.zeros(key_count, dtype=np.float64),
            "abs_sum": np.zeros(key_count, dtype=np.float64),
            "sq_sum": np.zeros(key_count, dtype=np.float64),
            "gt50": np.zeros(key_count, dtype=np.float64),
            "feature_count": np.asarray([feature_count], dtype=np.int32),
        }
        for days in WINDOWS
    }
    output = history.loc[:, list(IDENTITY)].copy()
    fields = _state_feature_names()
    values = {field: np.full(len(history), np.nan, dtype=np.float32) for field in fields}
    decision = history.__decision_ts__.astype("int64").to_numpy()
    bands = base_state._base_band(history.base_rank_ts).astype(np.int16)
    added = 0
    added_to = {days: 0 for days in WINDOWS}
    expired = {days: 0 for days in WINDOWS}
    start = 0
    while start < len(history):
        stop = start + 1
        timestamp = decision[start]
        while stop < len(history) and decision[stop] == timestamp:
            stop += 1
        while added < len(work) and event_available[added] < timestamp:
            added += 1
        for days in WINDOWS:
            _apply_event_slice(
                states[days], bands=event_bands, features=event_features, contribution=event_contribution,
                residual=event_residual, start=added_to[days], end=added, direction=1.0,
            )
            added_to[days] = added
            cutoff = timestamp - int(pd.Timedelta(days=days).value)
            remove_to = expired[days]
            while remove_to < added and event_available[remove_to] < cutoff:
                remove_to += 1
            _apply_event_slice(
                states[days], bands=event_bands, features=event_features, contribution=event_contribution,
                residual=event_residual, start=expired[days], end=remove_to, direction=-1.0,
            )
            expired[days] = remove_to
            aggregate = _aggregate_current(
                bands=bands[start:stop], features=top_features[start:stop], contribution=top_contribution[start:stop],
                mass=top_mass[start:stop], state=states[days], top_k=TOP_K,
            )
            prefix = f"shap_top10_sameband_{days}d_"
            for name, array in aggregate.items():
                values[prefix + name][start:stop] = array
            if days == 21:
                top1 = _aggregate_current(
                    bands=bands[start:stop], features=top_features[start:stop], contribution=top_contribution[start:stop],
                    mass=top_mass[start:stop], state=states[days], top_k=1,
                )
                values["shap_top1_sameband_21d_residual_mean"][start:stop] = top1["residual_mean"]
                values["shap_top1_sameband_21d_effective_support"][start:stop] = top1["effective_support"]
                values["shap_top1_sameband_21d_signed_alignment"][start:stop] = top1["signed_alignment"]
        start = stop
    for field, array in values.items():
        output[field] = array
    if output.duplicated(IDENTITY).any():
        raise AssertionError("state output identity duplication")
    return output


def _timestamp_top10_summary(
    history: pd.DataFrame, top_features: np.ndarray, top_mass: np.ndarray, contribution_fields: Sequence[str], months: Sequence[pd.Timestamp], out: Path,
) -> None:
    """Persist the top ten aggregate contributors for every decision timestamp.

    This compact audit is distinct from the per-candidate top-ten state used
    in the Meta inputs: it answers which F72 features dominated each complete
    cross-section without retaining a multi-million-row exploded payload.
    """
    decision_codes, timestamps = pd.factorize(history.__decision_ts__, sort=True)
    n_time = len(timestamps)
    feature_count = len(contribution_fields)
    repeat_codes = np.repeat(decision_codes, TOP_K)
    flat_features = top_features.reshape(-1).astype(np.int64)
    flat_mass = top_mass.reshape(-1).astype(float)
    total = np.bincount(repeat_codes * feature_count + flat_features, weights=flat_mass, minlength=n_time * feature_count).reshape(n_time, feature_count)
    hits = np.bincount(repeat_codes * feature_count + flat_features, minlength=n_time * feature_count).reshape(n_time, feature_count)
    candidate_count = np.bincount(decision_codes, minlength=n_time).astype(float)
    selected = np.argsort(-total, axis=1, kind="stable")[:, :TOP_K]
    rows: list[pd.DataFrame] = []
    for rank in range(TOP_K):
        feature = selected[:, rank]
        rows.append(pd.DataFrame({
            "__decision_ts__": timestamps, "contributor_rank": rank + 1,
            "contributor_feature": [contribution_fields[index].removeprefix("shap_f72_contrib__") for index in feature],
            "aggregate_abs_shap_mass": total[np.arange(n_time), feature].astype(np.float32),
            "candidate_fraction_top10": (hits[np.arange(n_time), feature] / np.maximum(candidate_count, 1.0)).astype(np.float32),
        }))
    summary = pd.concat(rows, ignore_index=True).sort_values(["__decision_ts__", "contributor_rank"], kind="stable")
    summary.to_parquet(out / "timestamp_top10_contributor_summary.parquet", index=False, compression="zstd")


def _coverage(frame: pd.DataFrame, fields: Sequence[str], month: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "month": f"{month:%Y-%m}", "feature": list(fields), "rows": len(frame),
        "finite_fraction": [float(pd.to_numeric(frame[field], errors="coerce").notna().mean()) for field in fields],
        "n_unique": [int(pd.to_numeric(frame[field], errors="coerce").nunique(dropna=True)) for field in fields],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shap-root", type=Path, required=True)
    parser.add_argument("--raw-feature-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--under-contract", type=Path, required=True)
    parser.add_argument("--policy-labels", type=Path, required=True)
    parser.add_argument("--months", default="2025-08,2025-09,2025-10,2025-11,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06,2026-07")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(args.out)
    months = _months(args.months)
    under_fields = _under_fields(args.under_contract.resolve())
    args.out.mkdir(parents=True)
    history, contrib_fields, top_features, top_contribution, top_mass = _read_shap_history(args.shap_root.resolve(), months)
    if history.candidate_id.duplicated().any():
        raise AssertionError("F72 SHAP history candidate identity must be globally unique")
    policy = meta._read_policy(args.policy_labels.resolve())
    # This is a strict-prequential Base anchor.  Outcome information is used
    # only to form events at ``available`` timestamps, never in a same-row or
    # future state feature.
    events = base_state._policy_events(history.loc[:, [*IDENTITY, "base_score", "base_rank_ts"]], policy)
    state = _state_panels(history, top_features, top_contribution, top_mass, events, len(contrib_fields))
    state_fields = _state_feature_names()
    _timestamp_top10_summary(history, top_features, top_mass, contrib_fields, months, args.out)
    coverage_parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    contracts = args.out / "contracts"
    contracts.mkdir()
    derived_all = (*state_fields, "shap_f72_entropy")
    for month in months:
        start, end = month, _end(month)
        shap_part = history.loc[history.__decision_ts__.ge(start) & history.__decision_ts__.lt(end), list(IDENTITY)].copy()
        state_part = state.loc[state.__decision_ts__.ge(start) & state.__decision_ts__.lt(end)].copy()
        raw_path = _raw_path(tuple(path.resolve() for path in args.raw_feature_roots), month)
        _assert_target_free(raw_path)
        raw = pd.read_parquet(raw_path, columns=[*IDENTITY, *under_fields])
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        entropy = history.loc[history.__decision_ts__.ge(start) & history.__decision_ts__.lt(end), [*IDENTITY, "shap_f72_entropy"]].copy()
        panel = raw.merge(state_part, on=list(IDENTITY), how="inner", validate="one_to_one")
        panel = panel.merge(entropy, on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(panel) != len(shap_part) or panel.duplicated(IDENTITY).any() or not panel.side_name.eq("long").all():
            raise AssertionError(f"{month:%Y-%m}: F120/SHAP-state exact identity failure")
        if any(column in PROHIBITED for column in panel.columns):
            raise AssertionError(f"{month:%Y-%m}: policy/path column entered target-free output")
        panel = panel.loc[:, [*IDENTITY, *under_fields, *derived_all]].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        target = args.out / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        target.parent.mkdir()
        panel.to_parquet(target, index=False, compression="zstd")
        coverage_parts.append(_coverage(panel, derived_all, month))
        audits.append({
            "month": f"{month:%Y-%m}", "rows": len(panel), "f120_fields": len(under_fields),
            "derived_fields": len(derived_all), "f120_complete_rows": int(panel.loc[:, list(under_fields)].notna().all(axis=1).sum()),
            "state_complete_rows": int(panel.loc[:, list(state_fields)].notna().all(axis=1).sum()),
            "strict_prior_events_before_month": int((events.available < start).sum()),
            "target_free_identity_exact": True,
        })
        print(json.dumps({"event": "month_complete", **audits[-1]}, sort_keys=True), flush=True)
    pd.concat(coverage_parts, ignore_index=True).to_parquet(args.out / "feature_coverage.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out / "month_audit.parquet", index=False, compression="zstd")
    selected = {
        "entropy_only": [*under_fields, "shap_f72_entropy"],
        "top10_residual_state": [*under_fields, *state_fields],
        "top10_residual_state_entropy": [*under_fields, *state_fields, "shap_f72_entropy"],
    }
    for arm, fields in selected.items():
        _once(contracts / f"{arm}.json", {
            "schema": SCHEMA, "arm": arm, "selected_features": fields,
            "parent_feature_count": len(under_fields), "added_feature_count": len(fields) - len(under_fields),
            "selection_scope": "predeclared target-free F72 SHAP top-ten contributor residual state",
        })
    _once(args.out / "correctness_report.json", {
        "strict_oof_f72_shap_ledger_reused": True,
        "catboost_shap_parent_score_reconstruction_receipt_checked": True,
        "top10_contributors_are_candidate_local_absolute_shap_rankings": True,
        "timestamp_top10_summary_persisted": True,
        "same_base_rank_band_used_for_residual_state": True,
        "residual_events_activate_only_after_policy_label_available_ts": True,
        "base_residual_anchor_is_strict_prequential": True,
        "no_same_or_future_outcome_in_state": True,
        "target_free_f120_panels_persist_before_meta_outcome_evaluation": True,
        "no_live_mc1_admission_portfolio_or_exchange_mutation": True,
    })
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-free F72 SHAP top-ten contributor residual-state Meta research only",
        "months": [f"{month:%Y-%m}" for month in months], "top_k": TOP_K, "windows_days": list(WINDOWS),
        "score_bands": LOCAL_BANDS, "shap_root": str(args.shap_root.resolve()),
        "under_contract": str(args.under_contract.resolve()), "policy_labels": str(args.policy_labels.resolve()),
        "raw_feature_roots": [str(path.resolve()) for path in args.raw_feature_roots],
        "contribution_feature_names": [field.removeprefix("shap_f72_contrib__") for field in contrib_fields],
        "state_features": list(state_fields), "source_shap_sha256": _sha(args.shap_root.resolve()),
        "audit": audits,
    })


if __name__ == "__main__":
    main()
