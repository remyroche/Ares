#!/usr/bin/env python3
"""Strict-OOS short P0/F90 residual-consensus evaluation.

This is deliberately an ensemble experiment, not ten standalone model
promotions.  A ten-head contract is frozen on Oct--Dec 2024; every 2025 held
month then fits each head only on earlier resolved policy-net residual rows,
ranks against its training-fold score distribution, and emits a median
consensus.  A later Jan--Jun OOF-only CMI selector may choose a complementary
ensemble for the untouched Jul--Dec evaluation block.
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

from extreme_price_movements.specialist_head_selection import select_complementary_heads  # noqa: E402
from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    ConsensusHeadSpec,
    _fit_consensus_head,
)
from extreme_price_movements.strict_r3_canonical_v2 import load_geometry_bundle  # noqa: E402


SIDE = "short"
EDGES = (-150.0, -50.0, 50.0, 150.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(value for value in path.rglob("*") if value.is_file())
    for value in paths:
        digest.update(str(value.relative_to(path) if path.is_dir() else value.name).encode())
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left"))


def _residual_grade(values: pd.Series, edges: tuple[float, ...] = EDGES) -> np.ndarray:
    residual = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if len(edges) != 4:
        raise ValueError("short consensus ordinal residual target needs four edges")
    return np.select(
        [residual <= edge for edge in edges], [0, 1, 2, 3], default=4,
    ).astype(np.int8)


def _load_ledger(root: Path, *, minimum_month: pd.Timestamp) -> pd.DataFrame:
    parts = [
        path for path in sorted((root / "ledger").glob("month=*/prequential_base_ledger.parquet"))
        if pd.Timestamp(path.parent.name.removeprefix("month=") + "-01", tz="UTC") >= minimum_month
    ]
    if not parts:
        raise FileNotFoundError(f"no monthly short ledger parts under {root}")
    result = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    if result.candidate_id.duplicated().any():
        raise ValueError(f"duplicate candidate identities in {root}")
    return result


def _valid_base_rows(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["base_feature_eligible"].fillna(False).astype(bool)
        & frame["stack_is_prequential"].fillna(False).astype(bool)
        & pd.to_numeric(frame["prequential_base_rank42"], errors="coerce").notna()
        & pd.to_numeric(frame["prequential_base_anchor_bps"], errors="coerce").notna()
    )


def _valid_residual_rows(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    return (
        _valid_base_rows(frame)
        & frame["policy_path_valid"].fillna(False).astype(bool)
        & _utc(frame["policy_label_available_at"]).lt(cutoff)
        & pd.to_numeric(frame["p0_canonical_net_bps"], errors="coerce").notna()
    )


def _head_specs(contract: dict[str, Any]) -> tuple[ConsensusHeadSpec, ...]:
    if contract.get("schema") not in {
        "strict_r3_short_p0_cmi_consensus_v1",
        "strict_r3_short_p0_cmi_consensus_v2",
    } or contract.get("side") != SIDE:
        raise ValueError("not a frozen short P0 CMI consensus contract")
    edges = tuple(float(value) for value in contract["target"]["edges_bps"])
    if len(edges) != 4:
        raise ValueError("short residual target must declare exactly four edges")
    result = []
    for raw in contract["heads"]:
        fields = tuple(str(value) for value in raw["fields"])
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"invalid fields for {raw.get('name')}")
        head_edges = tuple(float(value) for value in raw.get("target_edges_bps", edges))
        if len(head_edges) != 4:
            raise ValueError(f"{raw.get('name')} has invalid target edges")
        result.append(ConsensusHeadSpec(
            name=str(raw["name"]), cap=int(raw["cap"]),
            weight_mode=str(raw["weight_mode"]), query=str(raw["query"]),
            fields=fields, target_edges_bps=head_edges,
            params=dict(raw.get("ranker_params", contract["ranker_params"])),
        ))
    if contract.get("schema") == "strict_r3_short_p0_cmi_consensus_v1":
        if len(result) != 10 or len({spec.name for spec in result}) != 10:
            raise ValueError("v1 short consensus contract must define ten unique heads")
    elif not result or len(result) > 10 or len({spec.name for spec in result}) != len(result):
        raise ValueError("v2 short consensus contract must define one to ten unique promoted heads")
    return tuple(result)


def _with_geometry(frame: pd.DataFrame, geometry: Any, required: list[str]) -> pd.DataFrame:
    state = geometry.transform(frame).reset_index(drop=True)
    missing = sorted(set(required).difference(frame.columns).difference(state.columns))
    if missing:
        raise KeyError(f"short consensus contract fields missing from base/geometry state: {missing}")
    overlap = set(frame.columns).intersection(state.columns)
    if overlap:
        raise AssertionError(f"geometry state overwrote ledger fields: {sorted(overlap)}")
    return pd.concat([frame.reset_index(drop=True), state], axis=1)


def _rank_ic(frame: pd.DataFrame, score: str) -> float:
    work = frame.loc[
        frame[score].notna() & frame["p0_canonical_net_bps"].notna(),
        ["__decision_ts__", score, "p0_canonical_net_bps"],
    ].copy()
    values = []
    for _, group in work.groupby("__decision_ts__", sort=False):
        if len(group) < 3 or group[score].nunique() < 2 or group.p0_canonical_net_bps.nunique() < 2:
            continue
        value = group[score].corr(group.p0_canonical_net_bps, method="spearman")
        if np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _tail_rows(frame: pd.DataFrame, score: str, *, tail: float) -> pd.DataFrame:
    valid = frame.loc[frame[score].notna() & frame.p0_canonical_net_bps.notna()].copy()
    if valid.empty:
        return valid
    threshold = valid[score].quantile(1.0 - tail, interpolation="higher")
    return valid.loc[valid[score].ge(threshold)].copy()


def _metrics(frame: pd.DataFrame, *, score: str, stage: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    month = _utc(frame["__decision_ts__"]).dt.strftime("%Y-%m")
    for label, subset in [("pooled", frame), *[(value, frame.loc[month.eq(value)]) for value in sorted(month.unique())]]:
        valid = subset.loc[subset.policy_path_valid.fillna(False) & subset.p0_canonical_net_bps.notna()].copy()
        if valid.empty:
            continue
        common = {
            "stage": stage, "score": score, "month": label, "valid_rows": int(len(valid)),
            "rank_ic": _rank_ic(valid, score),
        }
        for tail in (0.01, 0.02, 0.05):
            tail_rows = _tail_rows(valid, score, tail=tail)
            rows.append({
                **common, "tail": tail, "trades": int(len(tail_rows)),
                "net_bps_per_trade": float(tail_rows.p0_canonical_net_bps.mean()) if len(tail_rows) else float("nan"),
                "total_net_bps": float(tail_rows.p0_canonical_net_bps.sum()) if len(tail_rows) else float("nan"),
            })
    return pd.DataFrame(rows)


def run(
    *, ledger_roots: list[Path], contract_path: Path, geometry_dir: Path,
    start: pd.Timestamp, end: pd.Timestamp, selection_end: pd.Timestamp, out: Path,
    history_start: pd.Timestamp | None = None,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    contract = json.loads(contract_path.read_text())
    heads = _head_specs(contract)
    geometry = load_geometry_bundle(geometry_dir)
    if geometry.bundle_sha256 != contract["geometry"]["bundle_sha256"]:
        raise ValueError("short consensus geometry hash differs from the frozen field contract")
    # The P0 base and residual labels are strict-prequential already.  Do not
    # silently discard compatible earlier training support just because the
    # held evaluation starts later; that would change the causal model from
    # the intended expanding-history contract.  The old three-month default
    # remains for backwards-compatible, compact experiments only.
    resolved_history_start = (
        (start.tz_convert(None).to_period("M") - 3).to_timestamp().tz_localize("UTC")
        if history_start is None else pd.Timestamp(history_start)
    )
    resolved_history_start = (
        resolved_history_start.tz_localize("UTC")
        if resolved_history_start.tzinfo is None
        else resolved_history_start.tz_convert("UTC")
    )
    ledger = pd.concat(
        [_load_ledger(root, minimum_month=resolved_history_start) for root in ledger_roots],
        ignore_index=True,
    )
    if ledger.candidate_id.duplicated().any():
        raise ValueError("source ledger roots overlap")
    for column in ("__decision_ts__", "policy_label_available_at"):
        ledger[column] = _utc(ledger[column])
    if not ledger.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("short consensus source includes non-short rows")
    required = sorted({field for head in heads for field in head.fields})
    ledger = _with_geometry(ledger, geometry, required)
    ledger["policy_residual_bps"] = (
        pd.to_numeric(ledger.p0_canonical_net_bps, errors="coerce")
        - pd.to_numeric(ledger.prequential_base_anchor_bps, errors="coerce")
    )
    ledger["residual_grade"] = _residual_grade(ledger.policy_residual_bps)
    held_months = _month_range(start, end)
    out.mkdir(parents=True)
    predictions: list[pd.DataFrame] = []
    head_metrics: list[pd.DataFrame] = []
    for fold_index, held_start in enumerate(held_months):
        held_end = held_start + pd.offsets.MonthBegin(1)
        train = ledger.loc[
            ledger.__decision_ts__.lt(held_start) & _valid_residual_rows(ledger, held_start)
        ].copy()
        held = ledger.loc[
            ledger.__decision_ts__.ge(held_start) & ledger.__decision_ts__.lt(held_end) & _valid_base_rows(ledger)
        ].copy()
        if len(train) < 1_000 or held.empty:
            continue
        held["__held_month__"] = held_start.strftime("%Y-%m")
        head_ranks: list[np.ndarray] = []
        for head_index, spec in enumerate(heads):
            # A HPO-selected head may use a different ordinalisation, but it
            # remains a policy-net residual target and is computed from
            # training rows only.
            grade = _residual_grade(train.policy_residual_bps, spec.target_edges_bps)
            fitted = _fit_consensus_head(train, grade, spec, seed=20260821 + fold_index * 100 + head_index)
            raw, rank = fitted.predict_rank(held)
            held[f"head__{spec.name}__raw"] = raw
            held[f"head__{spec.name}__rank"] = rank
            head_ranks.append(rank)
            model_dir = out / "fold_models" / f"month={held_start:%Y-%m}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted, model_dir / f"{spec.name}.joblib", compress=3)
        held["conditional_consensus_rank_all10"] = np.nanmedian(np.column_stack(head_ranks), axis=1).astype(np.float32)
        held["upstream_all10"] = (
            .75 * pd.to_numeric(held.prequential_base_rank42, errors="coerce").to_numpy(float)
            + .25 * held.conditional_consensus_rank_all10.to_numpy(float)
        ).astype(np.float32)
        predictions.append(held)
    if not predictions:
        raise ValueError("short consensus OOF has no eligible held folds")
    output = pd.concat(predictions, ignore_index=True)
    head_rank_columns = [f"head__{head.name}__rank" for head in heads]
    # Head-specific standalone metrics are emitted as diagnostics only.  The
    # selection below is based on their incremental *ensemble* information.
    for column in head_rank_columns:
        head_metrics.append(_metrics(output, score=column, stage="diagnostic_head"))
    all10_metrics = _metrics(output, score="upstream_all10", stage="all10_ensemble")
    selection_train = output.loc[
        output.__decision_ts__.lt(selection_end)
        & output.policy_path_valid.fillna(False)
        & output.p0_canonical_net_bps.notna()
    ].copy()
    if len(selection_train) < 1_000:
        raise ValueError("insufficient predeclared OOF support for ensemble selection")
    selected, selection_audit = select_complementary_heads(
        selection_train,
        head_rank_columns,
        target_column="residual_grade",
        base_score_column="prequential_base_rank42",
        max_heads=6,
        minimum_cmi=.001,
    )
    if not selected:
        raise ValueError("OOF CMI selector retained no consensus heads")
    output["conditional_consensus_rank_selected"] = np.nanmedian(
        output.loc[:, selected].to_numpy(float), axis=1,
    ).astype(np.float32)
    output["upstream_selected"] = (
        .75 * output.prequential_base_rank42.to_numpy(float)
        + .25 * output.conditional_consensus_rank_selected.to_numpy(float)
    ).astype(np.float32)
    selected_test = output.loc[output.__decision_ts__.ge(selection_end)].copy()
    selected_metrics = _metrics(selected_test, score="upstream_selected", stage="selected_ensemble_untouched")
    output.to_parquet(out / "short_consensus_oof_predictions.parquet", index=False, compression="zstd")
    pd.concat([all10_metrics, selected_metrics], ignore_index=True).to_parquet(
        out / "ensemble_metrics.parquet", index=False, compression="zstd",
    )
    pd.concat(head_metrics, ignore_index=True).to_parquet(
        out / "head_diagnostic_metrics.parquet", index=False, compression="zstd",
    )
    selection_audit.to_parquet(out / "oof_ensemble_cmi_selection.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_short_p0_consensus_oof_v1", "status": "complete", "side": SIDE,
        "held_window": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "history_start": resolved_history_start.isoformat(),
        "head_training": "strictly earlier decision rows whose policy labels resolved before each held month",
        "base_dependency": "strict-prequential P0/F90 ranks and policy anchors only",
        "rank_domain": "each head's complete-query capped resolved training-score distribution; never its held month",
        "geometry": {"bundle_sha256": geometry.bundle_sha256, "monthly_refit": False},
        "ensemble_selection": {
            "training_window_end_exclusive": selection_end.isoformat(), "selected_rank_columns": selected,
            "evaluation": "selection is assessed only on held rows on/after this date",
        },
        "source_hashes": {
            "contract": _sha256(contract_path), "geometry_manifest": _sha256(geometry_dir / "run_manifest.json"),
            "ledgers": {str(root): _sha256(root / "run_manifest.json") for root in ledger_roots},
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-01-01T00:00:00Z")
    parser.add_argument("--selection-end", default="2025-07-01T00:00:00Z")
    parser.add_argument(
        "--history-start",
        default=None,
        help="Earliest compatible strict-prequential ledger month; defaults to three months before --start.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        ledger_roots=args.ledger_root, contract_path=args.contract, geometry_dir=args.geometry_dir,
        start=pd.Timestamp(args.start), end=pd.Timestamp(args.end_exclusive),
        selection_end=pd.Timestamp(args.selection_end), out=args.out,
        history_start=None if args.history_start is None else pd.Timestamp(args.history_start),
    ))


if __name__ == "__main__":
    main()
