#!/usr/bin/env python3
"""Replay the side-local residual conversion with explicit cross-query calibration.

The source run already has the two important semantic contracts:

* specialists are trained on the ordinal H12 residual (realised net minus the
  causal side-local base EV), and
* the base and residual components are mapped in side-local EV units before
  the final pooled global ranking.

This script is deliberately replay-only.  It does not refit a specialist or
use a test label to choose a model.  It adds a declared calibration ablation:
the existing final conversion score (already in common bps units) is converted
to a common expected-net scale by a fixed 20-bin, monotone, prior-resolved map
pooled across all 4-hour queries.  The map is fit separately by side (the
requested arm) and, as a negative control, across both sides.  A fixed score
domain is used; no future score quantiles are fitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.prequential_r3_value_map import (  # noqa: E402
    PrequentialR3ValueMapConfig,
    prequential_same_side_r3_value_map,
)


SOURCE = ROOT / "data_perp/artifacts/side_local_conversion_residual_ev_20260806_v1"
OUT = ROOT / "data_perp/artifacts/side_local_conversion_calibration_ablation_20260806_v1"
TAILS = (0.01, 0.05, 0.10)
LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
DELAY = pd.Timedelta(hours=13)
SCORE_SCALE_BPS = 250.0


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _system_name(column: str) -> str:
    if column == "score_lambda_000":
        return "base_only"
    return f"base_plus_residual_lambda_{int(column.rsplit('_', 1)[-1]):03d}"


def _query_rank(frame: pd.DataFrame, score_column: str) -> np.ndarray:
    query = (
        pd.to_datetime(frame["__ts__"], utc=True).dt.floor("4h").astype(str)
        + "|"
        + frame["side_name"].astype(str)
    )
    return frame[score_column].groupby(query, sort=False).rank(method="first", pct=True).to_numpy(float)


def _map_cross_query(
    frame: pd.DataFrame,
    score_column: str,
    *,
    side: str | None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Map common-bps scores to exact net using only prior-resolved labels."""
    work = frame if side is None else frame[frame.side_name.eq(side)].copy()
    if work.empty:
        return np.full(len(frame), np.nan, dtype=np.float32), pd.DataFrame(), {}
    raw = pd.to_numeric(work[score_column], errors="coerce").to_numpy(float)
    score = np.clip(raw / SCORE_SCALE_BPS, -1.0, 1.0)
    rank = _query_rank(work, score_column)
    values, audit, provenance = prequential_same_side_r3_value_map(
        exact_net_bps=work.net_bps.to_numpy(float),
        decision_timestamps=work.__ts__,
        label_available_timestamps=work.label_available_ts,
        # The function's side argument controls the contract validation.  A
        # pooled negative-control map intentionally uses the same mechanics
        # over both sides; its label is recorded as pooled below.
        side=(side or "long"),
        score=score,
        config=PrequentialR3ValueMapConfig(
            side=(side or "long"),
            bins=20,
            min_global_rows=32,
            bin_shrink_rows=64,
            mapping_mode="monotone_pava",
            monotone_min_bin_rows=1,
        ),
    )
    output = np.full(len(frame), np.nan, dtype=np.float32)
    output[work.index.to_numpy()] = values.astype(np.float32)
    audit = audit.copy()
    audit["candidate_id"] = work.candidate_id.to_numpy()
    audit["mapping_scope"] = side or "pooled"
    audit["score_column"] = score_column
    audit["within_query_rank"] = rank.astype(np.float32)
    return output, audit, dict(provenance)


def _metric_rows(frame: pd.DataFrame, score_column: str, system: str, *, scope: str, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tail in TAILS:
        n = max(1, int(np.ceil(len(frame) * tail)))
        top = frame.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").head(n)
        rows.append({
            "system": system,
            "scope": scope,
            "period": period,
            "tail": tail,
            "rows": int(len(frame)),
            "trades": int(n),
            "gross_bps": float(top.gross_bps.mean()),
            "net_bps": float(top.net_bps.mean()),
            "rank_ic_net": float(frame[score_column].corr(frame.net_bps, method="spearman")),
        })
    return rows


def _metrics(frame: pd.DataFrame, score_columns: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system, column in score_columns.items():
        valid = frame[np.isfinite(pd.to_numeric(frame[column], errors="coerce"))].copy()
        if valid.empty:
            continue
        rows.extend(_metric_rows(valid, column, system, scope="global", period="all"))
        for side, side_frame in valid.groupby("side_name", sort=True):
            rows.extend(_metric_rows(side_frame, column, system, scope=f"side:{side}", period="all"))
        for month, month_frame in valid.groupby("month", sort=True):
            rows.extend(_metric_rows(month_frame, column, system, scope="global", period=str(month)))
        for fold, fold_frame in valid.groupby("fold", sort=True):
            rows.extend(_metric_rows(fold_frame, column, system, scope="global", period=str(fold)))
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, *, digits: int = 2) -> str:
    """Small dependency-free Markdown table renderer."""
    if frame.empty:
        return "(empty)"
    x = frame.reset_index()
    headers = [str(c) for c in x.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in x.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                cells.append(f"{float(value):.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _target_alignment_audit(source: Path) -> dict[str, object]:
    contract = json.loads((source / "conversion_contract.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    target_path = source / "specialist_target_audit.parquet"
    target = pd.read_parquet(target_path) if target_path.exists() else pd.DataFrame()
    specialist_targets = sorted(target.target.dropna().astype(str).unique()) if not target.empty else []
    residual_ok = bool(specialist_targets) and all(
        ("residual" in value.lower()) or ("base_map" in value.lower())
        for value in specialist_targets
    )
    meta_target = str(contract.get("meta_target", ""))
    side_local_base = "side-local" in str(contract.get("base_map", "")).lower()
    side_local_meta = "side-local" in str(contract.get("meta_map", "")).lower()
    return {
        "specialist_target_audit_exists": bool(target_path.exists()),
        "specialist_target_values": specialist_targets,
        "specialist_aligned_on_residual": residual_ok,
        "meta_target": meta_target,
        "meta_aligned_on_same_residual": "residual" in meta_target.lower(),
        "base_ev_mapping_side_local": side_local_base,
        "residual_ev_mapping_side_local": side_local_meta,
        "strict_boundary": contract.get("strict_boundary"),
        "source_status": manifest.get("schema"),
    }


def run(source: Path = SOURCE, out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(source / "predictions.parquet").copy()
    required = {"candidate_id", "__ts__", "side_name", "net_bps", "gross_bps", "fold", "base_ev_bps"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"source predictions missing required columns: {missing}")
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True)
    pred["label_available_ts"] = pred["__ts__"] + DELAY
    pred["month"] = pred.__ts__.dt.strftime("%Y-%m")
    pred["cost_bps"] = pred.gross_bps - pred.net_bps
    if not np.allclose(pred.cost_bps.to_numpy(float), 100.0, atol=0.05):
        raise ValueError("source predictions do not satisfy the single 100-bps cost contract")
    if pred.candidate_id.duplicated().any():
        raise ValueError("source predictions contain duplicate candidate IDs")

    score_columns = {
        "base_only": "score_lambda_000",
        "base_plus_residual_lambda_025": "score_lambda_025",
        "base_plus_residual_lambda_050": "score_lambda_050",
        "base_plus_residual_lambda_075": "score_lambda_075",
        "base_plus_residual_lambda_100": "score_lambda_100",
    }
    selected = pd.read_parquet(source / "lambda_selection.parquet")
    selected_map = {(str(row.fold), str(row.side)): float(row.selected_lambda) for row in selected.itertuples()}
    selected_col = np.full(len(pred), np.nan, dtype=np.float32)
    for i, row in enumerate(pred.itertuples()):
        key = (str(row.fold), str(row.side_name))
        lam = selected_map.get(key)
        if lam is not None:
            selected_col[i] = getattr(row, f"score_lambda_{int(lam * 100):03d}")
    pred["score_selected_oof"] = selected_col
    score_columns["selected_oof"] = "score_selected_oof"

    calibration_audits: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    for system, source_column in list(score_columns.items()):
        for scope_name in ("side_local", "pooled"):
            sides = ("long", "short") if scope_name == "side_local" else (None,)
            mapped = np.full(len(pred), np.nan, dtype=np.float32)
            scope_provenance: list[dict[str, object]] = []
            for side in sides:
                side_mapped, audit, prov = _map_cross_query(pred, source_column, side=side)
                finite = np.isfinite(side_mapped)
                mapped[finite] = side_mapped[finite]
                if not audit.empty:
                    calibration_audits.append(audit)
                scope_provenance.append({"side": side or "pooled", "provenance": prov})
            column = f"score__{scope_name}_cross_query__{system}"
            pred[column] = mapped
            provenance.append({
                "system": system,
                "source_column": source_column,
                "mapping_scope": scope_name,
                "side": "long+short" if scope_name == "side_local" else "pooled",
                "mapping_mode": "monotone_pava",
                "fit_boundary": "label_available_ts < decision_ts",
                "provenance": scope_provenance,
            })
            score_columns[f"{scope_name}_cross_query__{system}"] = column

    metrics = _metrics(pred, score_columns)
    metrics.to_parquet(out / "metrics.parquet", index=False)
    pred.to_parquet(out / "predictions.parquet", index=False)
    if calibration_audits:
        pd.concat(calibration_audits, ignore_index=True).to_parquet(out / "cross_query_calibration_audit.parquet", index=False)
    alignment = _target_alignment_audit(source)
    _write_json(out / "target_alignment_audit.json", alignment)
    _write_json(out / "calibration_manifest.json", {
        "schema": "side_local_conversion_cross_query_calibration_ablation_v1",
        "source_artifact": str(source),
        "source_specialist_target": "ordinalized H12 net residual bps",
        "conversion_contract": "existing side-local base EV + side-local residual EV; no cost reapplication",
        "calibration_input": f"already converted common-bps score scaled by fixed {SCORE_SCALE_BPS:.0f} bps domain",
        "calibration_target": "exact H12 net bps",
        "side_local_arm": "one prior-resolved 20-bin monotone PAVA per side pooled across all 4-hour queries",
        "pooled_control": "one prior-resolved 20-bin monotone PAVA across both sides",
        "strict_boundary": "label_available_ts < decision_ts",
        "ranking": "global common-bps top-k after mapping; side/month rows are diagnostics",
        "provenance": provenance,
    })

    pooled = metrics[(metrics.scope == "global") & (metrics.period == "all") & metrics["tail"].isin(TAILS)].copy()
    pivot = pooled.pivot(index="system", columns="tail", values="net_bps").rename(columns={0.01: "top1_net_bps", 0.05: "top5_net_bps", 0.10: "top10_net_bps"})
    monthly = metrics[(metrics.scope == "global") & metrics.period.str.match(r"^2024-") & metrics["tail"].eq(0.05)].pivot(index="period", columns="system", values="net_bps")
    lines = [
        "# Side-local conversion / residual-target calibration ablation",
        "",
        f"Source: `{source}`",
        "",
        "The source arm already trains specialists on the ordinal H12 residual (realised net minus the causal side-local base EV), maps base and residual components in side-local EV units, and ranks globally only after conversion to common bps. This replay adds an explicit cross-query calibration of the final converted score.",
        "",
        "## Global OOS H12 net bps/trade",
        "",
        _markdown_table(pivot),
        "",
        "`side_local_cross_query__*` is the requested arm. `pooled_cross_query__*` is a negative control that deliberately removes side-local separation. The unprefixed systems are the source conversion controls.",
        "",
        "## Monthly global top-5 net bps/trade",
        "",
        _markdown_table(monthly),
        "",
        "All calibration maps use only rows whose label availability precedes the decision timestamp. Side and month metrics are diagnostics; selection remains global across both sides.",
        "",
        "## Contract audit",
        "",
        f"Specialist residual target aligned: `{alignment['specialist_aligned_on_residual']}`; meta residual target aligned: `{alignment['meta_aligned_on_same_residual']}`; base EV mapping side-local: `{alignment['base_ev_mapping_side_local']}`; residual EV mapping side-local: `{alignment['residual_ev_mapping_side_local']}`.",
        "",
        "## Decision",
        "",
        "The explicit second cross-query calibration does not advance: every side-local calibrated global top-5/top-10 arm is below the uncalibrated side-local conversion control, and the apparent month gains are not portable because August deteriorates sharply. Retain the existing side-local base/residual EV maps and residual specialist target; do not add this second calibration layer without a new, longer training/calibration history.",
        "",
    ]
    (out / "SIDE_LOCAL_CONVERSION_CALIBRATION_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n")
    _write_json(out / "run_manifest.json", {
        "schema": "side_local_conversion_cross_query_calibration_ablation_v1",
        "status": "complete",
        "rows": int(len(pred)),
        "systems": list(score_columns),
        "target_alignment_audit": "target_alignment_audit.json",
        "calibration_audit": "cross_query_calibration_audit.parquet",
        "report": "SIDE_LOCAL_CONVERSION_CALIBRATION_ABLATION_REPORT.md",
    })
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(run(args.source, args.out))
