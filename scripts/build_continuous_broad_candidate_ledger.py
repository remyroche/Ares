#!/usr/bin/env python3
"""Build or audit a continuous broad simple-policy candidate ledger.

The dynamic HR-surprise replay needs one continuous, schema-compatible broad
candidate ledger before monthly Optuna runs are trustworthy.  This tool scans
available artifacts, optionally assembles a best-effort non-overlapping ledger,
and writes an explicit trust verdict instead of silently mixing unrelated
windows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/continuous_broad_candidate_ledger")
EARLY_UTC = pd.Timestamp("1900-01-01T00:00:00Z")
LATE_UTC = pd.Timestamp("2100-01-01T00:00:00Z")


@dataclass(frozen=True)
class CandidateMetadata:
    path: Path
    root: Path
    row_count: int
    timestamp_min: pd.Timestamp | None
    timestamp_max: pd.Timestamp | None
    timestamp_count: int
    heads: tuple[str, ...]
    columns: tuple[str, ...]
    contract_family: str
    score_columns: tuple[str, ...]
    rank_columns: tuple[str, ...]
    p_hit_columns: tuple[str, ...]
    return_columns: tuple[str, ...]
    missing_essential_columns: tuple[str, ...]
    eligible_for_assembly: bool
    rejection_reasons: tuple[str, ...]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _infer_head(strategy_id: Any) -> str:
    value = str(strategy_id)
    for head in HEADS:
        if value.startswith(head):
            return head
    return "unknown"


def _artifact_root(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "simple_policy_optimiser" else path.parent


def _contract_family(root: Path, columns: set[str]) -> str:
    name = root.name.lower()
    if "market_state_controller" in name:
        return "market_state_controller"
    if "t1_global_rank" in name or "global_rank" in name:
        return "t1_global_rank"
    if "t1_timestamp_rank" in name or "timestamp_rank" in name:
        return "t1_timestamp_rank"
    if "anchor_scored" in name:
        return "anchor_scored"
    if "anchor_policy" in name:
        return "anchor_policy"
    if "native" in name or "reliability_blend" in name:
        return "native_reliability_blend"
    if "no_mkt4" in name:
        return "no_mkt4_policy"
    if "calibrated_score" in columns:
        return "calibrated_candidate_ledger"
    return "unknown"


def _discover_paths(search_roots: list[Path], *, include_market_state_controller: bool) -> list[Path]:
    names = {"simple_policy_candidates_broad.parquet"}
    if include_market_state_controller:
        names.add("market_state_controller_candidates_broad.parquet")
    out: dict[str, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.name in names:
            out[str(root)] = root
            continue
        for name in names:
            for path in root.rglob(name):
                out[str(path)] = path
    return [out[key] for key in sorted(out)]


def _available_columns(path: Path) -> tuple[str, ...]:
    return tuple(pq.ParquetFile(path).schema_arrow.names)


def _read_candidate_metadata(path: Path) -> CandidateMetadata:
    root = _artifact_root(path)
    columns = _available_columns(path)
    column_set = set(columns)
    parquet_file = pq.ParquetFile(path)
    row_count = int(parquet_file.metadata.num_rows)
    reasons: list[str] = []
    read_cols = [col for col in ("timestamp", "head", "strategy_id") if col in column_set]
    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None
    timestamp_count = 0
    heads: tuple[str, ...] = tuple()
    if "timestamp" not in column_set:
        reasons.append("missing_timestamp")
    if not ({"head", "strategy_id"} & column_set):
        reasons.append("missing_head_or_strategy_id")
    try:
        if read_cols:
            preview = pd.read_parquet(path, columns=read_cols)
            if "timestamp" in preview:
                ts = pd.to_datetime(preview["timestamp"], utc=True, errors="coerce").dropna()
                if not ts.empty:
                    timestamp_min = pd.Timestamp(ts.min())
                    timestamp_max = pd.Timestamp(ts.max())
                    timestamp_count = int(ts.nunique())
                else:
                    reasons.append("no_finite_timestamps")
            if "head" in preview:
                head_series = preview["head"].astype(str)
            elif "strategy_id" in preview:
                head_series = preview["strategy_id"].map(_infer_head)
            else:
                head_series = pd.Series(dtype=str)
            heads = tuple(sorted(h for h in head_series.dropna().astype(str).unique() if h in HEADS))
    except Exception as exc:  # pragma: no cover - defensive scan path
        reasons.append(f"metadata_read_failed:{type(exc).__name__}")

    score_columns = tuple(col for col in ("normalized_rank_score", "calibrated_score", "reliability_blend_score") if col in column_set)
    rank_columns = tuple(col for col in ("policy_rank_pct", "rank_pct", "strategy_rank_pct") if col in column_set)
    p_hit_columns = tuple(
        col
        for col in (
            "simple_policy_calibrated_good_trade_prob",
            "calibrated_score",
            "reliability_blend_score",
            "normalized_rank_score",
        )
        if col in column_set
    )
    return_columns = tuple(col for col in ("net_return", "fixed_return_net_after_cost", "gross_return") if col in column_set)
    missing: list[str] = []
    if not score_columns:
        missing.append("score_column")
    if not rank_columns:
        missing.append("rank_column")
    if not p_hit_columns:
        missing.append("p_hit_column")
    if not return_columns:
        missing.append("return_column")
    reasons.extend(f"missing_{col}" for col in missing)
    if row_count <= 0:
        reasons.append("empty_ledger")
    if not heads:
        reasons.append("no_known_heads")
    eligible = not reasons and timestamp_min is not None and timestamp_max is not None
    return CandidateMetadata(
        path=path,
        root=root,
        row_count=row_count,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        timestamp_count=timestamp_count,
        heads=heads,
        columns=columns,
        contract_family=_contract_family(root, column_set),
        score_columns=score_columns,
        rank_columns=rank_columns,
        p_hit_columns=p_hit_columns,
        return_columns=return_columns,
        missing_essential_columns=tuple(missing),
        eligible_for_assembly=bool(eligible),
        rejection_reasons=tuple(dict.fromkeys(reasons)),
    )


def _metadata_frame(items: list[CandidateMetadata]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in items:
        duration_days = None
        if item.timestamp_min is not None and item.timestamp_max is not None:
            duration_days = (item.timestamp_max - item.timestamp_min).total_seconds() / 86400.0
        rows.append(
            {
                "path": str(item.path),
                "root": str(item.root),
                "row_count": item.row_count,
                "timestamp_min": item.timestamp_min.isoformat() if item.timestamp_min is not None else "",
                "timestamp_max": item.timestamp_max.isoformat() if item.timestamp_max is not None else "",
                "duration_days": duration_days,
                "timestamp_count": item.timestamp_count,
                "heads": ",".join(item.heads),
                "head_count": len(item.heads),
                "contract_family": item.contract_family,
                "score_columns": ",".join(item.score_columns),
                "rank_columns": ",".join(item.rank_columns),
                "p_hit_columns": ",".join(item.p_hit_columns),
                "return_columns": ",".join(item.return_columns),
                "missing_essential_columns": ",".join(item.missing_essential_columns),
                "eligible_for_assembly": item.eligible_for_assembly,
                "rejection_reasons": ";".join(item.rejection_reasons),
            }
        )
    return pd.DataFrame(rows)


def _select_best_effort(
    metadata: list[CandidateMetadata],
    *,
    allow_mixed_contracts: bool,
    contract_family: str | None,
    max_gap_hours: float,
) -> list[CandidateMetadata]:
    candidates = [item for item in metadata if item.eligible_for_assembly]
    if contract_family:
        candidates = [item for item in candidates if item.contract_family == contract_family]
    if not candidates:
        return []
    if not allow_mixed_contracts and not contract_family:
        family_scores: dict[str, tuple[float, int, pd.Timestamp]] = {}
        for item in candidates:
            assert item.timestamp_min is not None and item.timestamp_max is not None
            span = (item.timestamp_max - item.timestamp_min).total_seconds()
            current = family_scores.get(item.contract_family)
            score = (span, int(item.row_count), item.timestamp_max)
            if current is None or score > current:
                family_scores[item.contract_family] = score
        best_family = max(family_scores.items(), key=lambda kv: kv[1])[0]
        candidates = [item for item in candidates if item.contract_family == best_family]

    max_gap = pd.Timedelta(hours=float(max_gap_hours))
    selected: list[CandidateMetadata] = []
    cursor: pd.Timestamp | None = None
    guard = 0
    while guard < len(candidates) + 5:
        guard += 1
        if cursor is None:
            earliest = min(item.timestamp_min for item in candidates if item.timestamp_min is not None)
            pool = [item for item in candidates if item.timestamp_min is not None and item.timestamp_min <= earliest + max_gap]
        else:
            pool = [
                item
                for item in candidates
                if item.timestamp_min is not None
                and item.timestamp_max is not None
                and item.timestamp_min <= cursor + max_gap
                and item.timestamp_max > cursor
                and item.path not in {selected_item.path for selected_item in selected}
            ]
        if not pool:
            break
        chosen = max(
            pool,
            key=lambda item: (
                item.timestamp_max or EARLY_UTC,
                len(item.heads),
                item.row_count,
                -len(str(item.path)),
            ),
        )
        selected.append(chosen)
        cursor = max(cursor, chosen.timestamp_max) if cursor is not None and chosen.timestamp_max is not None else chosen.timestamp_max
    return selected


def _read_for_assembly(item: CandidateMetadata) -> pd.DataFrame:
    frame = pd.read_parquet(item.path).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "head" not in frame.columns and "strategy_id" in frame.columns:
        frame["head"] = frame["strategy_id"].map(_infer_head)
    frame["source_candidate_path"] = str(item.path)
    frame["source_artifact_root"] = str(item.root)
    frame["source_contract_family"] = item.contract_family
    return frame


def _assemble_ledger(selected: list[CandidateMetadata]) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame()
    frame = pd.concat([_read_for_assembly(item) for item in selected], ignore_index=True, sort=False)
    frame = frame.dropna(subset=["timestamp"]).copy()
    frame = frame.loc[frame["head"].astype(str).isin(HEADS)].copy()
    dedupe_cols = [
        col
        for col in (
            "timestamp",
            "head",
            "strategy_id",
            "symbol",
            "side",
            "normalized_rank_score",
            "calibrated_score",
            "net_return",
        )
        if col in frame.columns
    ]
    if dedupe_cols:
        frame = frame.drop_duplicates(subset=dedupe_cols, keep="first")
    sort_cols = [col for col in ("timestamp", "head", "normalized_rank_score", "calibrated_score") if col in frame.columns]
    ascending = [True, True] + [False] * max(0, len(sort_cols) - 2)
    frame = frame.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return frame


def _coverage_frame(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    work = ledger[["timestamp", "head"]].copy()
    work["date"] = pd.to_datetime(work["timestamp"], utc=True).dt.floor("D")
    return (
        work.groupby(["date", "head"], sort=True)
        .size()
        .rename("row_count")
        .reset_index()
    )


def _selected_frame(selected: list[CandidateMetadata]) -> pd.DataFrame:
    frame = _metadata_frame(selected)
    if frame.empty:
        return frame
    frame.insert(0, "selection_order", np.arange(1, len(frame) + 1))
    prev_end: pd.Timestamp | None = None
    gaps: list[float | None] = []
    overlaps: list[float | None] = []
    for item in selected:
        if prev_end is None or item.timestamp_min is None:
            gaps.append(None)
            overlaps.append(None)
        else:
            delta_hours = (item.timestamp_min - prev_end).total_seconds() / 3600.0
            gaps.append(max(delta_hours, 0.0))
            overlaps.append(max(-delta_hours, 0.0))
        if item.timestamp_max is not None:
            prev_end = item.timestamp_max if prev_end is None else max(prev_end, item.timestamp_max)
    frame["gap_from_previous_hours"] = gaps
    frame["overlap_with_previous_hours"] = overlaps
    return frame


def _trust_verdict(
    ledger: pd.DataFrame,
    selected: list[CandidateMetadata],
    *,
    target_min_days: float,
    late_june_cutoff: pd.Timestamp,
    max_gap_hours: float,
) -> dict[str, Any]:
    if ledger.empty:
        return {
            "trusted_for_monthly_optuna": False,
            "reasons": ["empty_assembled_ledger"],
        }
    ts = pd.to_datetime(ledger["timestamp"], utc=True, errors="coerce").dropna()
    heads = sorted(ledger["head"].dropna().astype(str).unique())
    start = pd.Timestamp(ts.min())
    end = pd.Timestamp(ts.max())
    span_days = (end - start).total_seconds() / 86400.0
    selected_view = _selected_frame(selected)
    gaps = pd.to_numeric(selected_view.get("gap_from_previous_hours"), errors="coerce").dropna()
    max_gap = float(gaps.max()) if not gaps.empty else 0.0
    reasons: list[str] = []
    if span_days < float(target_min_days):
        reasons.append(f"span_days<{target_min_days:g}")
    if end < late_june_cutoff:
        reasons.append(f"does_not_reach_{late_june_cutoff.isoformat()}")
    missing_heads = sorted(set(HEADS) - set(heads))
    if missing_heads:
        reasons.append("missing_heads:" + ",".join(missing_heads))
    if max_gap > float(max_gap_hours):
        reasons.append(f"max_gap_hours>{max_gap_hours:g}")
    contract_families = sorted({item.contract_family for item in selected})
    if len(contract_families) > 1:
        reasons.append("mixed_contract_families:" + ",".join(contract_families))
    return {
        "trusted_for_monthly_optuna": len(reasons) == 0,
        "reasons": reasons,
        "ledger_start": start.isoformat(),
        "ledger_end": end.isoformat(),
        "span_days": span_days,
        "row_count": int(len(ledger)),
        "timestamp_count": int(ts.nunique()),
        "heads": heads,
        "max_gap_hours": max_gap,
        "contract_families": contract_families,
        "selected_source_count": int(len(selected)),
        "target_min_days": float(target_min_days),
        "late_june_cutoff": late_june_cutoff.isoformat(),
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    frame.to_parquet(path, index=False)


def _write_report(
    output_dir: Path,
    audit: pd.DataFrame,
    selected: pd.DataFrame,
    verdict: dict[str, Any],
) -> None:
    latest = ""
    if not audit.empty:
        latest_ts = pd.to_datetime(audit["timestamp_max"], utc=True, errors="coerce").dropna()
        latest = latest_ts.max().isoformat() if not latest_ts.empty else ""
    lines = [
        "# Continuous Broad Candidate Ledger Audit",
        "",
        f"- Audited broad ledgers: `{len(audit)}`",
        f"- Selected sources: `{len(selected)}`",
        f"- Latest audited timestamp: `{latest}`",
        f"- Trusted for monthly Optuna: `{bool(verdict.get('trusted_for_monthly_optuna'))}`",
        f"- Verdict reasons: `{';'.join(verdict.get('reasons') or []) or 'none'}`",
        "",
        "## Selected Sources",
        "",
    ]
    if selected.empty:
        lines.append("_No selected sources._")
    else:
        view_cols = [
            "selection_order",
            "timestamp_min",
            "timestamp_max",
            "duration_days",
            "row_count",
            "heads",
            "contract_family",
            "gap_from_previous_hours",
            "path",
        ]
        view = selected[[col for col in view_cols if col in selected.columns]].copy()
        lines.extend(_markdown_table(view).splitlines())
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{output_dir / 'candidate_ledger_audit.csv'}`",
            f"- `{output_dir / 'selected_candidate_sources.csv'}`",
            f"- `{output_dir / 'continuous_broad_candidate_ledger.parquet'}`",
            f"- `{output_dir / 'continuous_broad_candidate_coverage.csv'}`",
            f"- `{output_dir / 'continuous_broad_candidate_ledger_manifest.json'}`",
            "",
        ]
    )
    (output_dir / "continuous_broad_candidate_ledger_report.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.fillna("").astype(str)
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(row[col].replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", action="append", type=Path, default=None)
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--allow-mixed-contracts", action="store_true")
    parser.add_argument("--contract-family", default=None)
    parser.add_argument("--include-market-state-controller", action="store_true")
    parser.add_argument("--max-gap-hours", type=float, default=6.0)
    parser.add_argument("--target-min-days", type=float, default=180.0)
    parser.add_argument("--late-june-cutoff", default="2026-06-25T00:00:00Z")
    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--fail-if-untrusted", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_re = re.compile(args.include_regex) if args.include_regex else None
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None
    if args.input:
        paths = [Path(path) for path in args.input]
    else:
        paths = _discover_paths(
            list(args.search_root or [Path("data_perp/artifacts")]),
            include_market_state_controller=bool(args.include_market_state_controller),
        )
    if include_re:
        paths = [path for path in paths if include_re.search(str(path))]
    if exclude_re:
        paths = [path for path in paths if not exclude_re.search(str(path))]
    if not paths:
        raise SystemExit("No broad candidate ledgers were found")

    metadata = [_read_candidate_metadata(path) for path in paths]
    audit = _metadata_frame(metadata).sort_values(["timestamp_min", "timestamp_max", "row_count"], ascending=[True, False, False])
    _write_frame(output_dir / "candidate_ledger_audit.parquet", audit)

    if args.scan_only:
        selected: list[CandidateMetadata] = []
        ledger = pd.DataFrame()
    elif args.input:
        selected = [item for item in metadata if item.path in set(paths) and item.eligible_for_assembly]
        selected = sorted(selected, key=lambda item: (item.timestamp_min or LATE_UTC, item.timestamp_max or EARLY_UTC))
        ledger = _assemble_ledger(selected)
    else:
        selected = _select_best_effort(
            metadata,
            allow_mixed_contracts=bool(args.allow_mixed_contracts),
            contract_family=args.contract_family,
            max_gap_hours=float(args.max_gap_hours),
        )
        ledger = _assemble_ledger(selected)

    selected_df = _selected_frame(selected)
    _write_frame(output_dir / "selected_candidate_sources.parquet", selected_df)
    coverage = _coverage_frame(ledger)
    if not coverage.empty:
        _write_frame(output_dir / "continuous_broad_candidate_coverage.parquet", coverage)
    if not ledger.empty:
        _write_frame(output_dir / "continuous_broad_candidate_ledger.parquet", ledger)

    late_june_cutoff = pd.Timestamp(args.late_june_cutoff)
    if late_june_cutoff.tzinfo is None:
        late_june_cutoff = late_june_cutoff.tz_localize("UTC")
    else:
        late_june_cutoff = late_june_cutoff.tz_convert("UTC")
    verdict = _trust_verdict(
        ledger,
        selected,
        target_min_days=float(args.target_min_days),
        late_june_cutoff=late_june_cutoff,
        max_gap_hours=float(args.max_gap_hours),
    )
    manifest = {
        "generated_by": "build_continuous_broad_candidate_ledger",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_roots": [str(path) for path in list(args.search_root or [Path("data_perp/artifacts")])],
        "input_paths": [str(path) for path in args.input],
        "scan_only": bool(args.scan_only),
        "allow_mixed_contracts": bool(args.allow_mixed_contracts),
        "contract_family": args.contract_family,
        "max_gap_hours": float(args.max_gap_hours),
        "target_min_days": float(args.target_min_days),
        "late_june_cutoff": late_june_cutoff.isoformat(),
        "audited_ledger_count": int(len(audit)),
        "eligible_ledger_count": int(audit["eligible_for_assembly"].sum()) if "eligible_for_assembly" in audit else 0,
        "selected_source_count": int(len(selected)),
        "assembled_ledger_path": str(output_dir / "continuous_broad_candidate_ledger.parquet") if not ledger.empty else "",
        "coverage_path": str(output_dir / "continuous_broad_candidate_coverage.parquet") if not coverage.empty else "",
        "verdict": verdict,
    }
    (output_dir / "continuous_broad_candidate_ledger_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir, audit, selected_df, verdict)
    print(json.dumps(_json_safe(verdict), indent=2))
    if bool(args.fail_if_untrusted) and not bool(verdict.get("trusted_for_monthly_optuna")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
