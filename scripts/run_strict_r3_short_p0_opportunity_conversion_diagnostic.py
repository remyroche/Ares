#!/usr/bin/env python3
"""Diagnose short M4 cross-era performance using exact P0 path/conversion labels.

This is an offline diagnostic.  It preserves the stored strict-OOF M4 score,
train-p80 admission decision, P0 winner identity, and parent policy.  It adds
no inference feature and creates no new admission rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_short_p0_opportunity_conversion_diagnostic_v1"
SIDE = "short"
M4 = "M4"
IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
ERAS = (
    ("2024", pd.Timestamp("2024-05-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    ("2025", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC")),
    ("2026", pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC")),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def _era(values: pd.Series) -> pd.Series:
    result = pd.Series(index=values.index, dtype="string")
    ts = _utc(values)
    for name, start, end in ERAS:
        result.loc[ts.ge(start) & ts.lt(end)] = name
    return result


def _read_m4(roots: Iterable[Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    columns = [
        *IDENTITY, "arm", "held_month", "expected_net_bps", "raw_meta_score", "train_p80_expected_bps",
        "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps",
    ]
    for root in roots:
        path = root / "short_absolute_conversion_oof_predictions.parquet"
        manifest = root / "run_manifest.json"
        if not path.exists() or not manifest.exists():
            raise FileNotFoundError(f"not an immutable absolute-conversion artifact: {root}")
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["arm"].astype(str).eq(M4)].copy()
        if frame.empty:
            raise ValueError(f"absolute-conversion root has no {M4} OOF rows: {root}")
        frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
        frame["policy_label_available_at"] = _utc(frame["policy_label_available_at"])
        pieces.append(frame)
        hashes[str(root.resolve())] = _sha256(manifest)
    result = pd.concat(pieces, ignore_index=True)
    if result.candidate_id.duplicated().any() or not result.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("M4 OOF sources do not form a unique short-only population")
    result["m4_p80_admitted"] = (
        pd.to_numeric(result["expected_net_bps"], errors="coerce")
        >= pd.to_numeric(result["train_p80_expected_bps"], errors="coerce")
    )
    return result, hashes


def _read_rich_labels(root: Path) -> tuple[pd.DataFrame, str]:
    manifest = root / "run_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    columns = [
        *IDENTITY, "__label_available_at__", "rich_path_label_valid", "rich_path_target_invalid",
        "mfe_1h_bps", "mfe_3h_bps", "mfe_12h_bps", "reached_100bps", "reached_200bps",
        "time_to_100bps_minutes", "mae_before_100bps_bps", "policy_capture_ratio_cost_clear",
        "policy_regret_bps", "policy_giveback_bps", "policy_giveback_ratio", "policy_exit_reason",
        "policy_net_bps", "policy_gross_bps", "policy_conversion_category",
    ]
    paths = sorted(root.glob("parts/month=*/side=short.parquet"))
    if not paths:
        raise FileNotFoundError(f"no rich path parts under {root}")
    result = pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)
    result["__decision_ts__"] = _utc(result["__decision_ts__"])
    result["__label_available_at__"] = _utc(result["__label_available_at__"])
    if result.candidate_id.duplicated().any() or not result.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("rich-path source does not form a unique short-only identity set")
    return result, _sha256(manifest)


def _valid(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["m4_p80_admitted"].astype("boolean").fillna(False).astype(bool)
        & frame["policy_path_valid"].astype("boolean").fillna(False).astype(bool)
        & frame["rich_path_label_valid"].astype("boolean").fillna(False).astype(bool)
        & ~frame["rich_path_target_invalid"].astype("boolean").fillna(True).astype(bool)
        & pd.to_numeric(frame["policy_net_bps"], errors="coerce").notna()
    )


def _ratio(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _summary(frame: pd.DataFrame) -> dict[str, float]:
    reached100 = frame["reached_100bps"].astype("boolean").fillna(False).astype(bool)
    reached200 = frame["reached_200bps"].astype("boolean").fillna(False).astype(bool)
    time = pd.to_numeric(frame.loc[reached100, "time_to_100bps_minutes"], errors="coerce")
    mae = pd.to_numeric(frame.loc[reached100, "mae_before_100bps_bps"], errors="coerce")
    reason = frame["policy_exit_reason"].astype("string")
    return {
        "trades": float(len(frame)),
        "mean_mfe_1h_bps": float(pd.to_numeric(frame["mfe_1h_bps"], errors="coerce").mean()),
        "median_mfe_1h_bps": float(pd.to_numeric(frame["mfe_1h_bps"], errors="coerce").median()),
        "mean_mfe_3h_bps": float(pd.to_numeric(frame["mfe_3h_bps"], errors="coerce").mean()),
        "median_mfe_3h_bps": float(pd.to_numeric(frame["mfe_3h_bps"], errors="coerce").median()),
        "mean_mfe_12h_bps": float(pd.to_numeric(frame["mfe_12h_bps"], errors="coerce").mean()),
        "median_mfe_12h_bps": float(pd.to_numeric(frame["mfe_12h_bps"], errors="coerce").median()),
        "pct_ever_100bps": _ratio(reached100),
        "pct_ever_200bps": _ratio(reached200),
        "median_time_to_100bps_minutes": float(time.median()),
        "mean_mae_before_100bps": float(mae.mean()),
        "median_mae_before_100bps": float(mae.median()),
        "mean_capture_ratio": float(pd.to_numeric(frame["policy_capture_ratio_cost_clear"], errors="coerce").mean()),
        "median_capture_ratio": float(pd.to_numeric(frame["policy_capture_ratio_cost_clear"], errors="coerce").median()),
        "mean_policy_regret_bps": float(pd.to_numeric(frame["policy_regret_bps"], errors="coerce").mean()),
        "median_policy_regret_bps": float(pd.to_numeric(frame["policy_regret_bps"], errors="coerce").median()),
        "mean_giveback_bps": float(pd.to_numeric(frame["policy_giveback_bps"], errors="coerce").mean()),
        "mean_policy_gross_bps": float(pd.to_numeric(frame["policy_gross_bps"], errors="coerce").mean()),
        "mean_policy_net_bps": float(pd.to_numeric(frame["policy_net_bps"], errors="coerce").mean()),
        "stop_rate": _ratio(reason.eq("stop_loss")),
        "timeout_rate": _ratio(reason.eq("timeout")),
        "trailing_rate": _ratio(reason.eq("trailing")),
    }


def _single_changepoint(monthly: pd.DataFrame, *, column: str) -> dict[str, object]:
    local = monthly.loc[pd.to_numeric(monthly[column], errors="coerce").notna(), ["month", column]].copy()
    values = pd.to_numeric(local[column], errors="coerce").to_numpy(float)
    if len(values) < 6:
        return {"metric": column, "status": "insufficient_months"}
    baseline = float(np.square(values - values.mean()).sum())
    candidates: list[tuple[float, int]] = []
    for split in range(3, len(values) - 2):
        left, right = values[:split], values[split:]
        sse = float(np.square(left - left.mean()).sum() + np.square(right - right.mean()).sum())
        candidates.append((sse, split))
    best_sse, split = min(candidates)
    return {
        "metric": column,
        "status": "descriptive_single_split",
        "break_before_month": str(local["month"].iat[split]),
        "pre_months": int(split),
        "post_months": int(len(values) - split),
        "pre_mean": float(values[:split].mean()),
        "post_mean": float(values[split:].mean()),
        "sse_reduction": baseline - best_sse,
        "sse_reduction_fraction": float((baseline - best_sse) / baseline) if baseline > 0.0 else float("nan"),
    }


def _markdown(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown renderer for immutable reports."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows: list[str] = []
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append("" if not np.isfinite(value) else f"{float(value):.3f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, rule, *rows])


def _report(out: Path, *, era: pd.DataFrame, change: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# Short P0 M4 opportunity versus conversion diagnostic",
        "",
        "This is a target-label diagnostic only. It preserves the stored strict-OOF M4 p80 admission and does not create a new rule.",
        "",
        "## Cross-era M4 p80 decomposition",
        "",
        _markdown(era),
        "",
        "## Descriptive monthly changepoints",
        "",
        _markdown(change),
        "",
        "## Lineage and invariants",
        "",
        "```json",
        json.dumps(audit, sort_keys=True, indent=2),
        "```",
        "",
        "Interpretation: lower MFE/reachability denotes opportunity-generation weakness; similar MFE with lower capture or higher regret denotes policy conversion weakness. The changepoints are descriptive and cannot be used as live gates.",
    ]
    (out / "SHORT_P0_M4_OPPORTUNITY_CONVERSION_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n")


def run(*, absolute_roots: list[Path], rich_labels_root: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    m4, m4_hashes = _read_m4(absolute_roots)
    labels, label_hash = _read_rich_labels(rich_labels_root)
    merged = m4.merge(labels, on=list(IDENTITY), how="left", validate="one_to_one")
    source_net = pd.to_numeric(merged["p0_canonical_net_bps"], errors="coerce")
    rich_net = pd.to_numeric(merged["policy_net_bps"], errors="coerce")
    parity = source_net.notna() & rich_net.notna()
    if parity.any() and not np.allclose(source_net[parity], rich_net[parity], rtol=0.0, atol=2e-4):
        raise AssertionError("stored M4 policy outcomes do not match the reopened rich-path policy replay")
    if not merged.loc[merged["rich_path_label_valid"].fillna(False), "__label_available_at__"].eq(
        merged.loc[merged["rich_path_label_valid"].fillna(False), "__decision_ts__"] + pd.Timedelta(hours=12)
    ).all():
        raise AssertionError("rich label availability is not decision + 12 hours")
    merged["era"] = _era(merged["__decision_ts__"])
    valid = merged.loc[_valid(merged)].copy()
    if valid.empty:
        raise ValueError("no valid stored-M4 p80 rows overlap rich-path labels")
    era_rows = []
    for name, _start, _end in ERAS:
        block = valid.loc[valid["era"].eq(name)]
        era_rows.append({"era": name, **_summary(block)})
    era_summary = pd.DataFrame(era_rows)
    valid["month"] = valid["__decision_ts__"].dt.strftime("%Y-%m")
    monthly = valid.groupby("month", as_index=False).agg(
        trades=("candidate_id", "size"),
        mean_mfe_12h_bps=("mfe_12h_bps", "mean"),
        pct_ever_200bps=("reached_200bps", "mean"),
        mean_capture_ratio=("policy_capture_ratio_cost_clear", "mean"),
        mean_policy_net_bps=("policy_net_bps", "mean"),
    )
    changes = pd.DataFrame([
        _single_changepoint(monthly, column=column)
        for column in ("mean_mfe_12h_bps", "pct_ever_200bps", "mean_capture_ratio", "mean_policy_net_bps")
    ])
    categories = valid.groupby(["era", "policy_conversion_category"], dropna=False).agg(
        trades=("candidate_id", "size"), mean_policy_net_bps=("policy_net_bps", "mean")
    ).reset_index()
    categories["fraction_within_era"] = categories["trades"] / categories.groupby("era")["trades"].transform("sum")
    out.mkdir(parents=True)
    merged.to_parquet(out / "m4_p80_rich_path_join.parquet", index=False, compression="zstd")
    era_summary.to_parquet(out / "cross_era_opportunity_conversion_metrics.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "monthly_opportunity_conversion_metrics.parquet", index=False, compression="zstd")
    changes.to_parquet(out / "descriptive_changepoints.parquet", index=False, compression="zstd")
    categories.to_parquet(out / "conversion_category_by_era.parquet", index=False, compression="zstd")
    audit = {
        "schema": SCHEMA,
        "status": "complete_diagnostic_only",
        "side": SIDE,
        "control": "stored strict-OOF M4 ordinal policy-margin score; causal train-p80 admission",
        "m4_rows": int(len(m4)),
        "m4_p80_rows": int(m4["m4_p80_admitted"].sum()),
        "rich_label_overlap_rows": int(merged["rich_path_label_valid"].fillna(False).sum()),
        "valid_m4_p80_rows": int(len(valid)),
        "outcome_parity_rows": int(parity.sum()),
        "max_policy_net_abs_delta_bps": float((source_net[parity] - rich_net[parity]).abs().max()) if parity.any() else float("nan"),
        "m4_manifest_hashes": m4_hashes,
        "rich_labels_manifest_sha256": label_hash,
        "inference_change": "none; all rich path fields are realised labels only",
    }
    (out / "run_manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    _report(out, era=era_summary, change=changes, audit=audit)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute-root", type=Path, action="append", required=True)
    parser.add_argument("--rich-labels-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(absolute_roots=[path.resolve() for path in args.absolute_root], rich_labels_root=args.rich_labels_root.resolve(), out=args.out.resolve()))


if __name__ == "__main__":
    main()
