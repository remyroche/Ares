#!/usr/bin/env python3
"""Recover selected strict-R3 long supportive-path label months safely.

This is an offline recovery producer for a narrowly scoped storage failure:
some original label parts were offloaded by macOS and the old complete source
panel was intentionally archived.  It reconstructs only the target-only
fields consumed by the P8u Meta research stack from three surviving sources:

* the immutable target-free Base candidates (identity only);
* the immutable canonical policy ledger (provenance only);
* frozen historical 15-minute OHLCV (path supervision only).

The output is a new immutable, month-scoped artifact.  It never alters the
original label root and cannot contribute any path labels to inference data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import materialize_strict_r3_long_supportive_path_labels as parent  # noqa: E402


SCHEMA = "strict_r3_p8u_supportive_path_label_recovery_v1"
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
POLICY_COLUMNS = (
    "candidate_id", "policy_path_valid", "policy_label_available_ts",
    "policy_entry_price", "policy_gross_bps", "policy_net_bps",
    "policy_exit_reason", "policy_cost_bps",
)
H12_COLUMNS = (
    "h12_label_valid", "h12_label_available_ts",
    "h12_tp6_sl4_gross_bps", "h12_tp6_sl4_net_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(values: list[str]) -> tuple[pd.Timestamp, ...]:
    parsed = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in values)
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("--months must contain one or more distinct YYYY-MM values")
    return tuple(sorted(parsed))


def _candidate_month(root: Path, month: pd.Timestamp) -> pd.DataFrame:
    path = root / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    fields = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "side_name"])
    fields["__decision_ts__"] = pd.to_datetime(fields["__decision_ts__"], utc=True, errors="raise")
    if fields.candidate_id.duplicated().any() or not fields.side_name.eq("long").all():
        raise AssertionError(f"{month:%Y-%m}: invalid target-free candidate identity")
    parsed = fields.candidate_id.astype(str).str.split("|", n=2, expand=True)
    if parsed.shape[1] != 3 or not parsed.iloc[:, 1].eq("long").all():
        raise AssertionError(f"{month:%Y-%m}: cannot reconstruct long candidate identity")
    fields["__symbol__"] = parsed.iloc[:, 0].astype(str)
    fields["__ts__"] = pd.to_datetime(parsed.iloc[:, 2], utc=True, errors="raise")
    if not fields.__decision_ts__.eq(fields.__ts__ + pd.Timedelta(hours=1)).all():
        raise AssertionError(f"{month:%Y-%m}: candidate signal/decision timing is not the frozen one-hour contract")
    return fields.loc[:, list(IDENTITY)].copy()


def _policy(path: Path) -> pd.DataFrame:
    values = pd.read_parquet(path, columns=list(POLICY_COLUMNS)).copy()
    if values.candidate_id.duplicated().any():
        raise AssertionError("canonical policy ledger has duplicate candidate IDs")
    values["policy_label_available_ts"] = pd.to_datetime(values["policy_label_available_ts"], utc=True, errors="coerce")
    return values


def _source_month(candidate_root: Path, policy: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    candidates = _candidate_month(candidate_root, month)
    values = candidates.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(values) != len(candidates):
        raise AssertionError(f"{month:%Y-%m}: policy join changed candidate identity count")
    # H12 TP6/SL4 provenance is retained by the historical parent artifact but
    # is not consumed by the Meta pipeline's PATH_COLUMNS.  Null placeholders
    # preserve that no alternative outcome source is silently introduced.
    values["h12_label_valid"] = pd.NA
    values["h12_label_available_ts"] = pd.NaT
    values["h12_tp6_sl4_gross_bps"] = float("nan")
    values["h12_tp6_sl4_net_bps"] = float("nan")
    expected = [*IDENTITY, *POLICY_COLUMNS[1:], *H12_COLUMNS]
    if list(values.columns) != expected:
        raise AssertionError(f"{month:%Y-%m}: recovery source ordering drift")
    return values


def _assert_frozen_bars_readable(source: pd.DataFrame, bars_root: Path, month: pd.Timestamp) -> None:
    """Fail before writing when a required archived bar part is unavailable."""
    unavailable: list[str] = []
    for symbol in sorted(source["__symbol__"].astype(str).unique()):
        path = parent._bar_path(bars_root, symbol)
        try:
            pq.ParquetFile(path).schema_arrow
        except Exception as exc:  # pragma: no cover - storage dependent
            unavailable.append(f"{path}: {type(exc).__name__}: {exc}")
    if unavailable:
        preview = "\n".join(unavailable[:16])
        raise RuntimeError(
            f"{month:%Y-%m}: required frozen 15-minute source parts are unavailable; "
            f"restore them before label recovery:\n{preview}"
        )


def _build_bar_overlay(*, base_root: Path, override_root: Path | None, out: Path) -> Path:
    """Create a link-only source view with explicit recovered-bar precedence."""
    if override_root is None:
        return base_root
    view = out / "frozen_bar_source_view"
    view.mkdir()
    sources = {path.name: path.resolve() for path in base_root.glob("*_15m.parquet")}
    for path in override_root.glob("*_15m.parquet"):
        sources[path.name] = path.resolve()
    if not sources:
        raise AssertionError("bar overlay sources are empty")
    for name, target in sorted(sources.items()):
        (view / name).symlink_to(target)
    return view


def run(*, candidate_root: Path, policy_path: Path, bars_root: Path, bars_override_root: Path | None,
        months: tuple[pd.Timestamp, ...], out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable recovery output exists: {out}")
    policy = _policy(policy_path)
    records: list[dict[str, Any]] = []
    out.mkdir(parents=True)
    effective_bars_root = _build_bar_overlay(base_root=bars_root, override_root=bars_override_root, out=out)
    for month in months:
        source = _source_month(candidate_root, policy, month)
        _assert_frozen_bars_readable(source, effective_bars_root, month)
        sidecar, record = parent._materialize_month(source, effective_bars_root)
        record["month"] = f"{month:%Y-%m}"
        destination = out / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sidecar.to_parquet(destination, index=False, compression="zstd")
        records.append(record)
    pd.DataFrame(records).to_parquet(out / "coverage_by_month.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "candidate_identities_are_immutable_target_free_base_rows": True,
        "decision_time_is_reconstructed_only_from_frozen_candidate_id": True,
        "policy_ledger_is_provenance_only": True,
        "labels_use_frozen_15m_post_decision_paths": True,
        "atr_uses_pre_decision_wilder14_from_parent_materializer": True,
        "recovery_is_target_only_and_never_an_inference_feature_source": True,
        "h12_provenance_is_not_reconstructed_or_substituted": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline target-only recovery for explicitly listed offloaded path-label months",
        "months": [f"{month:%Y-%m}" for month in months],
        "parent_schema": parent.SCHEMA,
        "candidate_root": str(candidate_root.resolve()),
        "candidate_root_note": "target-free identity source; parsed signal timestamp must equal decision minus one hour",
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": _sha(policy_path),
        "bars_root": str(bars_root.resolve()),
        "bars_override_root": str(bars_override_root.resolve()) if bars_override_root else None,
        "effective_bars_root": str(effective_bars_root.resolve()),
        "bar_source_precedence": "explicit recovered exact filename overrides base cache; all other files are link-only base references",
        "recovery_h12_provenance": "null; irrelevant to P8u Meta PATH_COLUMNS and never substituted",
        "coverage": records,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--bars-root", type=Path, default=ROOT / "15m_ohlcv_perp")
    parser.add_argument("--bars-override-root", type=Path, help="small immutable recovery cache that takes filename-level precedence")
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(
        candidate_root=args.candidate_root.resolve(), policy_path=args.policy.resolve(),
        bars_root=args.bars_root.resolve(),
        bars_override_root=args.bars_override_root.resolve() if args.bars_override_root else None,
        months=_months(list(args.months)), out=args.out.resolve(),
    ))


if __name__ == "__main__":
    main()
