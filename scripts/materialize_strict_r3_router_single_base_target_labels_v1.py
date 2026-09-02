#!/usr/bin/env python3
"""Materialise label-only targets for the router-first single-base funnel.

This producer is deliberately separate from every target-free feature or score
panel.  It retains the exact point-in-time Router-50 candidate identities and
joins canonical rich-policy outcomes only to build supervised labels:

* T0: fixed rich-policy ordinal bins;
* T1: raw rich-policy magnitude (fold-specific bins are fitted by the scorer);
* T2: policy magnitude divided by sqrt(decision-time ATR bps);
* T3: policy magnitude divided by decision-time ATR bps.

The normalised targets use the Wilder-14 decision-time ATR persisted by the
supportive-path label sidecar.  Invalid policy paths, absent ATR warm-up, and
incomplete supportive paths are invalid supervision; they are never converted
to an economic-loss label.  None of the produced columns may be inference
features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_router_single_base_target_labels_v1"
IDENTITY = ("candidate_id", "__decision_ts__", "side_name")


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _utc(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.date_range(start, end, freq="MS", inclusive="left", tz="UTC"))


def _source_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / f"month={month:%Y-%m}" / "scores_features.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _router_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / "target_free_scores" / f"month={month:%Y-%m}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _router_top_half(router: pd.DataFrame) -> pd.DataFrame:
    """Return exact Router50 identities with the frozen deterministic tie rule."""
    work = router.loc[:, [*IDENTITY, "router_primary_rank"]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["__decision_ts__", "router_primary_rank", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    work["__rank__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    work["__size__"] = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    return work.loc[
        work["__rank__"].le(np.ceil(work["__size__"].to_numpy(float) * .50)),
        list(IDENTITY),
    ].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _support_path(root: Path, month: pd.Timestamp) -> Path:
    path = root / "parts" / f"month={month:%Y-%m}" / "side=long.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_policy(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    policy = pd.read_parquet(
        path,
        columns=(
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_label_available_ts",
        ),
    )
    policy["policy_label_available_ts"] = _utc(policy["policy_label_available_ts"])
    policy["policy_path_valid"] = policy["policy_path_valid"].fillna(False).astype(bool)
    policy["policy_net_bps"] = pd.to_numeric(policy["policy_net_bps"], errors="coerce")
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy ledger has duplicate candidate IDs")
    # Candidate IDs are stable, so this merely reduces the retained policy
    # sidecar after label provenance has been normalised.  It never defines the
    # score-time candidate population.
    return policy


def _fixed_ordinal(net_bps: pd.Series) -> np.ndarray:
    return np.searchsorted(
        np.asarray((0.0, 50.0, 100.0, 200.0, 400.0)),
        pd.to_numeric(net_bps, errors="coerce").to_numpy(float),
        side="right",
    ).astype(np.int8)


def _coverage_row(output: pd.DataFrame, month: pd.Timestamp) -> dict[str, object]:
    """Return the immutable coverage receipt for one completed month part.

    This is intentionally derived from the persisted label part rather than
    the source inputs.  A resumed materialisation may therefore retain an
    already-written part without recomputing, replacing, or silently changing
    any historical target values.
    """
    required = {
        *IDENTITY,
        "policy_net_bps", "decision_atr_bps", "policy_ordinal_valid",
        "raw_magnitude_valid", "normalised_magnitude_valid",
    }
    missing = sorted(required.difference(output.columns))
    if missing:
        raise AssertionError(f"{month:%Y-%m}: existing target-label part misses {missing}")
    if output["candidate_id"].duplicated().any():
        raise AssertionError(f"{month:%Y-%m}: duplicate candidate identity in persisted target labels")
    policy_valid = output["policy_ordinal_valid"].fillna(False).astype(bool)
    raw_valid = output["raw_magnitude_valid"].fillna(False).astype(bool)
    normalised_valid = output["normalised_magnitude_valid"].fillna(False).astype(bool)
    if not raw_valid.equals(policy_valid):
        raise AssertionError(f"{month:%Y-%m}: raw-magnitude and policy validity disagree")
    valid_atr = pd.to_numeric(output.loc[normalised_valid, "decision_atr_bps"], errors="coerce")
    return {
        "month": f"{month:%Y-%m}",
        "candidate_rows": int(len(output)),
        "policy_valid_rows": int(policy_valid.sum()),
        "raw_magnitude_valid_rows": int(raw_valid.sum()),
        "normalised_valid_rows": int(normalised_valid.sum()),
        "invalid_policy_rows": int((~policy_valid).sum()),
        "invalid_normalised_rows": int((~normalised_valid).sum()),
        "decision_atr_bps_p05": float(valid_atr.quantile(.05)) if len(valid_atr) else np.nan,
        "decision_atr_bps_p50": float(valid_atr.quantile(.50)) if len(valid_atr) else np.nan,
        "decision_atr_bps_p95": float(valid_atr.quantile(.95)) if len(valid_atr) else np.nan,
    }


def run(*, candidate_root: Path | None, router_root: Path | None, support_root: Path, policy_path: Path, out: Path,
        start: pd.Timestamp, end: pd.Timestamp, resume: bool = False) -> Path:
    if out.exists() and not resume:
        raise FileExistsError(f"immutable output already exists: {out}")
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(out)
    if resume and (out / "run_manifest.json").exists():
        raise AssertionError("refusing to resume a completed immutable target-label receipt")
    if start.tzinfo is None or end.tzinfo is None or start >= end or start.day != 1 or end.day != 1:
        raise ValueError("start/end must be increasing UTC month boundaries")
    if (candidate_root is None) == (router_root is None):
        raise ValueError("supply exactly one of candidate_root or router_root")
    months = _month_range(start, end)
    policy = _read_policy(policy_path, start, end)
    out.mkdir(parents=True, exist_ok=resume)
    audit: list[dict[str, object]] = []
    source_paths: list[Path] = []
    support_paths: list[Path] = []

    for month in months:
        source_path = _source_path(candidate_root, month) if candidate_root is not None else _router_path(router_root, month)
        support_path = _support_path(support_root, month)
        source_paths.append(source_path)
        support_paths.append(support_path)
        destination = out / f"month={month:%Y-%m}"
        persisted = destination / "target_labels.parquet"
        if persisted.exists():
            # This is the only resume path: existing immutable label values
            # are audited and retained verbatim.  It never writes the part.
            audit.append(_coverage_row(pd.read_parquet(persisted), month))
            continue
        if destination.exists():
            raise AssertionError(
                f"{month:%Y-%m}: partial target-label directory exists without its immutable parquet"
            )
        if candidate_root is not None:
            candidates = pd.read_parquet(source_path, columns=list(IDENTITY)).copy()
        else:
            router = pd.read_parquet(source_path, columns=[*IDENTITY, "router_primary_rank"]).copy()
            router["__decision_ts__"] = _utc(router["__decision_ts__"])
            if router["candidate_id"].duplicated().any():
                raise AssertionError(f"{month:%Y-%m}: duplicate router candidate identity")
            candidates = _router_top_half(router)
        support = pd.read_parquet(
            support_path,
            columns=(
                "candidate_id", "__decision_ts__", "side_name",
                "supportive_path_valid", "supportive_label_available_ts",
                "path_arch_atr_fraction",
            ),
        ).copy()
        candidates["__decision_ts__"] = _utc(candidates["__decision_ts__"])
        support["__decision_ts__"] = _utc(support["__decision_ts__"])
        support["supportive_label_available_ts"] = _utc(support["supportive_label_available_ts"])
        if candidates["candidate_id"].duplicated().any() or support["candidate_id"].duplicated().any():
            raise AssertionError(f"{month:%Y-%m}: duplicate candidate identity")
        if not candidates["side_name"].astype(str).str.lower().eq("long").all():
            raise AssertionError(f"{month:%Y-%m}: expected long-only candidates")
        frame = candidates.merge(
            support,
            on=list(IDENTITY),
            how="left",
            validate="one_to_one",
        ).merge(policy, on="candidate_id", how="left", validate="one_to_one")
        if len(frame) != len(candidates):
            raise AssertionError(f"{month:%Y-%m}: label join changed candidate identities")
        atr_fraction = pd.to_numeric(frame["path_arch_atr_fraction"], errors="coerce")
        frame["decision_atr_bps"] = (10_000.0 * atr_fraction).astype(np.float32)
        policy_valid = (
            frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["policy_label_available_ts"].notna()
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        )
        normalised_valid = (
            policy_valid
            & frame["supportive_path_valid"].fillna(False).astype(bool)
            & frame["supportive_label_available_ts"].notna()
            & np.isfinite(frame["decision_atr_bps"])
            & frame["decision_atr_bps"].gt(0.0)
        )
        output = frame.loc[:, list(IDENTITY)].copy()
        output["policy_net_bps"] = pd.to_numeric(frame["policy_net_bps"], errors="coerce").astype(np.float32)
        output["decision_atr_bps"] = frame["decision_atr_bps"]
        output["policy_ordinal_valid"] = policy_valid.astype(bool)
        output["policy_ordinal_grade"] = _fixed_ordinal(output["policy_net_bps"])
        output.loc[~policy_valid, "policy_ordinal_grade"] = -1
        output["raw_magnitude_valid"] = policy_valid.astype(bool)
        output["normalised_magnitude_valid"] = normalised_valid.astype(bool)
        output["policy_label_available_ts"] = frame["policy_label_available_ts"]
        output["normalised_label_available_ts"] = pd.concat(
            [frame["policy_label_available_ts"], frame["supportive_label_available_ts"]],
            axis=1,
        ).max(axis=1)
        # Raw and two volatility-normalised magnitudes remain continuous here.
        # Fold-specific clipping and monotonic relevance bins are fitted only
        # on the strictly prior training rows by the ranker screen.
        output["magnitude_raw_bps"] = output["policy_net_bps"]
        atr = output["decision_atr_bps"].to_numpy(float)
        net = output["policy_net_bps"].to_numpy(float)
        output["magnitude_sqrt_atr"] = (net / np.sqrt(atr)).astype(np.float32)
        output["magnitude_atr"] = (net / atr).astype(np.float32)
        output.loc[~policy_valid, "magnitude_raw_bps"] = np.nan
        output.loc[~normalised_valid, ["magnitude_sqrt_atr", "magnitude_atr"]] = np.nan
        destination.mkdir()
        output.to_parquet(destination / "target_labels.parquet", index=False, compression="zstd")
        audit.append(_coverage_row(output, month))
    coverage = pd.DataFrame(audit)
    coverage.to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "supervised labels only; never target-free source or inference features",
        "side": "long",
        "months": [f"{month:%Y-%m}" for month in months],
        "candidate_root": str(candidate_root) if candidate_root is not None else None,
        "router_root": str(router_root) if router_root is not None else None,
        "candidate_population": (
            "direct exact frozen Router50 identities" if router_root is not None
            else "immutable supplied target-free candidate identities"
        ),
        "candidate_sha256": _sha256(source_paths),
        "support_root": str(support_root),
        "support_sha256": _sha256(support_paths),
        "policy_path": str(policy_path),
        "policy_sha256": _sha256([policy_path]),
        "targets": {
            "T0_policy_ordinal": "<=0/0-50/50-100/100-200/200-400/>400 bps",
            "T1_raw_magnitude": "policy net bps; clipping and bins fit on each training fold only",
            "T2_sqrt_atr_magnitude": "policy net bps / sqrt(decision-time ATR bps)",
            "T3_atr_magnitude": "policy net bps / decision-time ATR bps",
        },
        "atr_contract": "Wilder-14 hourly ATR from bars completed before the decision open; persisted as path_arch_atr_fraction and converted to bps",
        "label_availability": {
            "T0": "canonical policy label availability",
            "T1_T2_T3": "max(canonical policy availability, supportive path availability)",
        },
        "invalidity": "invalid paths/ATR are excluded from supervision and diagnostics, never encoded as failures",
        "coverage": audit,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--router-root", type=Path, default=None)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2025-11-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument(
        "--resume", action="store_true",
        help="complete only a failed partial immutable materialisation after auditing every existing month part",
    )
    args = parser.parse_args()
    print(run(
        candidate_root=args.candidate_root.resolve() if args.candidate_root else None,
        router_root=args.router_root.resolve() if args.router_root else None,
        support_root=args.support_root.resolve(),
        policy_path=args.policy_path.resolve(),
        out=args.out.resolve(),
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
