#!/usr/bin/env python3
"""Join frozen B/E/T target-free scores into the downstream base source.

The three inputs are independently strict-OOF, target-free monthly score
ledgers.  Their raw scales are intentionally retained as diagnostic component
coordinates, while the upstream coordinate is a predeclared weighted blend of
their deterministic timestamp-local ranks.  Canonical policy outcomes are not
read by this materialiser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_strict_r3_enhanced_base_live_stack_challenger as core


def _exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        children = sorted(path.rglob("*.parquet"))
    else:
        children = [path]
    for child in children:
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _read_score(root: Path, month: pd.Timestamp, name: str) -> pd.DataFrame:
    path = root / f"month={month:%Y-%m}" / "target_free_scores.parquet"
    frame = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "side_name", "head_score", "held_month"]).copy()
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["head_score"] = pd.to_numeric(frame["head_score"], errors="coerce")
    if frame.duplicated(["candidate_id", "__decision_ts__", "side_name"]).any() or not np.isfinite(frame.head_score).all():
        raise AssertionError(f"{name}/{month:%Y-%m}: invalid target-free score ledger")
    if not frame["held_month"].eq(f"{month:%Y-%m}").all():
        raise AssertionError(f"{name}/{month:%Y-%m}: held-month provenance mismatch")
    return frame.rename(columns={"head_score": f"{name}_score"}).drop(columns="held_month")


def _base_fields(bundle_root: Path) -> tuple[str, ...]:
    # This helper only reads the sealed historical base feature contract; the
    # new upstream does not reuse the old base booster or score panel.
    paths = core.Paths(
        raw_ledger=Path("."), direct_root=Path("."), policy_root=Path("."),
        current_mc1=Path("."), bcf_mc1=Path("."), bundle_root=bundle_root,
    )
    return core._base_fields(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-root", type=Path)
    parser.add_argument("--e-root", type=Path)
    parser.add_argument("--t-root", type=Path)
    parser.add_argument(
        "--b-only-preserve-source-geometry", action="store_true",
        help=(
            "replace only the upstream rank with the strict-OOF B score while "
            "retaining the incumbent target-free E/T raw coordinates for "
            "downstream disagreement geometry; requires --b-root and "
            "--source-target-free-root"
        ),
    )
    parser.add_argument(
        "--source-score-geometry", action="store_true",
        help=(
            "form a weighted timestamp-local B/E/T upstream entirely from the "
            "three target-free incumbent raw score coordinates already present "
            "in --source-target-free-root; used for source-B0 blend controls"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--raw-ledger", type=Path,
        help="legacy target-free raw ledger containing the sealed base fields",
    )
    parser.add_argument(
        "--copy-source-target-free-root", type=Path,
        help=(
            "copy an incumbent immutable target-free source into a versioned "
            "matched-control artifact; mutually exclusive with B/E/T scoring inputs"
        ),
    )
    parser.add_argument(
        "--copy-identity-root", type=Path,
        help=(
            "optional immutable monthly score root whose candidate identities "
            "define the matched-control router population; valid only in copy-source mode"
        ),
    )
    source.add_argument(
        "--source-target-free-root", type=Path,
        help=(
            "immutable monthly target-free panels containing the sealed base "
            "fields; only identity and frozen feature columns are read"
        ),
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--start", default="2025-11-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--b-weight", type=float, default=.40)
    parser.add_argument("--e-weight", type=float, default=.55)
    parser.add_argument("--t-weight", type=float, default=.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    scoring_roots = (args.b_root, args.e_root, args.t_root)
    if args.copy_source_target_free_root is not None:
        if any(root is not None for root in scoring_roots):
            raise ValueError("copy-source mode cannot receive B/E/T score roots")
        if args.raw_ledger is not None or args.source_target_free_root is not None:
            raise ValueError("copy-source mode cannot receive an independent feature source")
    elif args.copy_identity_root is not None:
        raise ValueError("copy-identity-root is valid only in copy-source mode")
    elif args.b_only_preserve_source_geometry and args.source_score_geometry:
        raise ValueError("B-only source-geometry mode and source-score-geometry mode are mutually exclusive")
    elif args.b_only_preserve_source_geometry:
        if args.b_root is None or args.e_root is not None or args.t_root is not None:
            raise ValueError("B-only source-geometry mode requires --b-root only")
        if args.source_target_free_root is None or args.raw_ledger is not None:
            raise ValueError("B-only source-geometry mode requires --source-target-free-root only")
    elif args.source_score_geometry:
        if any(root is not None for root in scoring_roots):
            raise ValueError("source-score-geometry mode cannot receive B/E/T score roots")
        if args.source_target_free_root is None or args.raw_ledger is not None:
            raise ValueError("source-score-geometry mode requires --source-target-free-root only")
    elif any(root is None for root in scoring_roots):
        raise ValueError("B/E/T score roots are required unless copy-source mode is selected")
    elif args.raw_ledger is None and args.source_target_free_root is None:
        raise ValueError("an independent raw ledger or target-free feature source is required")
    weights = np.array([args.b_weight, args.e_weight, args.t_weight], dtype=float)
    if not args.b_only_preserve_source_geometry and ((weights < 0.0).any() or not np.isclose(weights.sum(), 1.0)):
        raise ValueError("B/E/T weights must be non-negative and sum to 1")
    start, end = core._utc(args.start), core._utc(args.end)
    months = tuple(pd.date_range(start, end, freq="MS", tz="UTC"))
    fields = _base_fields(args.bundle_root)
    args.out.mkdir(parents=True)
    source_path = args.copy_source_target_free_root or args.raw_ledger or args.source_target_free_root
    source_kind = (
        "matched_control_target_free_monthly" if args.copy_source_target_free_root is not None
        else "raw_ledger" if args.raw_ledger is not None else "target_free_monthly"
    )
    upstream = (
        {
            "coordinate": (
                "incumbent immutable target-free upstream on frozen router-50 intersection"
                if args.copy_identity_root is not None
                else "incumbent immutable target-free upstream"
            ),
            "mode": "matched_control_copy",
            "identity_source": str(args.copy_identity_root) if args.copy_identity_root is not None else None,
        }
        if args.copy_source_target_free_root is not None
        else (
            {
                "coordinate": "B0 F72 strict-OOF timestamp-local rank with incumbent E/T geometry retained",
                "mode": "b_only_preserve_incumbent_et_geometry",
                "B_weight": 1.0, "E_weight": 0.0, "T_weight": 0.0,
            }
            if args.b_only_preserve_source_geometry
            else (
                {
                    "coordinate": "incumbent source B/E/T timestamp-local rank",
                    "mode": "source_score_geometry",
                    "B_weight": args.b_weight, "E_weight": args.e_weight, "T_weight": args.t_weight,
                }
                if args.source_score_geometry
                else {"coordinate": "weighted timestamp-local B/E/T rank", "B_weight": args.b_weight, "E_weight": args.e_weight, "T_weight": args.t_weight}
            )
        )
    )
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_frozen_threeway_targetfree_v1",
        "scope": "offline research-only target-free upstream materialisation; no policy, consensus, MC1, admission, portfolio, execution, live, or exchange mutation",
        "inputs": {
            **(
                ({"B": str(args.b_root)} if args.b_only_preserve_source_geometry else ({} if args.source_score_geometry else {"B": str(args.b_root), "E": str(args.e_root), "T": str(args.t_root)}))
                if args.copy_source_target_free_root is None else {}
            ),
            source_kind: str(source_path), "bundle_root": str(args.bundle_root),
            **({"matched_control_identity_root": str(args.copy_identity_root)} if args.copy_identity_root is not None else {}),
        },
        "input_sha256": {
            **(
                ({"B": _sha(args.b_root)} if args.b_only_preserve_source_geometry else ({} if args.source_score_geometry else {"B": _sha(args.b_root), "E": _sha(args.e_root), "T": _sha(args.t_root)}))
                if args.copy_source_target_free_root is None else {}
            ),
            source_kind: _sha(source_path),
            **({"matched_control_identity_root": _sha(args.copy_identity_root)} if args.copy_identity_root is not None else {}),
        },
        "months": [f"{month:%Y-%m}" for month in months],
        "head_score_provenance": (
            "incumbent immutable target-free score source copied without score or feature mutation"
            if args.copy_source_target_free_root is not None
            else (
                "strict-OOF B0 F72 scorer with incumbent target-free E/T raw geometry retained; complete point-in-time top-50 router population"
                if args.b_only_preserve_source_geometry
                else (
                    "target-free incumbent B/E/T score coordinates re-ranked only within each decision timestamp; complete point-in-time top-50 router population"
                    if args.source_score_geometry
                    else "independent strict-OOF frozen-head scorers; complete point-in-time top-50 router population"
                )
            )
        ),
        "upstream": upstream,
        "raw_component_coordinates": "unscaled model scores retained for disagreement features; downstream heads are retrained under this exact source",
        "base_feature_contract": {"count": len(fields), "sha256": hashlib.sha256("\n".join(fields).encode()).hexdigest()},
        "target_fields_in_output": False,
    })
    audit: list[dict[str, object]] = []
    keys = ["candidate_id", "__decision_ts__", "side_name"]
    for month in months:
        if args.copy_source_target_free_root is not None:
            source_panel = args.copy_source_target_free_root / f"month={month:%Y-%m}" / "scores_features.parquet"
            if not source_panel.exists():
                raise FileNotFoundError(f"{month:%Y-%m}: missing matched-control source panel {source_panel}")
            output = pd.read_parquet(source_panel)
            prohibited = {
                "policy_path_valid", "policy_net_bps", "policy_gross_bps",
                "policy_label_available_ts", "policy_exit_bar_15m", "policy_entry_price",
                "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
            }
            leaked = sorted(prohibited.intersection(output.columns))
            if leaked:
                raise AssertionError(f"{month:%Y-%m}: matched-control source contains policy fields: {leaked}")
            missing = sorted(set([*keys, *fields]) - set(output.columns))
            if missing or output.columns.duplicated().any():
                raise AssertionError(f"{month:%Y-%m}: matched-control source schema invalid: {missing[:3]}")
            output["__decision_ts__"] = pd.to_datetime(output["__decision_ts__"], utc=True, errors="raise")
            if output.duplicated(keys).any() or not output.side_name.astype(str).str.lower().eq("long").all():
                raise AssertionError(f"{month:%Y-%m}: matched-control source identity/side invalid")
            if args.copy_identity_root is not None:
                identity_path = args.copy_identity_root / f"month={month:%Y-%m}" / "target_free_scores.parquet"
                if not identity_path.exists():
                    raise FileNotFoundError(f"{month:%Y-%m}: missing matched-control identity panel {identity_path}")
                identity = pd.read_parquet(identity_path, columns=keys)
                identity["__decision_ts__"] = pd.to_datetime(identity["__decision_ts__"], utc=True, errors="raise")
                if identity.duplicated(keys).any():
                    raise AssertionError(f"{month:%Y-%m}: matched-control identity source duplicates")
                output = identity.merge(output, on=keys, how="inner", validate="one_to_one")
                if len(output) != len(identity):
                    raise AssertionError(f"{month:%Y-%m}: matched-control identity join changed router population")
                output["base_rank_ts"] = core._rank_pct(output, "enhanced_base_bps").to_numpy(np.float32)
                output["enhanced_base_routed"] = core._exact_timestamp_top_fraction(
                    output, "enhanced_base_bps", core.BASE_ROUTE,
                ).to_numpy(bool)
            target = args.out / f"month={month:%Y-%m}"
            target.mkdir()
            output.to_parquet(target / "scores_features.parquet", index=False, compression="zstd")
            audit.append({
                "month": f"{month:%Y-%m}", "rows": int(len(output)), "timestamps": int(output.__decision_ts__.nunique()),
                "feature_complete_fraction": float(output.loc[:, list(fields)].notna().all(axis=1).mean()),
                "router50_rows": int(len(output)), "base_route30_rows": int(output.enhanced_base_routed.sum()),
                "target_free": True,
            })
            print(json.dumps({"event": "month_complete", **audit[-1]}), flush=True)
            continue
        if args.source_score_geometry:
            source_panel = args.source_target_free_root / f"month={month:%Y-%m}" / "scores_features.parquet"
            if not source_panel.exists():
                raise FileNotFoundError(f"{month:%Y-%m}: missing target-free source panel {source_panel}")
            probe = pd.read_parquet(source_panel)
            prohibited = {
                "policy_path_valid", "policy_net_bps", "policy_gross_bps",
                "policy_label_available_ts", "policy_exit_bar_15m", "policy_entry_price",
                "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
            }
            leaked = sorted(prohibited.intersection(probe.columns))
            if leaked:
                raise AssertionError(f"{month:%Y-%m}: target-free source contains policy fields: {leaked}")
            required = set([*keys, "base_bps", "efficiency_bps", "timing_bps", *fields])
            missing = sorted(required - set(probe.columns))
            if missing:
                raise AssertionError(f"{month:%Y-%m}: target-free source lacks B/E/T coordinates or sealed fields: {missing[:3]}")
            frame = probe.loc[:, [*keys, "base_bps", "efficiency_bps", "timing_bps", *fields]].copy().rename(columns={
                "base_bps": "B_score", "efficiency_bps": "E_score", "timing_bps": "T_score",
            })
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
            if frame.duplicated(keys).any():
                raise AssertionError(f"{month:%Y-%m}: source B/E/T score duplicate identity")
        else:
            b = _read_score(args.b_root, month, "B")
        if args.b_only_preserve_source_geometry:
            source_panel = args.source_target_free_root / f"month={month:%Y-%m}" / "scores_features.parquet"
            if not source_panel.exists():
                raise FileNotFoundError(f"{month:%Y-%m}: missing target-free source panel {source_panel}")
            probe = pd.read_parquet(source_panel)
            prohibited = {
                "policy_path_valid", "policy_net_bps", "policy_gross_bps",
                "policy_label_available_ts", "policy_exit_bar_15m", "policy_entry_price",
                "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
            }
            leaked = sorted(prohibited.intersection(probe.columns))
            if leaked:
                raise AssertionError(f"{month:%Y-%m}: target-free source contains policy fields: {leaked}")
            required = set([*keys, "efficiency_bps", "timing_bps", *fields])
            missing = sorted(required - set(probe.columns))
            if missing:
                raise AssertionError(f"{month:%Y-%m}: target-free source lacks retained E/T geometry or sealed fields: {missing[:3]}")
            raw = probe.loc[:, [*keys, "efficiency_bps", "timing_bps", *fields]].copy()
            raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
            if raw.duplicated(keys).any():
                raise AssertionError(f"{month:%Y-%m}: retained E/T source duplicate identity")
            frame = b.merge(raw, on=keys, how="inner", validate="one_to_one")
            if len(frame) != len(b):
                raise AssertionError(f"{month:%Y-%m}: retained E/T source join changed B0 score identities")
        elif not args.source_score_geometry:
            e = _read_score(args.e_root, month, "E")
            t = _read_score(args.t_root, month, "T")
            frame = b.merge(e, on=keys, how="inner", validate="one_to_one").merge(t, on=keys, how="inner", validate="one_to_one")
            if len(frame) != len(b) or len(frame) != len(e) or len(frame) != len(t):
                raise AssertionError(f"{month:%Y-%m}: B/E/T target-free identity mismatch")
        if args.raw_ledger is not None:
            raw = pd.read_parquet(
                args.raw_ledger, columns=[*keys, *fields],
                filters=[("__decision_ts__", ">=", month), ("__decision_ts__", "<", month + pd.offsets.MonthBegin(1))],
            )
        elif not args.b_only_preserve_source_geometry:
            source_panel = args.source_target_free_root / f"month={month:%Y-%m}" / "scores_features.parquet"
            if not source_panel.exists():
                raise FileNotFoundError(f"{month:%Y-%m}: missing target-free source panel {source_panel}")
            probe = pd.read_parquet(source_panel)
            prohibited = {
                "policy_path_valid", "policy_net_bps", "policy_gross_bps",
                "policy_label_available_ts", "policy_exit_bar_15m", "policy_entry_price",
                "policy_exit_price", "policy_exit_reason", "policy_cost_bps",
            }
            leaked = sorted(prohibited.intersection(probe.columns))
            if leaked:
                raise AssertionError(f"{month:%Y-%m}: target-free source contains policy fields: {leaked}")
            missing = sorted(set([*keys, *fields]) - set(probe.columns))
            if missing:
                raise AssertionError(f"{month:%Y-%m}: target-free source lacks sealed fields: {missing[:3]}")
            raw = probe.loc[:, [*keys, *fields]].copy()
        raw["__decision_ts__"] = pd.to_datetime(raw["__decision_ts__"], utc=True, errors="raise")
        if raw.duplicated(keys).any():
            raise AssertionError(f"{month:%Y-%m}: raw ledger duplicate identity")
        if not args.b_only_preserve_source_geometry and not args.source_score_geometry:
            frame = frame.merge(raw, on=keys, how="inner", validate="one_to_one")
            if len(frame) != len(b):
                raise AssertionError(f"{month:%Y-%m}: raw target-free join changed score identities")
        if not frame.side_name.astype(str).str.lower().eq("long").all():
            raise AssertionError(f"{month:%Y-%m}: source is not long-only")
        frame["B_rank_ts"] = core._rank_desc(frame, "B_score")
        if not args.b_only_preserve_source_geometry:
            for name in ("E", "T"):
                frame[f"{name}_rank_ts"] = core._rank_desc(frame, f"{name}_score")
        frame["base_bps"] = frame["B_score"].to_numpy(np.float32)
        if not args.b_only_preserve_source_geometry:
            frame["efficiency_bps"] = frame["E_score"].to_numpy(np.float32)
            frame["timing_bps"] = frame["T_score"].to_numpy(np.float32)
            frame["enhanced_base_bps"] = (
                args.b_weight * frame["B_rank_ts"].to_numpy(float)
                + args.e_weight * frame["E_rank_ts"].to_numpy(float)
                + args.t_weight * frame["T_rank_ts"].to_numpy(float)
            ).astype(np.float32)
        else:
            frame["enhanced_base_bps"] = frame["B_rank_ts"].to_numpy(np.float32)
        frame["base_rank_ts"] = core._rank_pct(frame, "enhanced_base_bps").to_numpy(np.float32)
        frame["enhanced_base_routed"] = core._exact_timestamp_top_fraction(frame, "enhanced_base_bps", core.BASE_ROUTE).to_numpy(bool)
        frame["e_minus_t"] = frame["efficiency_bps"] - frame["timing_bps"]
        frame["e_minus_b0"] = frame["efficiency_bps"] - frame["base_bps"]
        frame["t_minus_b0"] = frame["timing_bps"] - frame["base_bps"]
        frame["base_component_std"] = np.nanstd(frame.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].to_numpy(float), axis=1).astype(np.float32)
        output_columns = [
            *keys, "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed",
            "base_bps", "efficiency_bps", "timing_bps", "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std", *fields,
        ]
        output = frame.loc[:, output_columns]
        if output.columns.duplicated().any() or output.duplicated(keys).any():
            raise AssertionError(f"{month:%Y-%m}: output identity/schema invalid")
        target = args.out / f"month={month:%Y-%m}"
        target.mkdir()
        output.to_parquet(target / "scores_features.parquet", index=False, compression="zstd")
        audit.append({
            "month": f"{month:%Y-%m}", "rows": int(len(output)), "timestamps": int(output.__decision_ts__.nunique()),
            "feature_complete_fraction": float(output.loc[:, list(fields)].notna().all(axis=1).mean()),
            "router50_rows": int(len(output)), "base_route30_rows": int(output.enhanced_base_routed.sum()),
            "target_free": True,
        })
        print(json.dumps({"event": "month_complete", **audit[-1]}), flush=True)
    pd.DataFrame(audit).to_parquet(args.out / "coverage_audit.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
