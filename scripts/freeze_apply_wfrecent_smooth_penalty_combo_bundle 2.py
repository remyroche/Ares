#!/usr/bin/env python3
"""Freeze and apply bounded wf_recent smooth-penalty combo bundles."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_wfrecent_smooth_rank_penalty import SmoothRule, _fit_threshold, _penalty_values  # noqa: E402
from scripts.freeze_apply_wfrecent_smooth_penalty_bundle import (  # noqa: E402
    _coverage_table,
    _refs_from_npz,
    _refs_to_npz,
    _sha256_file,
    _sha256_json,
)
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    _apply_risk_scores,
    _fit_percentile_reference,
    _fmt_table,
    _head_name,
    _json_safe,
)
from scripts.validate_wfrecent_smooth_penalty_combo_holdout import Combo, ComboLeg, RULE_LIBRARY  # noqa: E402


DEFAULT_COMBOS: dict[str, Combo] = {
    "q85_only": Combo("q85_only", (ComboLeg("q85_aggressive", 1.0),), 0.05),
    "q85_plus_drift_quarter": Combo(
        "q85_plus_drift_quarter",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("drift_all_q90_pen0p025_pow1p0", 0.25)),
        0.06,
    ),
    "q85_plus_drift_long_dist_quarter": Combo(
        "q85_plus_drift_long_dist_quarter",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("drift_long_dist_q90_pen0p025_pow1p0", 0.25)),
        0.06,
    ),
    "q85_plus_recent_hr_quarter": Combo(
        "q85_plus_recent_hr_quarter",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 0.25)),
        0.06,
    ),
    "q85_plus_recent_hr_half": Combo(
        "q85_plus_recent_hr_half",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 0.5)),
        0.07,
    ),
    "q85_plus_recent_hr_three_quarter": Combo(
        "q85_plus_recent_hr_three_quarter",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 0.75)),
        0.08,
    ),
    "q85_plus_recent_hr_pen05_quarter": Combo(
        "q85_plus_recent_hr_pen05_quarter",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_all_q85_pen0p05_pow2p0", 0.25)),
        0.07,
    ),
    "q85_plus_recent_hr_pen05_half": Combo(
        "q85_plus_recent_hr_pen05_half",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_all_q85_pen0p05_pow2p0", 0.5)),
        0.08,
    ),
    "recent_hr_only": Combo(
        "recent_hr_only",
        (ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 1.0),),
        0.07,
    ),
    "recent_hr_only_pen05": Combo(
        "recent_hr_only_pen05",
        (ComboLeg("recent_hr_all_q85_pen0p05_pow2p0", 1.0),),
        0.08,
    ),
    "q85_plus_recent_hr_long_bars_half": Combo(
        "q85_plus_recent_hr_long_bars_half",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_long_bars_q85_pen0p025_pow2p0", 0.5)),
        0.07,
    ),
    "q85_plus_recent_hr_long_dist_half": Combo(
        "q85_plus_recent_hr_long_dist_half",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_long_dist_q85_pen0p025_pow2p0", 0.5)),
        0.07,
    ),
    "q85_plus_recent_hr_short_asset_half": Combo(
        "q85_plus_recent_hr_short_asset_half",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_short_asset_q85_pen0p025_pow2p0", 0.5)),
        0.07,
    ),
    "q85_plus_recent_hr_short_bollinger_half": Combo(
        "q85_plus_recent_hr_short_bollinger_half",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("recent_hr_short_bollinger_q85_pen0p025_pow2p0", 0.5)),
        0.07,
    ),
    "q85_plus_recent_hr_short_heads_half": Combo(
        "q85_plus_recent_hr_short_heads_half",
        (
            ComboLeg("q85_aggressive", 1.0),
            ComboLeg("recent_hr_short_asset_q85_pen0p025_pow2p0", 0.5),
            ComboLeg("recent_hr_short_bollinger_q85_pen0p025_pow2p0", 0.5),
        ),
        0.07,
    ),
    "q85_plus_recent_hr_no_short_asset_half": Combo(
        "q85_plus_recent_hr_no_short_asset_half",
        (
            ComboLeg("q85_aggressive", 1.0),
            ComboLeg("recent_hr_long_bars_q85_pen0p025_pow2p0", 0.5),
            ComboLeg("recent_hr_long_dist_q85_pen0p025_pow2p0", 0.5),
            ComboLeg("recent_hr_short_bollinger_q85_pen0p025_pow2p0", 0.5),
        ),
        0.07,
    ),
    "uncertainty_recent_balanced": Combo(
        "uncertainty_recent_balanced",
        (
            ComboLeg("uncertainty_all_q90_pen0p025_pow2p0", 0.75),
            ComboLeg("recent_hr_all_q85_pen0p025_pow2p0", 0.35),
        ),
        0.06,
    ),
    "q85_plus_ood_quarter": Combo(
        "q85_plus_ood_quarter",
        (ComboLeg("q85_aggressive", 1.0), ComboLeg("ood_all_q90_pen0p025_pow1p0", 0.25)),
        0.06,
    ),
    "q85_plus_drift_ood_quarter": Combo(
        "q85_plus_drift_ood_quarter",
        (
            ComboLeg("q85_aggressive", 1.0),
            ComboLeg("drift_all_q90_pen0p025_pow1p0", 0.25),
            ComboLeg("ood_all_q90_pen0p025_pow1p0", 0.25),
        ),
        0.07,
    ),
    "q85_plus_ood_short_heads_quarter": Combo(
        "q85_plus_ood_short_heads_quarter",
        (
            ComboLeg("q85_aggressive", 1.0),
            ComboLeg("ood_short_asset_q90_pen0p025_pow1p0", 0.25),
            ComboLeg("ood_short_bollinger_q90_pen0p025_pow1p0", 0.25),
        ),
        0.07,
    ),
}


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    frame["head"] = frame["strategy_id"].map(_head_name)
    if "portfolio_rank_adjustment" not in frame.columns:
        frame["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        frame["portfolio_rank_adjustment"] = (
            pd.to_numeric(frame["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).astype("float32")
        )
    return frame


def _required_rule_names(combos: dict[str, Combo]) -> list[str]:
    names: list[str] = []
    for combo in combos.values():
        for leg in combo.legs:
            if leg.rule_name not in names:
                names.append(leg.rule_name)
    return names


def _combo_payload(combo: Combo) -> dict[str, Any]:
    return {
        "label": combo.label,
        "total_cap": float(combo.total_cap),
        "legs": [{"rule_name": leg.rule_name, "weight": float(leg.weight)} for leg in combo.legs],
    }


def _combo_from_payload(payload: dict[str, Any]) -> Combo:
    return Combo(
        str(payload["label"]),
        tuple(ComboLeg(str(leg["rule_name"]), float(leg["weight"])) for leg in payload["legs"]),
        float(payload["total_cap"]),
    )


def _select_combos(names: str) -> dict[str, Combo]:
    if not names.strip():
        return dict(DEFAULT_COMBOS)
    selected: dict[str, Combo] = {}
    if "=" in names:
        for raw_combo in [item.strip() for item in names.split(";") if item.strip()]:
            label, legs_part = raw_combo.split("=", 1)
            label = label.strip()
            total_cap = 0.07
            legs_text = legs_part
            if "|cap=" in legs_part:
                legs_text, cap_text = legs_part.rsplit("|cap=", 1)
                total_cap = float(cap_text)
            legs: list[ComboLeg] = []
            for raw_leg in [item.strip() for item in legs_text.split("+") if item.strip()]:
                if "*" in raw_leg:
                    weight_text, rule_name = raw_leg.split("*", 1)
                    weight = float(weight_text)
                    rule_name = rule_name.strip()
                else:
                    weight = 1.0
                    rule_name = raw_leg
                if rule_name not in RULE_LIBRARY:
                    raise ValueError(f"Unknown rule {rule_name} in combo {label}. Available: {sorted(RULE_LIBRARY)}")
                legs.append(ComboLeg(rule_name, weight))
            if not legs:
                raise ValueError(f"Combo {label} has no legs")
            selected[label] = Combo(label, tuple(legs), float(total_cap))
        return selected
    for name in [item.strip() for item in names.split(",") if item.strip()]:
        if name not in DEFAULT_COMBOS:
            raise ValueError(f"Unknown combo {name}. Available: {sorted(DEFAULT_COMBOS)}")
        selected[name] = DEFAULT_COMBOS[name]
    return selected


def _apply_combo_with_thresholds(
    scored: pd.DataFrame,
    combo: Combo,
    thresholds: dict[str, float],
) -> tuple[np.ndarray, pd.DataFrame]:
    total = np.zeros(len(scored), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for leg in combo.legs:
        rule = RULE_LIBRARY[leg.rule_name]
        threshold = float(thresholds[leg.rule_name])
        penalty = _penalty_values(scored, rule, threshold).astype(np.float32) * float(leg.weight)
        total += penalty
        mask = penalty < 0.0
        rows.append(
            {
                "combo": combo.label,
                "rule_name": leg.rule_name,
                "weight": float(leg.weight),
                "threshold": threshold,
                "penalized_rows": int(np.sum(mask)),
                "penalized_share": float(np.mean(mask)) if len(mask) else 0.0,
                "mean_penalty": float(np.mean(penalty[mask])) if np.any(mask) else 0.0,
                "min_penalty": float(np.min(penalty[mask])) if np.any(mask) else 0.0,
            }
        )
    capped = np.clip(total, -float(combo.total_cap), 0.0).astype(np.float32)
    mask = capped < 0.0
    rows.append(
        {
            "combo": combo.label,
            "rule_name": "__combined__",
            "weight": 1.0,
            "threshold": np.nan,
            "penalized_rows": int(np.sum(mask)),
            "penalized_share": float(np.mean(mask)) if len(mask) else 0.0,
            "mean_penalty": float(np.mean(capped[mask])) if np.any(mask) else 0.0,
            "min_penalty": float(np.min(capped[mask])) if np.any(mask) else 0.0,
            "total_cap": float(combo.total_cap),
        }
    )
    return capped, pd.DataFrame(rows)


def _freeze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combos = _select_combos(str(args.combos))
    rule_names = _required_rule_names(combos)
    candidates = _load_candidates(args.candidates)
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    train = candidates[candidates["timestamp"].lt(cutoff)].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError(f"No rows before cutoff {cutoff.isoformat()}")

    coverage = _coverage_table(train)
    refs = _fit_percentile_reference(train)
    scored = _apply_risk_scores(train, refs)
    refs_path = args.output_dir / "risk_percentile_refs.npz"
    ref_mapping = _refs_to_npz(refs, refs_path)

    rules_payload: dict[str, dict[str, Any]] = {}
    thresholds: dict[str, float] = {}
    for name in rule_names:
        rule = RULE_LIBRARY[name]
        threshold = _fit_threshold(scored, rule)
        if not np.isfinite(float(threshold)):
            raise ValueError(f"Non-finite threshold for {name}: {threshold}")
        rules_payload[name] = rule.__dict__
        thresholds[name] = float(threshold)

    manifest = {
        "generated_by": "freeze_apply_wfrecent_smooth_penalty_combo_bundle.freeze",
        "candidate_source": str(args.candidates),
        "candidate_source_sha256": _sha256_file(args.candidates),
        "cutoff": cutoff.isoformat(),
        "train_rows": int(len(train)),
        "train_start": train["timestamp"].min().isoformat(),
        "train_end": train["timestamp"].max().isoformat(),
        "combos": {name: _combo_payload(combo) for name, combo in combos.items()},
        "rules": rules_payload,
        "thresholds": thresholds,
        "reference_mapping": ref_mapping,
        "reference_npz": refs_path.name,
    }
    manifest["bundle_hash"] = _sha256_json({k: v for k, v in manifest.items() if k != "bundle_hash"})
    (args.output_dir / "smooth_penalty_combo_bundle_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    coverage.to_csv(args.output_dir / "smooth_penalty_combo_bundle_feature_coverage.csv", index=False)

    rule_rows = []
    for combo_name, combo in combos.items():
        for leg in combo.legs:
            rule = RULE_LIBRARY[leg.rule_name]
            rule_rows.append(
                {
                    "combo": combo_name,
                    "rule_name": leg.rule_name,
                    "weight": float(leg.weight),
                    "total_cap": float(combo.total_cap),
                    **rule.__dict__,
                    "threshold": thresholds[leg.rule_name],
                }
            )
    rule_table = pd.DataFrame(rule_rows)
    rule_table.to_csv(args.output_dir / "smooth_penalty_combo_bundle_rules.csv", index=False)

    lines = [
        "# wf_recent Smooth Penalty Combo Frozen Bundle",
        "",
        f"Cutoff: `{cutoff.isoformat()}`",
        f"Training rows: `{len(train)}`",
        f"Bundle hash: `{manifest['bundle_hash']}`",
        "",
        "## Combo Rules",
        "",
        rule_table.to_markdown(index=False),
        "",
        "## Feature Coverage",
        "",
        _fmt_table(coverage, ["column", "present", "finite_rate", "missing_count"]),
    ]
    (args.output_dir / "smooth_penalty_combo_bundle_report.md").write_text("\n".join(lines) + "\n")


def _apply(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.bundle_dir / "smooth_penalty_combo_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    refs = _refs_from_npz(args.bundle_dir / str(manifest["reference_npz"]), dict(manifest["reference_mapping"]))
    candidates = _load_candidates(args.candidates)
    scored = _apply_risk_scores(candidates, refs)
    base_adjustment = pd.to_numeric(scored["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    combo_audits: list[pd.DataFrame] = []
    output_rows: list[dict[str, Any]] = []
    for combo_name, payload in manifest["combos"].items():
        combo = _combo_from_payload(payload)
        penalty, audit = _apply_combo_with_thresholds(scored, combo, dict(manifest["thresholds"]))
        combo_audits.append(audit)
        out = scored.copy()
        out["smooth_penalty_variant"] = combo_name
        out["smooth_penalty_value"] = penalty.astype("float32")
        out["smooth_penalty_bundle_hash"] = str(manifest["bundle_hash"])
        out["smooth_penalty_components"] = json.dumps(payload["legs"], sort_keys=True)
        out["portfolio_rank_adjustment"] = np.clip(base_adjustment + penalty, -1.0, 1.0).astype("float32")
        out_path = args.output_dir / f"{combo_name}_smooth_penalty_combo_candidates.parquet"
        out.drop(columns=["head"], errors="ignore").to_parquet(out_path, index=False)
        combined = audit[audit["rule_name"].eq("__combined__")].iloc[0]
        output_rows.append(
            {
                "combo": combo_name,
                "output": str(out_path),
                "output_sha256": _sha256_file(out_path),
                "candidate_rows": int(len(out)),
                "penalized_rows": int(combined["penalized_rows"]),
                "penalized_share": float(combined["penalized_share"]),
                "mean_penalty": float(combined["mean_penalty"]),
                "min_penalty": float(combined["min_penalty"]),
                "total_cap": float(combo.total_cap),
            }
        )

    audit_out = pd.concat(combo_audits, ignore_index=True) if combo_audits else pd.DataFrame()
    outputs = pd.DataFrame(output_rows)
    audit_out.to_csv(args.output_dir / "smooth_penalty_combo_apply_component_audit.csv", index=False)
    outputs.to_csv(args.output_dir / "smooth_penalty_combo_apply_audit.csv", index=False)
    apply_manifest = {
        "generated_by": "freeze_apply_wfrecent_smooth_penalty_combo_bundle.apply",
        "bundle_dir": str(args.bundle_dir),
        "bundle_hash": manifest["bundle_hash"],
        "candidate_source": str(args.candidates),
        "candidate_source_sha256": _sha256_file(args.candidates),
        "candidate_rows": int(len(candidates)),
        "candidate_start": candidates["timestamp"].min().isoformat(),
        "candidate_end": candidates["timestamp"].max().isoformat(),
        "outputs": output_rows,
    }
    (args.output_dir / "smooth_penalty_combo_apply_manifest.json").write_text(
        json.dumps(_json_safe(apply_manifest), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# wf_recent Smooth Penalty Combo Bundle Apply",
        "",
        f"Bundle: `{args.bundle_dir}`",
        f"Bundle hash: `{manifest['bundle_hash']}`",
        f"Candidate rows: `{len(candidates)}`",
        "",
        "## Output Audit",
        "",
        _fmt_table(
            outputs,
            [
                "combo",
                "candidate_rows",
                "penalized_rows",
                "penalized_share",
                "mean_penalty",
                "min_penalty",
                "total_cap",
                "output_sha256",
            ],
        ),
        "",
        "## Component Audit",
        "",
        _fmt_table(
            audit_out,
            ["combo", "rule_name", "weight", "penalized_rows", "penalized_share", "mean_penalty", "min_penalty"],
        ),
    ]
    (args.output_dir / "smooth_penalty_combo_apply_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument(
        "--candidates",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701/combo_candidates.parquet"),
    )
    freeze.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_freeze_20260701"),
    )
    freeze.add_argument("--cutoff", default="2026-06-27T00:00:00+00:00")
    freeze.add_argument("--combos", default="")
    apply = sub.add_parser("apply")
    apply.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_freeze_20260701"),
    )
    apply.add_argument(
        "--candidates",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701/combo_candidates.parquet"),
    )
    apply.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_freeze_apply_smoke_20260701"),
    )
    args = parser.parse_args()
    if args.cmd == "freeze":
        _freeze(args)
    elif args.cmd == "apply":
        _apply(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
