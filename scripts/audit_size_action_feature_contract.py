#!/usr/bin/env python3
"""Audit size-action feature artifacts for counterfactual/non-live columns."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_FEATURES = {
    "best_multiplier",
    "best_gain",
    "best_margin",
    "best_gain_per_notional",
    "best_margin_per_notional",
    "group_affected_notional",
    "best_immediate_gain",
    "best_capacity_gain",
    "best_immediate_gain_per_notional",
    "best_capacity_gain_per_notional",
    "best_nonbaseline_gain",
    "worst_nonbaseline_gain",
    "best_nonbaseline_multiplier",
    "group_can_bind",
    "y_intervene",
    "group_best_projected_notional_removed_to_remaining_capital",
    "group_best_projected_removed_trade_share_timestamp",
    "group_best_projected_removed_trade_share_strategy",
    "group_best_projected_notional_removed_to_open_notional",
    "delta_full_J",
    "delta_immediate_J",
    "delta_full_net_pnl",
    "delta_full_cost_pnl",
    "delta_full_turnover",
    "delta_full_J_per_notional",
    "delta_immediate_J_per_notional",
    "zero_cut_target",
    "zero_cut_trainable",
    "action_positive",
    "action_economic_positive",
    "rank_relevance",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _features_from_selected_csv(path: Path) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        return []
    return sorted(set(frame["feature"].dropna().astype(str)))


def _features_from_manifest(path: Path) -> list[str]:
    data = _read_json(path)
    values: set[str] = set()
    for key in ("feature_columns", "required_columns", "column_order", "required_input_columns"):
        raw = data.get(key)
        if isinstance(raw, list):
            values.update(str(x) for x in raw)
    feature_contract = data.get("feature_contract")
    if isinstance(feature_contract, dict):
        for key in ("feature_columns", "required_columns", "column_order", "required_input_columns"):
            raw = feature_contract.get(key)
            if isinstance(raw, list):
                values.update(str(x) for x in raw)
    return sorted(values)


def _forbidden_hits(features: list[str]) -> list[str]:
    hits = set(features) & FORBIDDEN_FEATURES
    for feature in features:
        for forbidden in FORBIDDEN_FEATURES:
            if feature == f"{forbidden}_group":
                hits.add(feature)
    return sorted(hits)


def audit_feature_contract(*, run_dir: Path | None, scorer_bundle_dir: Path | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if run_dir is not None:
        selected_path = run_dir / "size_action_selected_features.csv"
        features = _features_from_selected_csv(selected_path)
        checks.append(
            {
                "source": "selected_features",
                "path": str(selected_path),
                "exists": bool(selected_path.exists()),
                "feature_count": int(len(features)),
                "forbidden_features": _forbidden_hits(features),
            }
        )
    if scorer_bundle_dir is not None:
        for filename in ("size_action_live_scorer_manifest.json", "size_action_live_feature_contract.json"):
            path = scorer_bundle_dir / filename
            features = _features_from_manifest(path)
            checks.append(
                {
                    "source": filename,
                    "path": str(path),
                    "exists": bool(path.exists()),
                    "feature_count": int(len(features)),
                    "forbidden_features": _forbidden_hits(features),
                }
            )
    blockers = [
        f"{check['source']}:{feature}"
        for check in checks
        for feature in check["forbidden_features"]
    ]
    return {
        "generated_by": "audit_size_action_feature_contract",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "scorer_bundle_dir": str(scorer_bundle_dir) if scorer_bundle_dir is not None else None,
        "live_feature_contract_clean": not blockers,
        "blockers": blockers,
        "checks": checks,
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Size-Action Feature Contract Audit",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Run dir: `{payload.get('run_dir')}`",
        f"Scorer bundle: `{payload.get('scorer_bundle_dir')}`",
        "",
        f"- Live feature contract clean: `{payload['live_feature_contract_clean']}`",
        "",
        "## Checks",
        "",
        "| source | exists | feature_count | forbidden_features |",
        "|---|---:|---:|---|",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| `{check['source']}` | `{check['exists']}` | {check['feature_count']} | "
            f"`{', '.join(check['forbidden_features']) or 'none'}` |"
        )
    lines.extend(["", "## Blockers", ""])
    if payload["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    else:
        lines.append("- none")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--scorer-bundle-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = audit_feature_contract(run_dir=args.run_dir, scorer_bundle_dir=args.scorer_bundle_dir)
    (args.out_dir / "size_action_feature_contract_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    _write_markdown(args.out_dir / "size_action_feature_contract_audit.md", payload)
    print(
        {
            "out_dir": str(args.out_dir),
            "live_feature_contract_clean": payload["live_feature_contract_clean"],
            "blockers": payload["blockers"],
        }
    )


if __name__ == "__main__":
    main()
