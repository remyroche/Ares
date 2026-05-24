#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


VARIANTS = {
    "rollout_on_new": {
        "run_id": "20260522_101001",
        "rollout": True,
        "selector": "new",
    },
    "rollout_off_new": {
        "run_id": "20260522_101002",
        "rollout": False,
        "selector": "new",
    },
    "rollout_on_old_selector": {
        "run_id": "20260522_101003",
        "rollout": True,
        "selector": "old",
    },
    "rollout_off_old_selector": {
        "run_id": "20260522_101004",
        "rollout": False,
        "selector": "old",
    },
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _first_strategy_payload(policy: dict) -> tuple[str | None, dict]:
    for key, value in policy.items():
        if str(key).startswith("__"):
            continue
        if isinstance(value, dict):
            return str(key), value
    return None, {}


def _feature_count(artifacts: Path, strategy_id: str | None) -> int | None:
    if not strategy_id:
        return None
    native = artifacts / "models" / "native"
    if not native.exists():
        return None
    matches = sorted(native.glob(f"{strategy_id}_H*/columns.json"))
    if not matches and strategy_id.startswith("short_"):
        matches = sorted(native.glob(f"{strategy_id[len('short_'):]}_H*/columns.json"))
    if not matches:
        return None
    cols = _read_json(matches[0])
    if isinstance(cols, list):
        return len(cols)
    if isinstance(cols, dict):
        for key in ("columns", "feature_names", "selected_features"):
            if isinstance(cols.get(key), list):
                return len(cols[key])
    return None


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data_perp_policy_rollout_feature_selector_experiment"
    )
    rows = []
    for variant, meta in VARIANTS.items():
        run_id = meta["run_id"]
        artifacts = root / f"{variant}_perp" / "artifacts" / run_id
        policy_path = artifacts / "policy_optimisation.json"
        if not policy_path.exists():
            continue
        policy = _read_json(policy_path)
        strategy_id, payload = _first_strategy_payload(policy)
        final = payload.get("final_policy_deployment_metrics") or {}
        threshold = payload.get("deployment_threshold_metrics") or {}
        rows.append(
            {
                "variant": variant,
                "run_id": run_id,
                "policy_rollout": meta["rollout"],
                "selector": meta["selector"],
                "strategy_id": strategy_id,
                "deploy_rank_threshold": threshold.get("deployment_rank_threshold"),
                "threshold_candidate_rows": threshold.get("candidate_rows"),
                "selected_trades": final.get("n_trades"),
                "hit_rate": final.get("hit_rate"),
                "mean_gross_trade": final.get("mean_gross_trade"),
                "mean_net_trade": final.get("mean_net_trade"),
                "net_pnl": final.get("net_pnl"),
                "max_drawdown": final.get("max_drawdown"),
                "sortino": final.get("sortino"),
                "avg_holding_hours": final.get("avg_holding_time_hours"),
                "selected_feature_count": _feature_count(artifacts, strategy_id),
                "policy_json_exists": bool(policy),
            }
        )
    df = pd.DataFrame(rows)
    out_dir = root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "policy_rollout_selector_comparison.csv", index=False)
    (out_dir / "policy_rollout_selector_comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
