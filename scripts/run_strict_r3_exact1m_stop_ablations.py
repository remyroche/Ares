#!/usr/bin/env python3
"""Isolated exact-1m stop-policy ablations on a frozen score/admission route.

Families are deliberately evaluated separately:

* ``stop_geometry`` changes only the current ATR stop transform;
* ``time_stop`` adds a causal, post-completed-bar low-MFE tightening stop to
  the frozen continuous smooth protection;
* ``stepped_mfe`` replaces continuous smooth protection with a small,
  monotone MFE-lock schedule.

No score, BCF/MC1 map, candidate route, entry delay, portfolio rule or policy
cost is refit.  All paths enter five minutes after the decision and use exact
complete Kraken one-minute bars.  February--August 2025 is the tuning window,
September--December 2025 selects a family winner, and 2026 is evaluated only
after that selection is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_rich_policy_contract import (  # noqa: E402
    Exact1mRichV2ExecutionContract,
    RichExitExtensions,
    exact_1m_rich_v2_receipt,
    replay_exact_1m_rich_policy_v2,
)
from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams  # noqa: E402
from scripts.run_strict_r3_exact_1m_rich_extensions_hpo import (  # noqa: E402
    ExactPaths,
    _load_dataset,
    _load_frozen_policy,
    _portfolio_metrics,
    _resort,
    _window,
)


TUNE_START = pd.Timestamp("2025-02-01T00:00:00Z")
TUNE_END = pd.Timestamp("2025-09-01T00:00:00Z")
SELECT_START = pd.Timestamp("2025-09-01T00:00:00Z")
SELECT_END = pd.Timestamp("2026-01-01T00:00:00Z")
FROZEN_START = pd.Timestamp("2026-01-01T00:00:00Z")
FROZEN_END = pd.Timestamp("2026-08-01T00:00:00Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_empty(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.mkdir(parents=True)


def _frozen_smooth(path: Path) -> RichExitExtensions:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ext = RichExitExtensions(**dict(payload["extensions"]))
    ext.validate()
    expected = (1.5, 0.5, 1.5)
    actual = (
        float(ext.protection_activation_atr),
        float(ext.protection_strength),
        float(ext.protection_power),
    )
    if actual != expected:
        raise AssertionError("frozen smooth-policy winner is not the declared 1.5/0.5/1.5 control")
    return ext


def _geometry_arms(base: RichPolicyParams, smooth: RichExitExtensions) -> list[tuple[str, RichPolicyParams, RichExitExtensions]]:
    arms = [("smooth_control", base, smooth)]
    for value in (1.00, 1.125, 1.40, 1.55):
        arms.append((f"sl_atr_multiplier_{value:g}", replace(base, sl_atr_multiplier=value), smooth))
    for value in (0.90, 1.10, 1.50):
        arms.append((f"sl_atr_power_{value:g}", replace(base, sl_atr_power=value), smooth))
    for value in (3.50, 3.90, 4.80, 5.20):
        arms.append((f"sl_mult_{value:g}", replace(base, sl_mult=value), smooth))
    for value in (0.004, 0.008, 0.010):
        arms.append((f"sl_floor_{value:.3f}", replace(base, sl_abs_floor_pct=value), smooth))
    for value in (0.035, 0.040, 0.060, 0.070):
        arms.append((f"sl_cap_{value:.3f}", replace(base, sl_abs_cap_pct=value), smooth))
    return arms


def _time_arms(base: RichPolicyParams, smooth: RichExitExtensions) -> list[tuple[str, RichPolicyParams, RichExitExtensions]]:
    arms = [("smooth_control", base, smooth)]
    # start minute, maximum completed-bar MFE ATR, tightened stop distance ATR
    for start, max_mfe, distance in (
        (30, 0.25, 0.75),
        (30, 0.50, 1.00),
        (60, 0.25, 1.00),
        (60, 0.50, 1.25),
        (90, 0.25, 1.25),
        (90, 0.50, 1.50),
    ):
        extension = replace(
            smooth,
            time_stop_start_minutes=start,
            time_stop_max_mfe_atr=max_mfe,
            time_stop_distance_atr=distance,
        )
        arms.append((f"time_{start}m_mfe{max_mfe:g}_stop{distance:g}", base, extension))
    return arms


def _stepped_extension(*, locks: tuple[float, float, float]) -> RichExitExtensions:
    return RichExitExtensions(
        step_protection_activation_1_atr=1.0,
        step_protection_lock_1_atr=locks[0],
        step_protection_activation_2_atr=1.5,
        step_protection_lock_2_atr=locks[1],
        step_protection_activation_3_atr=2.0,
        step_protection_lock_3_atr=locks[2],
    )


def _stepped_arms(base: RichPolicyParams, smooth: RichExitExtensions) -> list[tuple[str, RichPolicyParams, RichExitExtensions]]:
    # The step arms turn off the old legacy protection because continuous
    # smooth protection already suppresses it.  This makes the comparison
    # genuinely continuous-versus-stepped rather than an undocumented blend.
    stepped_base = replace(base, capital_protect_mfe_mult=0.0, capital_protect_min_lock_bps=0.0)
    return [
        ("continuous_smooth_control", base, smooth),
        ("step_breakeven_025_075", stepped_base, _stepped_extension(locks=(0.0, 0.25, 0.75))),
        ("step_breakeven_050_100", stepped_base, _stepped_extension(locks=(0.0, 0.50, 1.00))),
        ("step_breakeven_000_050", stepped_base, _stepped_extension(locks=(0.0, 0.00, 0.50))),
        ("step_breakeven_050_125", stepped_base, _stepped_extension(locks=(0.0, 0.50, 1.25))),
    ]


def _replay(paths: ExactPaths, params: RichPolicyParams, extensions: RichExitExtensions) -> dict[str, np.ndarray]:
    replay = replay_exact_1m_rich_policy_v2(
        entry=paths.entry,
        atr=paths.atr,
        highs=paths.high,
        lows=paths.low,
        closes=paths.close,
        entry_timestamps=paths.rows["entry_ts"],
        params=params,
        median_atr_fraction=MEDIAN_ATR,
        extensions=extensions,
        contract=CONTRACT,
    )
    valid = np.asarray(replay["path_valid"], dtype=bool)
    if not valid.all() or not np.isfinite(np.asarray(replay["net_bps"], dtype=float)[valid]).all():
        raise AssertionError("complete path panel produced an incomplete policy outcome")
    return replay


def _record(
    *, family: str, phase: str, arm: str, paths: ExactPaths,
    params: RichPolicyParams, extensions: RichExitExtensions, include_frames: bool,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    replay = _replay(paths, params, extensions)
    metrics, frames = _portfolio_metrics(paths, replay, arm=f"{family}_{arm}_{phase}", include_frames=include_frames)
    reasons = pd.Series(np.asarray(replay["exit_reason"], dtype=object)).value_counts()
    row = {
        "family": family,
        "phase": phase,
        "arm": arm,
        "rows": int(len(paths.rows)),
        "row_net_ev_bps": float(np.mean(np.asarray(replay["net_bps"], dtype=float))),
        "row_mean_exit_minutes": float(np.mean(np.asarray(replay["exit_minute"], dtype=float) + 1.0)),
        "params_json": json.dumps(params.to_dict(), sort_keys=True),
        "extensions_json": json.dumps(asdict(extensions), sort_keys=True),
        "time_stop_exits": int(reasons.get("time_stop", 0)),
        "stepped_mfe_protect_exits": int(reasons.get("stepped_mfe_protect", 0)),
        "smooth_capital_exits": int(reasons.get("smooth_capital_protect", 0)),
        "hard_stop_exits": int(reasons.get("stop_loss", 0)),
        "timeout_exits": int(reasons.get("timeout_h12", 0)),
        **{key: value for key, value in metrics.items() if key != "portfolio_monthly"},
    }
    return row, frames, replay


def _top(rows: Iterable[dict[str, Any]], count: int) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["portfolio_selection_score"]),
            -float(row["portfolio_net_ev_bps_per_trade"]),
            -float(row["portfolio_total_net_bps"]),
            str(row["arm"]),
        ),
    )
    return [str(row["arm"]) for row in ordered[:count]]


def _write_frames(out: Path, prefix: str, frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        frame.to_parquet(out / f"{prefix}_{name}.parquet", index=False, compression="zstd")


def run(args: argparse.Namespace) -> Path:
    global MEDIAN_ATR, CONTRACT
    output = Path(args.out_dir).resolve()
    _assert_empty(output)
    params, MEDIAN_ATR, policy_audit = _load_frozen_policy(Path(args.base_policy))
    smooth = _frozen_smooth(Path(args.smooth_winner))
    CONTRACT = Exact1mRichV2ExecutionContract(entry_delay_minutes=5)
    paths = _resort(_load_dataset(Path(args.dataset), expected_delay=5))
    tune = _window(paths, TUNE_START, TUNE_END)
    select = _window(paths, SELECT_START, SELECT_END)
    frozen = _window(paths, FROZEN_START, FROZEN_END)
    families = {
        "stop_geometry": _geometry_arms(params, smooth),
        "time_stop": _time_arms(params, smooth),
        "stepped_mfe": _stepped_arms(params, smooth),
    }
    all_records: list[dict[str, Any]] = []
    winners: dict[str, dict[str, Any]] = {}
    for family, arms in families.items():
        by_name = {name: (arm_params, arm_extensions) for name, arm_params, arm_extensions in arms}
        tune_records: list[dict[str, Any]] = []
        for arm, arm_params, arm_extensions in arms:
            record, _, _ = _record(
                family=family, phase="tune_2025feb_aug", arm=arm, paths=tune,
                params=arm_params, extensions=arm_extensions, include_frames=False,
            )
            tune_records.append(record)
            all_records.append(record)
        selected_names = _top(tune_records, int(args.top_per_family))
        control_name = arms[0][0]
        selection_names = list(dict.fromkeys([control_name, *selected_names]))
        selection_records: list[dict[str, Any]] = []
        for arm in selection_names:
            arm_params, arm_extensions = by_name[arm]
            record, _, _ = _record(
                family=family, phase="select_2025sep_dec", arm=arm, paths=select,
                params=arm_params, extensions=arm_extensions, include_frames=False,
            )
            selection_records.append(record)
            all_records.append(record)
        winner_name = _top(selection_records, 1)[0]
        winner_params, winner_extensions = by_name[winner_name]
        # 2026 has no selection authority: the exact selected arm is replayed
        # once, with the same route and no parameter adjustment.
        frozen_record, frames, replay = _record(
            family=family, phase="frozen_2026", arm=winner_name, paths=frozen,
            params=winner_params, extensions=winner_extensions, include_frames=True,
        )
        all_records.append(frozen_record)
        _write_frames(output, f"{family}_winner_frozen_2026", frames)
        winners[family] = {
            "tune_top": selected_names,
            "selection_winner": winner_name,
            "frozen_2026": frozen_record,
            "params": winner_params.to_dict(),
            "extensions": asdict(winner_extensions),
            "exact_1m_receipt": exact_1m_rich_v2_receipt(
                params=winner_params, extensions=winner_extensions, replay=replay, contract=CONTRACT,
            ),
        }
    results = pd.DataFrame(all_records)
    results.to_parquet(output / "isolated_stop_ablation_results.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_exact1m_isolated_stop_ablations_v1",
        "research_only": True,
        "no_score_or_admission_retraining": True,
        "entry": "decision + 5 minutes; exact complete Kraken 1m path",
        "cost": "100 bps exactly once in rich-policy net; zero second auction debit",
        "windows": {
            "tune": [str(TUNE_START), str(TUNE_END)],
            "selection": [str(SELECT_START), str(SELECT_END)],
            "frozen": [str(FROZEN_START), str(FROZEN_END)],
        },
        "dataset": paths.audit,
        "base_policy": policy_audit,
        "smooth_control": asdict(smooth),
        "contract": CONTRACT.to_dict(),
        "contract_sha256": CONTRACT.hash,
        "top_per_family": int(args.top_per_family),
        "winners": winners,
        "code_sha256": _sha256(Path(__file__).resolve()),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=ROOT / "data_perp/artifacts/strict_r3_exact_1m_dual30_plus5m_dataset_2025_2026_101h_20260817_v2",
    )
    parser.add_argument(
        "--base-policy", type=Path,
        default=ROOT / "data_perp/artifacts/strict_r3_rich_policy_hpo_long_20260817_v1/frozen_challenger.json",
    )
    parser.add_argument(
        "--smooth-winner", type=Path,
        default=ROOT / "data_perp/artifacts/strict_r3_exact_1m_rich_extensions_hpo_decision2025_frozen2026_101h_20260817_v2/frozen_extensions_winner.json",
    )
    parser.add_argument(
        "--top-per-family", type=int, default=3)
    parser.add_argument(
        "--out-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
