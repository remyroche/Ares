#!/usr/bin/env python3
"""Seal the authoritative pre-2026 regime-overlay gamma decision tables."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
RAW = ART / "pre2026_regime_overlay_gamma_hpo_20260730_v1"
OVERLAY = ART / "pre2026_nested_residual_context_failure_overlay_20260730_v3"
OVERLAY_AUDIT = ART / "pre2026_nested_residual_context_failure_overlay_20260730_v4"
SOURCE = ART / "pre2026_oof_model_failure_incremental_value_20260730_v3"
OUT = ART / "pre2026_regime_overlay_gamma_hpo_20260730_v2"
GAMMAS = (0.125, 0.25, 0.5)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(frame: pd.DataFrame, score: str) -> dict[str, float]:
    q10 = frame[score].quantile(0.1)
    q90 = frame[score].quantile(0.9)
    high_low = (
        frame.loc[frame[score].ge(q90), "execution_net_ev_12h"].mean()
        - frame.loc[frame[score].le(q10), "execution_net_ev_12h"].mean()
    )
    return {
        "auc": roc_auc_score(frame.y, frame[score]),
        "ap": average_precision_score(frame.y, frame[score]),
        "brier": brier_score_loss(frame.y, frame[score]),
        "high_low_ev": high_low,
    }


def run() -> Path:
    if OUT.exists():
        raise RuntimeError(f"immutable output exists: {OUT}")
    for artifact in (RAW, OVERLAY, OVERLAY_AUDIT, SOURCE):
        expected = (artifact / "manifest.sha256").read_text().split()[0]
        if sha(artifact / "manifest.json") != expected:
            raise RuntimeError(f"unsealed prerequisite: {artifact}")

    pred = pd.read_parquet(RAW / "raw_predictions.parquet")
    pred["__ts__"] = pd.to_datetime(pred["__ts__"], utc=True)
    if pred.candidate_id.duplicated().any():
        raise RuntimeError("duplicate gamma candidate")

    source = pd.read_parquet(
        SOURCE / "materialized_targets.parquet",
        columns=["candidate_id", "__ts__", "execution_label_end_utc"],
    )
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
    source["execution_label_end_utc"] = pd.to_datetime(
        source["execution_label_end_utc"], utc=True
    )
    bound = pred.merge(
        source,
        on=["candidate_id", "__ts__"],
        how="left",
        validate="one_to_one",
    )
    if bound.execution_label_end_utc.isna().any():
        raise RuntimeError("missing label boundary")
    if (
        bound["__ts__"].dt.minute.ne(0).any()
        or bound["__ts__"].dt.second.ne(0).any()
        or bound.execution_label_end_utc.le(bound["__ts__"]).any()
        or bound.execution_label_end_utc.ge(pd.Timestamp("2026-01-01", tz="UTC")).any()
    ):
        raise RuntimeError("cadence or pre-2026 label boundary failure")

    prior = pd.read_parquet(OVERLAY / "bound_predictions.parquet")
    prior = prior[prior.arm.eq("regime")].pivot(
        index="candidate_id", columns="kind", values="p"
    )
    parity = bound.set_index("candidate_id")[["p_core", "p_0.5"]].join(
        prior, how="outer", validate="one_to_one"
    )
    if parity.isna().any().any() or len(parity) != len(bound):
        raise RuntimeError("gamma/overlay candidate-set mismatch")
    core_max_abs = float((parity.p_core - parity.core).abs().max())
    gamma_half_max_abs = float((parity["p_0.5"] - parity.overlay).abs().max())
    if core_max_abs > 1e-12 or gamma_half_max_abs > 1e-12:
        raise RuntimeError("gamma=.5 does not reproduce the audited overlay")

    pooled_rows: list[dict[str, object]] = []
    side_rows: list[dict[str, object]] = []
    for gamma in (0.0,) + GAMMAS:
        score = "p_core" if gamma == 0 else f"p_{gamma}"
        for era, frame in bound.groupby("era", sort=True):
            pooled_rows.append({"gamma": gamma, "era": era} | metric(frame, score))
        for (era, side), frame in bound.groupby(["era", "side_name"], sort=True):
            side_rows.append(
                {"gamma": gamma, "era": era, "side": side} | metric(frame, score)
            )
    pooled = pd.DataFrame(pooled_rows)
    sides = pd.DataFrame(side_rows)

    base = pooled[pooled.gamma.eq(0)].drop(columns="gamma").rename(
        columns={
            "auc": "auc_core",
            "ap": "ap_core",
            "brier": "brier_core",
            "high_low_ev": "high_low_ev_core",
        }
    )
    deltas = pooled[pooled.gamma.ne(0)].merge(base, on="era", validate="many_to_one")
    for name in ("auc", "ap", "brier", "high_low_ev"):
        deltas[f"{name}_delta"] = deltas[name] - deltas[f"{name}_core"]

    side_base = sides[sides.gamma.eq(0)][["era", "side", "auc"]].rename(
        columns={"auc": "auc_core"}
    )
    side_deltas = sides[sides.gamma.ne(0)].merge(
        side_base, on=["era", "side"], validate="many_to_one"
    )
    side_deltas["auc_delta"] = side_deltas.auc - side_deltas.auc_core

    summaries: list[dict[str, object]] = []
    for gamma, frame in deltas.groupby("gamma", sort=True):
        sf = side_deltas[side_deltas.gamma.eq(gamma)]
        long_med = sf.loc[sf.side.eq("long"), "auc_delta"].median()
        short_med = sf.loc[sf.side.eq("short"), "auc_delta"].median()
        summary = {
            "gamma": gamma,
            "held_eras": len(frame),
            "median_auc_delta": frame.auc_delta.median(),
            "min_auc_delta": frame.auc_delta.min(),
            "positive_auc_fraction": frame.auc_delta.gt(0).mean(),
            "median_ap_delta": frame.ap_delta.median(),
            "median_brier_delta": frame.brier_delta.median(),
            "long_median_auc_delta": long_med,
            "short_median_auc_delta": short_med,
            "median_high_low_ev_delta": frame.high_low_ev_delta.median(),
            "economic_improvement_fraction": frame.high_low_ev_delta.lt(0).mean(),
        }
        summary["eligible"] = bool(
            summary["median_auc_delta"] > 0
            and summary["positive_auc_fraction"] >= 0.75
            and summary["min_auc_delta"] >= -0.02
            and summary["median_brier_delta"] <= 0
            and summary["median_ap_delta"] >= 0
            and long_med >= 0
            and short_med >= 0
            and summary["median_high_low_ev_delta"] < 0
            and summary["economic_improvement_fraction"] >= 0.625
        )
        summaries.append(summary)
    eligibility = pd.DataFrame(summaries)
    eligible = eligibility[eligibility.eligible].sort_values(
        [
            "min_auc_delta",
            "median_auc_delta",
            "median_high_low_ev_delta",
            "gamma",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )
    selected_gamma = None if eligible.empty else float(eligible.iloc[0].gamma)

    stage = Path(tempfile.mkdtemp(dir=OUT.parent, prefix=f".{OUT.name}."))
    try:
        bound.to_parquet(stage / "bound_raw_predictions.parquet", index=False)
        pooled.to_csv(stage / "pooled_metrics.csv", index=False)
        sides.to_csv(stage / "side_metrics.csv", index=False)
        deltas.to_csv(stage / "pooled_deltas.csv", index=False)
        side_deltas.to_csv(stage / "side_deltas.csv", index=False)
        eligibility.to_csv(stage / "eligibility.csv", index=False)
        parity_audit = {
            "rows": len(parity),
            "candidate_sets_equal": True,
            "core_max_abs_error": core_max_abs,
            "gamma_half_max_abs_error": gamma_half_max_abs,
        }
        (stage / "parity_audit.json").write_text(
            json.dumps(parity_audit, indent=2, sort_keys=True) + "\n"
        )
        contract = {
            "schema": "pre2026_regime_overlay_gamma_hpo_v2",
            "status": "SEALED_PRE2026_GAMMA_HPO_NON_PROMOTION",
            "decision_cadence": "1h",
            "exact_replay_bar_cadence": "1m_labels_only",
            "gamma_grid": list(GAMMAS),
            "selected_gamma": selected_gamma,
            "authorized_for_2026": False,
            "selection_rule": (
                "eligible first; maximize worst-era AUC delta, then median AUC "
                "delta, then economic separation, then prefer smaller gamma"
            ),
            "implementation_sha256": {
                str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                str((ROOT / "scripts/run_regime_overlay_gamma_hpo.py").resolve()): sha(
                    ROOT / "scripts/run_regime_overlay_gamma_hpo.py"
                ),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
            },
            "no_2026": True,
        }
        (stage / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
        files = [path for path in stage.iterdir() if path.is_file()]
        manifest = {
            "schema": contract["schema"],
            "status": contract["status"],
            "promotion_eligible": selected_gamma is not None,
            "contract": contract,
            "inputs_sha256": {
                str((RAW / "manifest.json").resolve()): sha(RAW / "manifest.json"),
                str((RAW / "raw_predictions.parquet").resolve()): sha(
                    RAW / "raw_predictions.parquet"
                ),
                str((OVERLAY / "manifest.json").resolve()): sha(
                    OVERLAY / "manifest.json"
                ),
                str((OVERLAY_AUDIT / "manifest.json").resolve()): sha(
                    OVERLAY_AUDIT / "manifest.json"
                ),
                str((SOURCE / "manifest.json").resolve()): sha(
                    SOURCE / "manifest.json"
                ),
                str((SOURCE / "materialized_targets.parquet").resolve()): sha(
                    SOURCE / "materialized_targets.parquet"
                ),
            },
            "outputs_sha256": {path.name: sha(path) for path in files},
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (stage / "manifest.sha256").write_text(
            f"{sha(stage / 'manifest.json')}  manifest.json\n"
        )
        os.replace(stage, OUT)
    except Exception:
        for path in stage.iterdir():
            path.unlink()
        stage.rmdir()
        raise
    return OUT


if __name__ == "__main__":
    print(run())
