#!/usr/bin/env python3
"""Final strict-OOS audit for the frozen short P0 -> O -> C -> K0 research stack.

This independently reconstructs the selected Round-4 K0 map from its frozen
outer-OOS O/C ledger and proves that the published winner is identity- and
score-identical.  It is a research handoff, never a live/canonical mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_short_p0_oc_k0_round1 as r1  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_refinement as r3b  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round3_c_targets as r3  # noqa: E402
import run_strict_r3_short_p0_oc_k0_round4_k0_refinement as r4  # noqa: E402


SCHEMA = "strict_r3_short_p0_oc_k0_final_audit_v1"
ROUND4 = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round4_k0_refinement_20260822_v1"
OUT = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_final_audit_20260822_v1"
DEFAULT_SOURCE_PREDICTION = ROOT / "data_perp/artifacts/strict_r3_short_p0_oc_k0_round3_c_hpo_20260822_v1/C60_uniform_control_outer_oof_predictions.parquet"
DEFAULT_PUBLISHED = ROUND4 / "round4_winner_outer_oof_predictions.parquet"
DEFAULT_MU1 = ("isotonic", 0)
DEFAULT_MU0 = ("anchor5", 50)
DEFAULT_ADMISSION = ("absolute", 50.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in ([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())):
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _c_fields(source_prediction: Path) -> tuple[str, ...]:
    manifest = json.loads((r3b.OUT.parent / "strict_r3_short_p0_oc_k0_round3_c_refinement_20260822_v2" / "run_manifest.json").read_text())
    fields = tuple(manifest["conversion"]["feature_contracts"]["C60_mda"])
    if len(fields) != 60 or len(set(fields)) != 60:
        raise AssertionError("frozen C60 feature contract malformed")
    if source_prediction.name.startswith("C59_"):
        fields = tuple(field for field in fields if field != "ob_trade_size_to_l1_depth_z_24h")
    if len(fields) != len(set(fields)):
        raise AssertionError("selected C feature contract is not unique")
    return fields


def _feature_audit(source_prediction: Path) -> pd.DataFrame:
    frame, o_fields, _, _ = r3._load_frame()
    c_fields = _c_fields(source_prediction)
    valid = frame.loc[r1._valid_label(frame)].copy()
    rows = []
    for role, fields in (("O", o_fields), ("C", c_fields)):
        for field in fields:
            values = pd.to_numeric(valid[field], errors="coerce")
            rows.append({
                "layer": role, "feature": field, "valid_rows": int(len(valid)),
                "coverage": float(values.notna().mean()),
                "nunique": int(values.dropna().nunique()),
                "variance": float(values.var(ddof=0)) if values.notna().any() else float("nan"),
                "passes_coverage_90": bool(values.notna().mean() >= .90),
                "passes_variance": bool(values.dropna().nunique() > 1),
            })
    output = pd.DataFrame(rows)
    if not output["passes_coverage_90"].all() or not output["passes_variance"].all():
        failed = output.loc[~(output["passes_coverage_90"] & output["passes_variance"]), "feature"].tolist()
        raise AssertionError(f"frozen O/C contract fails current coverage/variance audit: {failed}")
    return output


def _parity(rebuilt: pd.DataFrame, published: pd.DataFrame) -> dict[str, float | int]:
    fields = [
        "opportunity_probability_round4", "k0_mu1_round4_bps", "k0_mu0_round4_bps",
        "K0_expected_policy_net_bps", "K0_train_p80_expected_policy_net_bps",
    ]
    left = rebuilt.loc[:, ["candidate_id", *fields]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    right = published.loc[:, ["candidate_id", *fields]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not left["candidate_id"].equals(right["candidate_id"]):
        raise AssertionError("final K0 reconstruction changed candidate identities")
    result: dict[str, float | int] = {"candidate_ids": int(len(left))}
    for field in fields:
        delta = float(np.max(np.abs(left[field].to_numpy(float) - right[field].to_numpy(float))))
        if delta > 2e-6:
            raise AssertionError(f"published final K0 result mismatch for {field}: {delta}")
        result[f"{field}_max_abs_delta"] = delta
    return result


def _table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def run(
    out: Path,
    source_prediction: Path = DEFAULT_SOURCE_PREDICTION,
    published_prediction: Path = DEFAULT_PUBLISHED,
    mu1: tuple[str, int] = DEFAULT_MU1,
    mu0: tuple[str, int] = DEFAULT_MU0,
    admission: tuple[str, float] = DEFAULT_ADMISSION,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    source_prediction, published_prediction = Path(source_prediction), Path(published_prediction)
    ledger, source_hashes = r4._load_ledger(source_prediction)
    rebuilt, map_audit = r4._replay(ledger, mu1=mu1, mu0=mu0, admission=admission)
    published = pd.read_parquet(published_prediction)
    parity = _parity(rebuilt, published)
    formula = (
        rebuilt["opportunity_probability_round4"].to_numpy(float) * rebuilt["k0_mu1_round4_bps"].to_numpy(float)
        + (1.0 - rebuilt["opportunity_probability_round4"].to_numpy(float)) * rebuilt["k0_mu0_round4_bps"].to_numpy(float)
    )
    formula_delta = float(np.max(np.abs(formula - rebuilt["K0_expected_policy_net_bps"].to_numpy(float))))
    if formula_delta > 3e-5:
        raise AssertionError(f"K0 analytic formula mismatch: {formula_delta}")
    complete = map_audit.loc[map_audit["status"].eq("complete")].copy()
    if complete.empty or not pd.to_datetime(complete["history_max_label_available_at"], utc=True).lt(pd.to_datetime(complete["held_month"] + "-01", utc=True)).all():
        raise AssertionError("K0 map history is not strictly resolved before held month")
    if admission[0] == "absolute" and not np.isclose(rebuilt["K0_train_p80_expected_policy_net_bps"].to_numpy(float), float(admission[1]), rtol=0.0, atol=1e-6).all():
        raise AssertionError("final admission is not the declared absolute threshold")
    arm_name = f"P0_O250H6_C3C{len(_c_fields(source_prediction))}_K0_{mu0[0]}k{mu0[1]}_{admission[0]}{int(admission[1])}"
    monthly, era, summary = r4._metrics(rebuilt, arm_name)
    features = _feature_audit(source_prediction)
    selected = rebuilt.loc[rebuilt["K0_expected_policy_net_bps"].ge(float(admission[1]))].copy()
    unknown_selected = selected.loc[~r4._valid(selected)]
    contract = {
        "P0": "frozen target-free short P0 candidate source",
        "O": {"event": "mfe_6h_bps > 250", "model": "frozen Round-2 binary LightGBM", "features": 45, "calibration": "Platt", "weights": "uniform"},
        "C": {"target": "C3 normalized regret ordinal among true O-positive training rows", "model": "default C/uniform LightGBM; independent HPO did not advance", "features": len(_c_fields(source_prediction)), "weights": "uniform"},
        "K0": {"formula": "p(O)*mu1(C)+(1-p(O))*mu0(P0 anchor)", "mu1": f"{mu1[0]} k={mu1[1]}", "mu0": f"P0-anchor quintile empirical-Bayes k={mu0[1]}" if mu0[0] == "anchor5" else f"global empirical-Bayes k={mu0[1]}", "admission": f"mapped expected policy net {admission[0]} {admission[1]}"},
        "excluded": ["MC1 mapper", "risk head", "consensus", "live/canonical authority"],
    }
    correctness = {
        "candidate_identity_parity": parity,
        "formula_max_abs_delta": formula_delta,
        "all_complete_map_histories_resolved_before_held_month": True,
        "final_admission": {"kind": admission[0], "value": admission[1]},
        "target_free_scoring_rows": int(len(rebuilt)),
        "invalid_rows_scored_but_not_economic_training": int((~r4._valid(rebuilt)).sum()),
        "selected_rows_without_resolved_outcome": int(len(unknown_selected)),
        "no_extra_layer_columns": not any("mc1" in col.lower() or "consensus" in col.lower() for col in rebuilt.columns),
    }
    manifest = {
        "schema": SCHEMA, "status": "complete", "side": "short",
        "scope": "final strict-OOS research audit; neither canonical nor live promotion",
        "contract": contract, "summary": summary, "correctness": correctness,
        "sources": {"source_prediction": str(source_prediction), "source_prediction_sha256": _sha256(source_prediction), "published_prediction": str(published_prediction), "published_prediction_sha256": _sha256(published_prediction), **source_hashes},
    }
    out.mkdir(parents=True)
    rebuilt.to_parquet(out / "final_stack_outer_oof_predictions.parquet", index=False, compression="zstd")
    map_audit.to_parquet(out / "final_stack_map_audit.parquet", index=False, compression="zstd")
    monthly.to_parquet(out / "final_stack_monthly_metrics.parquet", index=False, compression="zstd")
    era.to_parquet(out / "final_stack_era_metrics.parquet", index=False, compression="zstd")
    features.to_parquet(out / "final_stack_feature_audit.parquet", index=False, compression="zstd")
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        f"# Short P0 -> O250/H6 -> C3/C{len(_c_fields(source_prediction))} -> K0 final strict-OOS audit", "",
        "Research winner only; no live or canonical promotion is implied.", "",
        "## Stack", "", "```json", json.dumps(contract, indent=2), "```", "",
        "## 2025–2026 economics", "", _table(pd.DataFrame([summary])), "",
        "## Era metrics", "", _table(era), "",
        "## Correctness", "", "```json", json.dumps(correctness, indent=2), "```", "",
    ]
    (out / "SHORT_P0_OC_K0_FINAL_AUDIT.md").write_text("\n".join(report))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--source-prediction", type=Path, default=DEFAULT_SOURCE_PREDICTION)
    parser.add_argument("--published-prediction", type=Path, default=DEFAULT_PUBLISHED)
    parser.add_argument("--mu1-kind", default=DEFAULT_MU1[0])
    parser.add_argument("--mu1-k", type=int, default=DEFAULT_MU1[1])
    parser.add_argument("--mu0-kind", default=DEFAULT_MU0[0])
    parser.add_argument("--mu0-k", type=int, default=DEFAULT_MU0[1])
    parser.add_argument("--admission-kind", default=DEFAULT_ADMISSION[0])
    parser.add_argument("--admission-value", type=float, default=DEFAULT_ADMISSION[1])
    args = parser.parse_args()
    print(run(
        args.out, args.source_prediction, args.published_prediction,
        (args.mu1_kind, args.mu1_k), (args.mu0_kind, args.mu0_k),
        (args.admission_kind, args.admission_value),
    ))


if __name__ == "__main__":
    main()
