#!/usr/bin/env python3
"""Apply the selected learned GateProxy to strict-OOF Meta trial descriptors.

This is the reusable Meta-HPO objective layer:

``strict-OOF descriptor bank -> GateProxy ranking -> MC1 confirmation proposal``.

It never opens downstream MC1 labels, portfolio outputs, or live/exchange
state.  The emitted plan is deliberately a *proposal*: a full matched
six-month MC1 replay remains the only promotion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import fit_strict_r3_p8u_meta_downstream_proxy_v1 as proxy  # noqa: E402


SCHEMA = "strict_r3_p8u_meta_gateproxy_objective_scores_v2"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding_contract(path: Path, *, proxy_root: Path, choice_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    binding = json.loads(path.read_text())
    if binding.get("schema") != "strict_r3_p8u_meta_hpo_objective_binding_v2":
        raise AssertionError("invalid GateProxy objective binding")
    objective_path = (ROOT / str(binding["active_learned_objective"])).resolve()
    objective = json.loads(objective_path.read_text())
    if objective.get("schema") != "strict_r3_p8u_meta_hpo_gateproxy_objective_v2":
        raise AssertionError("binding does not resolve to the v2 GateProxy objective")
    selected = json.loads((choice_root / "gateproxy_grouped_portability_choice.json").read_text())
    model_name = str(selected.get("selected", {}).get("model", ""))
    configured_model = (ROOT / str(objective["objective"]["model_artifact"])).resolve()
    expected_model = (proxy_root / "models" / f"dgate_shrunk__{model_name}.joblib").resolve()
    if model_name != objective["objective"].get("surrogate") or configured_model != expected_model:
        raise AssertionError("binding/model-choice/proxy-root mismatch")
    if _sha256(expected_model) != objective["objective"].get("model_sha256"):
        raise AssertionError("bound GateProxy model hash mismatch")
    return binding, objective


def _read_descriptor_roots(roots: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for source in (path.resolve() for path in roots):
        correctness = json.loads((source / "correctness_report.json").read_text())
        if not all(value is True for value in correctness.values() if isinstance(value, bool)):
            raise AssertionError(f"{source}: descriptor receipt is incomplete")
        summary = pd.read_parquet(source / "trial_descriptor_summary.parquet")
        summary["descriptor_root"] = source.name
        parts.append(summary)
    table = pd.concat(parts, ignore_index=True)
    if table.trial.duplicated().any():
        raise AssertionError("trial appears in more than one descriptor root")
    return table


def _load_selected_model(proxy_root: Path, choice_root: Path) -> tuple[str, object, list[str], dict[str, object]]:
    choice = json.loads((choice_root / "gateproxy_grouped_portability_choice.json").read_text())
    selected = dict(choice.get("selected", {}))
    model_name = str(selected.get("model", ""))
    if model_name not in {"P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise"}:
        raise AssertionError("unsupported selected GateProxy model")
    # Preserve the historic script's safe unpickle compatibility for the
    # pairwise surrogate when a bundle was first serialized as __main__.
    setattr(sys.modules["__main__"], "PairwiseSurrogate", proxy.PairwiseSurrogate)
    payload = joblib.load(proxy_root / "models" / f"dgate_shrunk__{model_name}.joblib")
    if payload.get("schema") != proxy.SCHEMA or payload.get("target") != "dgate_shrunk":
        raise AssertionError("selected model is not a GateProxy payload")
    return model_name, payload["model"], list(payload["fields"]), choice


def _propose(score: pd.DataFrame, *, top_k: int, uncertainty_k: int, diverse_k: int) -> pd.DataFrame:
    available = score.sort_values(["gateproxy_score", "trial"], ascending=[False, True], kind="stable").copy()
    rows: list[pd.DataFrame] = []
    chosen: set[str] = set()
    top = available.head(top_k).copy(); top["proposal_role"] = "highest_predicted_gate_value"
    rows.append(top); chosen.update(top.trial.astype(str))
    uncertain = score.loc[~score.trial.astype(str).isin(chosen)].sort_values(
        ["gateproxy_uncertainty", "trial"], ascending=[False, True], kind="stable"
    ).head(uncertainty_k).copy()
    uncertain["proposal_role"] = "high_surrogate_uncertainty"
    rows.append(uncertain); chosen.update(uncertain.trial.astype(str))
    # One highest-scoring representative for each still-unrepresented
    # descriptor family.  It is intentionally an exploration/control cohort,
    # never a substitute for the value-ranked MC1 confirmations.
    remaining = score.loc[~score.trial.astype(str).isin(chosen)].copy()
    family_columns = ["target_family", "loss", "feature_contract"]
    picked: list[pd.Series] = []
    while len(picked) < diverse_k and not remaining.empty:
        existing = pd.concat(rows, ignore_index=True) if rows else score.iloc[0:0]
        counts = {
            column: set(existing[column].dropna().astype(str)) if column in existing else set()
            for column in family_columns
        }
        work = remaining.copy()
        work["novelty"] = sum(~work[column].astype(str).isin(counts[column]) for column in family_columns)
        item = work.sort_values(["novelty", "gateproxy_score", "trial"], ascending=[False, False, True], kind="stable").iloc[0].drop(labels="novelty")
        picked.append(item)
        remaining = remaining.loc[~remaining.trial.eq(item.trial)]
    if picked:
        diverse = pd.DataFrame(picked)
        diverse["proposal_role"] = "diverse_descriptor_control"
        rows.append(diverse)
    proposal = pd.concat(rows, ignore_index=True) if rows else score.iloc[0:0].copy()
    proposal = proposal.drop_duplicates("trial", keep="first").sort_values(
        ["proposal_role", "gateproxy_score", "trial"], ascending=[True, False, True], kind="stable"
    ).reset_index(drop=True)
    return proposal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--choice-root", type=Path, required=True)
    parser.add_argument("--objective-binding", type=Path, required=True)
    parser.add_argument("--descriptor-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--uncertainty-k", type=int, default=1)
    parser.add_argument("--diverse-k", type=int, default=1)
    args = parser.parse_args()
    if min(args.top_k, args.uncertainty_k, args.diverse_k) < 0 or args.top_k < 1:
        raise ValueError("proposal sizes must be non-negative and --top-k must be positive")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    proxy_root, choice_root = args.proxy_root.resolve(), args.choice_root.resolve()
    binding_path = args.objective_binding.resolve()
    correctness = json.loads((proxy_root / "correctness_report.json").read_text())
    if correctness.get("proxy_has_no_direct_live_or_model_score_authority") is not True:
        raise AssertionError("proxy receipt is incomplete")
    binding, objective = _binding_contract(binding_path, proxy_root=proxy_root, choice_root=choice_root)
    model_name, model, fields, choice = _load_selected_model(proxy_root, choice_root)
    descriptor = _read_descriptor_roots(args.descriptor_root)
    missing = sorted(set(fields).difference(descriptor.columns))
    if missing:
        raise AssertionError(f"descriptor fields missing: {missing}")
    setattr(sys.modules["__main__"], "PairwiseSurrogate", proxy.PairwiseSurrogate)
    result = descriptor.loc[:, ["trial", "target_family", "loss", "feature_family", "feature_contract", "descriptor_root"]].copy()
    result["gateproxy_score"] = proxy._predict(model, descriptor[fields])
    # Uncertainty comes from the four independently specified low-capacity
    # surrogates, not from future MC1 performance.
    bundle = joblib.load(proxy_root / "proxy_models.joblib")
    ensemble = np.column_stack([
        proxy._predict(bundle["models"][f"dgate_shrunk::{name}"], descriptor[fields])
        for name in ("P0_ridge", "P1_elastic_net", "P2_depth2_gbdt", "P3_pairwise")
    ])
    result["gateproxy_ensemble_mean"] = ensemble.mean(axis=1)
    result["gateproxy_uncertainty"] = ensemble.std(axis=1, ddof=1)
    result["gateproxy_rank"] = result.gateproxy_score.rank(method="first", ascending=False).astype(int)
    result = result.sort_values(["gateproxy_rank", "trial"], kind="stable").reset_index(drop=True)
    proposal = _propose(result, top_k=args.top_k, uncertainty_k=args.uncertainty_k, diverse_k=args.diverse_k)
    out.mkdir(parents=True)
    result.to_parquet(out / "gateproxy_scores.parquet", index=False, compression="zstd")
    proposal.to_parquet(out / "mc1_confirmation_proposal.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline Meta-HPO proxy scoring and MC1 confirmation proposal only; no outcome/MC1/portfolio/live/exchange input or mutation",
        "proxy_root": str(proxy_root), "choice_root": str(choice_root),
        "selected_model": model_name,
        "selected_model_path": str(proxy_root / "models" / f"dgate_shrunk__{model_name}.joblib"),
        "selected_model_sha256": _sha256(proxy_root / "models" / f"dgate_shrunk__{model_name}.joblib"),
        "objective_binding": str(binding_path),
        "objective_binding_sha256": _sha256(binding_path),
        "objective": objective,
        "descriptor_roots": [str(path.resolve()) for path in args.descriptor_root],
        "descriptor_fields": fields,
        "proposal": {"top_k": args.top_k, "uncertainty_k": args.uncertainty_k, "diverse_k": args.diverse_k},
        "choice_receipt": choice,
        "selection_authority": "shortlist proposal only; a new strict matched six-month MC1 replay is required for every promoted trial",
    })
    _once(out / "correctness_report.json", {
        "descriptor_inputs_are_strict_oof_and_receipted": True,
        "no_downstream_mc1_label_or_portfolio_output_was_opened": True,
        "gateproxy_uses_only_the_versioned_selected_model": True,
        "objective_binding_matches_selected_model_and_hash": True,
        "uncertainty_uses_only_frozen_surrogate_disagreement": True,
        "proposal_has_no_direct_trial_promotion_or_live_authority": True,
    })
    print(out)


if __name__ == "__main__":
    main()
