"""Production contract for the selected long-only LDF post-admission sizing.

Schema v4 deliberately uses the *same two-forest implementation* that was
selected in the compact 12-field MDA/HPO funnel.  The historical schema-v3
single-forest proxy remains loadable only through its legacy artifacts; it is
not a substitute for current training, replay, or inference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .n5_forest_support_sizing import (
    CURRENT_CANONICAL_LDF_PARAMS,
    CURRENT_CANONICAL_SCHEMA,
    MODEL_DISPLAY_NAME,
    MODEL_FAMILY,
    CurrentCanonicalLDFBundle,
    fit_n5_forest,
)
from .trust_sizing_ablation import ParentExpectation, discover_cmi_edges


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config/strict_r3_ldf_support_v4.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_n5_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load and validate the exact two-forest v4 LDF contract."""

    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != CURRENT_CANONICAL_SCHEMA:
        raise ValueError("not the current canonical LDF sizing contract")
    if payload.get("model_display_name") != MODEL_DISPLAY_NAME:
        raise ValueError("canonical LDF display-name contract mismatch")
    if payload.get("model_family") != MODEL_FAMILY:
        raise ValueError("canonical LDF family contract mismatch")
    fields = tuple(map(str, payload.get("features", ())))
    if len(fields) != 12 or len(set(fields)) != 12:
        raise ValueError("current canonical LDF requires 12 unique causal fields")
    if any(field.startswith("k09__cluster_") for field in fields):
        raise ValueError("canonical LDF cannot use raw rolling K9 memberships")
    if payload.get("ranking_changes") is not False or payload.get("admission_changes") is not False:
        raise ValueError("LDF may not change canonical ranking or admission")
    if payload["model"]["params"] != asdict(CURRENT_CANONICAL_LDF_PARAMS):
        raise ValueError("canonical LDF params do not match selected two-forest contract")
    if payload.get("canonical_arm") != "compact12_two_forest_meanrisk":
        raise ValueError("canonical LDF arm identity mismatch")
    return payload


def _sample_equal_month(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sort_values(["__decision_ts__", "candidate_id"], kind="stable").copy()
    month = frame["__decision_ts__"].dt.to_period("M").astype(str)
    tokens = sorted(month.unique())
    quota = max(1, cap // len(tokens))
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for token in tokens:
        index = np.flatnonzero(month.eq(token).to_numpy())
        if len(index) > quota:
            index = np.sort(rng.choice(index, quota, replace=False))
        chosen.append(index)
    selected = np.concatenate(chosen)
    if len(selected) > cap:
        selected = np.sort(rng.choice(selected, cap, replace=False))
    return frame.iloc[selected].sort_values(["__decision_ts__", "candidate_id"], kind="stable")


def train_canonical_n5_bundle(
    ledger: pd.DataFrame,
    *,
    cutoff: object,
    fields: Sequence[str] | None = None,
) -> CurrentCanonicalLDFBundle:
    """Fit the exact selected LDF from strictly prior-resolved rows.

    This reproduces the MDA/HPO population: a three-month prior window,
    resolved policy labels, prior causal-EV-map availability, a frozen top-30%
    score gate, equal-month sample cap, and a parent expectation learned only
    from the same resolved pre-cutoff rows.
    """

    contract = load_n5_contract()
    fields = tuple(fields or contract["features"])
    training = contract["training"]
    cutoff_ts = pd.Timestamp(cutoff)
    cutoff_ts = (
        cutoff_ts.tz_localize("UTC")
        if cutoff_ts.tzinfo is None
        else cutoff_ts.tz_convert("UTC")
    )
    required = {
        "candidate_id",
        "__decision_ts__",
        "policy_label_available_ts",
        "policy_path_valid",
        "policy_net_bps",
        "raw_expected_bps",
        "final_score",
        *fields,
    }
    if bool(training.get("requires_mapped_ev_available", False)):
        required.add("mapped_ev_available")
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"canonical LDF training ledger lacks fields: {missing}")
    work = ledger.copy()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True)
    work["policy_label_available_ts"] = pd.to_datetime(
        work["policy_label_available_ts"], utc=True,
    )
    start = cutoff_ts - pd.DateOffset(months=int(training["window_months"]))
    train_all = work.loc[
        work["__decision_ts__"].ge(start)
        & work["__decision_ts__"].lt(cutoff_ts)
        & work["policy_label_available_ts"].lt(cutoff_ts)
        & work["policy_path_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(work["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(work["raw_expected_bps"], errors="coerce"))
    ].copy()
    if bool(training.get("requires_mapped_ev_available", False)):
        train_all = train_all.loc[
            train_all["mapped_ev_available"].fillna(False).astype(bool)
        ].copy()
    if len(train_all) < 2_000:
        raise ValueError("canonical LDF has insufficient resolved prior support")

    parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
    train_all["parent_expected_bps"] = parent.predict(train_all["final_score"])
    floor = float(
        train_all["final_score"].quantile(
            1.0 - float(training["top_fraction_by_frozen_final_score"]),
        )
    )
    train = train_all.loc[
        pd.to_numeric(train_all["final_score"], errors="coerce").ge(floor)
    ].copy()
    train = _sample_equal_month(
        train,
        int(training["row_cap"]),
        CURRENT_CANONICAL_LDF_PARAMS.seed,
    )
    cmi_source = train.loc[
        pd.to_numeric(train["final_score"], errors="coerce").ge(
            pd.to_numeric(train["final_score"], errors="coerce").quantile(0.80),
        )
    ].copy()
    edges, _bins = discover_cmi_edges(
        cmi_source,
        fields,
        mode=CURRENT_CANONICAL_LDF_PARAMS.cmi_weighting,
        stable=True,
        max_edges=8,
        sample_cap=30_000,
    )
    forest, _ = fit_n5_forest(
        train,
        fields,
        edges,
        params=CURRENT_CANONICAL_LDF_PARAMS,
    )
    return CurrentCanonicalLDFBundle(
        forest=forest,
        parent_expectation=parent,
        training_score_floor=floor,
        cutoff=cutoff_ts,
    )


def score_canonical_n5_bundle(
    bundle: CurrentCanonicalLDFBundle,
    admitted_features: pd.DataFrame,
) -> pd.DataFrame:
    """Return a target-free LDF decomposition and bounded size multiplier."""

    required = {"candidate_id", "final_score", "raw_expected_bps", *bundle.fields}
    missing = sorted(required.difference(admitted_features.columns))
    if missing:
        raise ValueError(f"canonical LDF scoring frame lacks: {missing}")
    forbidden = [
        column
        for column in admitted_features.columns
        if any(
            token in column.lower()
            for token in ("future_", "policy_net_bps", "policy_gross_bps", "target", "label")
        )
    ]
    if forbidden:
        raise ValueError(f"canonical LDF scoring frame contains outcomes/labels: {forbidden}")
    prediction, multiplier = bundle.score(admitted_features)
    output = prediction.as_frame()
    output.insert(0, "candidate_id", admitted_features["candidate_id"].astype(str).to_numpy())
    output["portfolio_size_multiplier"] = multiplier
    output["n5_bundle_cutoff"] = bundle.cutoff
    output["n5_schema"] = bundle.schema
    output["ldf_model_name"] = MODEL_DISPLAY_NAME
    return output


def persist_canonical_n5_bundle(
    bundle: CurrentCanonicalLDFBundle,
    directory: Path,
    *,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"immutable LDF bundle already exists: {directory}")
    directory.mkdir(parents=True)
    payload = directory / "ldf_bundle.joblib"
    joblib.dump(bundle, payload, compress=3)
    manifest = {
        **bundle.manifest(),
        "bundle_file": payload.name,
        "bundle_sha256": _sha256(payload),
        "contract_path": str(CONTRACT_PATH),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "source_hashes": dict(source_hashes or {}),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_canonical_n5_bundle(directory: Path) -> CurrentCanonicalLDFBundle:
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("schema") != CURRENT_CANONICAL_SCHEMA:
        raise ValueError("not a current canonical LDF bundle")
    payload = directory / manifest["bundle_file"]
    if _sha256(payload) != manifest["bundle_sha256"]:
        raise ValueError("canonical LDF bundle hash mismatch")
    bundle = joblib.load(payload)
    if (
        not isinstance(bundle, CurrentCanonicalLDFBundle)
        or bundle.schema != CURRENT_CANONICAL_SCHEMA
    ):
        raise ValueError("canonical LDF payload type/schema mismatch")
    contract = load_n5_contract()
    if asdict(bundle.params) != contract["model"]["params"]:
        raise ValueError("canonical LDF payload params are not the selected two-forest contract")
    if tuple(bundle.fields) != tuple(contract["features"]):
        raise ValueError("canonical LDF payload fields do not match the frozen contract")
    return bundle


__all__ = [
    "CONTRACT_PATH",
    "load_canonical_n5_bundle",
    "load_n5_contract",
    "persist_canonical_n5_bundle",
    "score_canonical_n5_bundle",
    "train_canonical_n5_bundle",
]
