from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts/materialize_marapr2025_exact_h12_current_mapping.py"
SPEC = importlib.util.spec_from_file_location("marapr_current_mapping", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _artifact(name: str) -> pd.DataFrame:
    return pd.read_parquet(MODULE.DEFAULT_OUTPUT / name)


def test_common_ledger_has_exact_identity_and_h12_reconciliation() -> None:
    frame = _artifact("common_candidates.parquet")
    assert len(frame) == 140_682
    assert not frame.duplicated(list(MODULE.IDENTITY)).any()
    assert set(frame["candidate_month"].unique()) == {"2025-03", "2025-04"}
    assert np.allclose(
        frame["execution_gross_ev_12h"] - frame["execution_cost_return"],
        frame["execution_net_ev_12h"], rtol=0.0, atol=1e-10,
    )


def test_direct_provenance_is_available_and_causal() -> None:
    frame = _artifact("direct_q25_oof_provenance.parquet")
    assert len(frame) == 140_682
    assert set(frame["oof_fold_name"].unique()) == {"old_march", "old_april"}
    assert pd.to_datetime(frame["max_training_label_resolution_utc"], utc=True).lt(pd.to_datetime(frame["fit_cutoff_utc"], utc=True)).all()
    assert pd.to_datetime(frame["fit_cutoff_utc"], utc=True).le(pd.to_datetime(frame["execution_decision_utc"], utc=True)).all()
    assert pd.to_datetime(frame["score_available_at"], utc=True).le(pd.to_datetime(frame["execution_decision_utc"], utc=True)).all()


def test_current_mapping_is_utc_prior_resolved_and_sealed() -> None:
    audit = _artifact("mapping_audit.parquet")
    mapped = _artifact("mapped_candidates.parquet")
    manifest_path = MODULE.DEFAULT_OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert audit["strictly_resolved_before_snapshot"].all()
    assert audit["snapshot_utc"].map(lambda x: pd.Timestamp(x).tzinfo is not None).all()
    assert mapped.loc[mapped["common_mapping_eligible"], [f"mapped_{name}_ev" for name in MODULE.SCORE_COLUMNS]].notna().all().all()
    assert manifest["rows"]["source"] == len(mapped)
    assert manifest["promotion_eligible"] is False
    assert MODULE.sha256(manifest_path) == (MODULE.DEFAULT_OUTPUT / "manifest.sha256").read_text().split()[0]
