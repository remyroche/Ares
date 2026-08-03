from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "materialize_mayjul_direct_q25_causal_mapping.py"
SPEC = importlib.util.spec_from_file_location("mayjul_direct_q25_causal_mapping", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _artifact(name: str) -> pd.DataFrame:
    return pd.read_parquet(MODULE.DEFAULT_OUTPUT / f"{name}.parquet")


def test_q25_is_bit_identical_and_exact_identity_complete() -> None:
    provenance = _artifact("direct_q25_oof_provenance")
    waterfall = pd.read_parquet(MODULE.WATERFALL / "allscore_waterfall.parquet")
    keys = list(MODULE.IDENTITY)
    joined = waterfall.merge(provenance.loc[:, [*keys, "q25_net_bps"]], on=keys, how="inner", validate="one_to_one")
    assert len(joined) == len(waterfall) == 127777
    assert np.array_equal(joined["score_direct_q25_challenger_bps"].to_numpy(), joined["q25_net_bps"].to_numpy())


def test_fold_uniqueness_cutoff_causality_and_feature_availability() -> None:
    provenance = _artifact("direct_q25_oof_provenance")
    assert not provenance.duplicated(list(MODULE.IDENTITY)).any()
    assert provenance["oof_fold_name"].value_counts().to_dict() == {"recent_may": 63351, "recent_june": 49259, "recent_july": 15167}
    assert pd.to_datetime(provenance["max_training_label_resolution_utc"], utc=True).lt(pd.to_datetime(provenance["fit_cutoff_utc"], utc=True)).all()
    assert pd.to_datetime(provenance["fit_cutoff_utc"], utc=True).le(pd.to_datetime(provenance["execution_decision_utc"], utc=True)).all()
    assert pd.to_datetime(provenance["feature_available_at"], utc=True).lt(pd.to_datetime(provenance["execution_decision_utc"], utc=True)).all()
    assert pd.to_datetime(provenance["score_available_at"], utc=True).le(pd.to_datetime(provenance["execution_decision_utc"], utc=True)).all()


def test_mapping_audit_has_no_label_leakage_and_keeps_warmup() -> None:
    audit = _artifact("causal_mapping_audit")
    mapped = _artifact("causal_mapped_direct_q25_candidates")
    assert audit["strictly_resolved_before_snapshot"].all()
    assert (~audit["mapping_available"]).sum() == 1
    assert int((~mapped["mapped_eligible"].astype(bool)).sum()) == 2226
    assert mapped.loc[mapped["mapped_eligible"].astype(bool), "causal_mapped_direct_q25_ev"].notna().all()
