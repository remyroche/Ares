from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_materialized_trailing_label_topk_lgbm_hpo as runner
import scripts.verify_base_model_input_parity as verifier


class _ArtifactModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return values[:, 0]


def _build_artifacts(tmp_path: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    report_dir = tmp_path / "report"
    fold = "2026-04"
    keys = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z"], utc=True
            ),
            "__symbol__": ["BTC", "ETH"],
            "side": [1, -1],
        }
    )
    columns = ["raw", "aegmm", "side"]
    fills = {"raw": 2.0, "aegmm": -3.0, "side": 1.0}
    runner._save_base_fold_model(
        model_dir=report_dir / "models",
        fold={"fold": fold, "month": fold},
        model=_ArtifactModel(),
        feature_names=columns,
        x_train=pd.DataFrame({"raw": [1.0], "aegmm": [2.0], "side": [1.0]}),
        imputation_fill_values=fills,
        params={"target_mode": "target_soft", "weight_arm": "W0_base"},
        trial_number=1,
        seed=7,
        train_rows_available=1,
        train_rows_fit=1,
        valid_rows=len(keys),
    )
    raw = pd.DataFrame({"raw": [np.nan, 4.0]}, dtype=np.float32)
    generated = pd.DataFrame({"aegmm": [5.0, np.nan]}, dtype=np.float32)
    before_contract = pd.DataFrame(
        {
            "raw": raw["raw"].fillna(2.0),
            "aegmm": generated["aegmm"].fillna(-3.0),
            "side": keys["side"].astype(np.float32),
        }
    )
    x_valid = before_contract.astype(np.float32).astype(np.float16).astype(np.float32)
    runner._persist_oos_model_input_parity(
        parity_root=report_dir / "model_input_parity",
        fold=fold,
        valid=keys,
        x_valid=x_valid,
        valid_sides=np.asarray(["long", "short"]),
        feature_contracts={"shared": columns},
        model_side_scope="shared",
        anchor_rows=2,
    )
    return report_dir, raw, generated


def _patch_sources(monkeypatch, raw: pd.DataFrame, generated: pd.DataFrame) -> None:
    monkeypatch.setattr(
        verifier,
        "_sidecar_output_features",
        lambda _path: (["aegmm"], {"test": True}),
    )
    monkeypatch.setattr(
        verifier,
        "_load_feature_store_columns",
        lambda keys, **_kwargs: (raw.reset_index(drop=True).copy(), {"reader": "test"}),
    )

    def _read_sidecar(_path, keys, _features):
        out = keys.copy()
        out["aegmm"] = generated["aegmm"].to_numpy(dtype=np.float32)
        return out

    monkeypatch.setattr(verifier, "_read_sidecar_sample", _read_sidecar)


@pytest.mark.parametrize("hash_mode", ["all", "sample", "anchors"])
def test_reconstructs_imputed_float16_inputs_and_checks_persisted_contracts(
    tmp_path: Path, monkeypatch, hash_mode
) -> None:
    report_dir, raw, generated = _build_artifacts(tmp_path)
    _patch_sources(monkeypatch, raw, generated)

    result = verifier.verify_base_model_input_parity(
        report_dir=report_dir,
        feature_store=tmp_path / "features" / "20260711_070000",
        sidecar=tmp_path / "sidecar.parquet",
        hash_mode=hash_mode,
    )

    assert result["pass"] is True
    side = result["fold_reports"][0]["model_sides"][0]
    assert side["hash_match"] is True
    assert side["hash_rows_checked"] == (0 if hash_mode == "anchors" else 2)
    assert side["anchors_match_exactly"] is True
    assert side["matrix_hash_match"] is (True if hash_mode == "all" else None)


def test_rejects_anchor_and_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    report_dir, raw, generated = _build_artifacts(tmp_path)
    _patch_sources(monkeypatch, raw, generated)
    generated.loc[0, "aegmm"] = 9.0

    result = verifier.verify_base_model_input_parity(
        report_dir=report_dir,
        feature_store=tmp_path / "features" / "20260711_070000",
        sidecar=tmp_path / "sidecar.parquet",
        hash_mode="all",
    )

    assert result["pass"] is False
    side = result["fold_reports"][0]["model_sides"][0]
    assert side["hash_match"] is False
    assert side["anchors_match_exactly"] is False


@pytest.mark.parametrize(
    ("artifact", "mutate", "message"),
    [
        (
            "row_hashes",
            lambda frame: frame.assign(feature_contract_hash="wrong"),
            "row-hash feature contract mismatch",
        ),
        (
            "anchors",
            lambda frame: frame.assign(__symbol__="missing-anchor-key"),
            "anchors contain keys absent",
        ),
    ],
)
def test_rejects_persisted_key_and_contract_mismatches(
    tmp_path: Path, monkeypatch, artifact, mutate, message
) -> None:
    report_dir, raw, generated = _build_artifacts(tmp_path)
    _patch_sources(monkeypatch, raw, generated)
    path = next((report_dir / "model_input_parity").glob(f"*/{artifact}.parquet"))
    mutate(pd.read_parquet(path)).to_parquet(path, index=False)

    with pytest.raises(ValueError, match=message):
        verifier.verify_base_model_input_parity(
            report_dir=report_dir,
            feature_store=tmp_path / "features" / "20260711_070000",
            sidecar=tmp_path / "sidecar.parquet",
            hash_mode="all",
        )


def test_rejects_numeric_contract_feature_order_mismatch(tmp_path: Path, monkeypatch) -> None:
    report_dir, raw, generated = _build_artifacts(tmp_path)
    _patch_sources(monkeypatch, raw, generated)
    path = next((report_dir / "model_input_parity").glob("*/manifest.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contracts_by_model_side"]["shared"]["numeric_contract"][
        "feature_names_hash"
    ] = "sha256:wrong"
    # Recompute neither the outer nor numeric hash: the verifier must detect the
    # persisted contract corruption before replaying any score.
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="numeric contract hash is invalid"):
        verifier.verify_base_model_input_parity(
            report_dir=report_dir,
            feature_store=tmp_path / "features" / "20260711_070000",
            sidecar=tmp_path / "sidecar.parquet",
            hash_mode="all",
        )
