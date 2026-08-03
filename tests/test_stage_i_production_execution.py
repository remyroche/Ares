from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd

from extreme_price_movements.packb_static_point_feature_loader import (
    FrozenFeatureContract,
    _feature_contract_digest,
)
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
)
from extreme_price_movements.stage_i_production_execution import (
    StageIProductionExecutionError,
    _coverage_records,
    load_stage_i_side_production_inputs,
    make_cached_stage_i_strict_generator,
    materialize_stage_i_selected_panels,
)
from extreme_price_movements.stage_i_production_oos import (
    StageIFeatureSelectionReuseException,
    StageIOOSCalendar,
    StageIProductionWinnerBundle,
    StageIWinnerCell,
    build_stage_i_production_plans,
)
from extreme_price_movements.stage_i_strict_oof import StageIStrictOOFResult
from extreme_price_movements.stage_i_winner_bundle import (
    load_stage_i_production_source_binding,
)


def _digest(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _contract() -> FrozenFeatureContract:
    fields = ("f_base", "f_meta")
    kwargs = {
        "feature_columns": fields,
        "candidate_universe_sha256": "1" * 64,
        "source_schema_sha256": "2" * 64,
        "raw_allowlist_sha256": "3" * 64,
        "generator_registry_sha256": "4" * 64,
        "store_scan_manifest_sha256": "5" * 64,
        "coverage_profile_sha256": "6" * 64,
        "min_exact_key_coverage": 0.99,
        "min_non_null_feature_coverage": 0.90,
        "max_feature_columns": 512,
        "coverage_admission_rejections": (),
    }
    return FrozenFeatureContract(
        **kwargs, feature_contract_sha256=_feature_contract_digest(**kwargs)
    )


def _write_partition(path, month: str, population: str, offset: int) -> dict:
    if month == "2026-07":
        start = "2026-07-10T18:00:00Z"
    else:
        start = f"{month}-01T00:00:00Z"
    signal = pd.date_range(start, periods=4, freq="h", tz="UTC")
    side = np.array(["long", "short", "long", "short"])
    net = np.array([180.0, -220.0, 35.0, 90.0]) + offset
    frame = pd.DataFrame({
        "candidate_id": [f"{month}-{offset}-{i}" for i in range(4)],
        "__ts__": signal, "__symbol__": ["BTC/USD:USD"] * 4,
        "side_name": side, "label_valid": True,
        "exact_gross_bps": net + 100.0, "exact_net_bps": net,
        "label_available_ts": signal + pd.Timedelta(hours=13),
        "t2_tp6_sl4_event": [2.0, 1.0, 0.0, 2.0],
        "robust_clear_event_b25": [1.0, 0.0, 0.0, 1.0],
        "robust_clear_soft_b25_t50": [0.9, 0.0, 0.3, 0.8],
    })
    frame.to_parquet(path, index=False)
    return {"path": str(path), "source_month": month, "population": population}


def _fixture(tmp_path):
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    contract = _contract()
    (input_root / "frozen_feature_contract.json").write_text(
        json.dumps(contract.to_dict()), encoding="utf-8"
    )
    records = [
        _write_partition(tmp_path / "2023.parquet", "2023-12", "historical_2022_2023", 0),
        _write_partition(tmp_path / "2024.parquet", "2024-01", "surface_2024", 10),
        _write_partition(tmp_path / "2026.parquet", "2026-07", "common30_2025_2026", 20),
    ]
    pd.DataFrame(records).to_parquet(input_root / "reference_partitions.parquet", index=False)
    input_manifest = {
        "schema": "stage_i_production_input_contract_v1", "status": "complete",
        "rows": 12, "min_signal_ts": "2023-12-01T00:00:00+00:00",
        "max_signal_ts": "2026-07-10T21:00:00+00:00",
        "feature_store": str(tmp_path / "unused_store"),
        "feature_contract_sha256": contract.feature_contract_sha256,
    }
    (input_root / "manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")
    source = load_stage_i_production_source_binding(input_root)
    cells = []
    for head in STAGE_I_ACTIVE_CONTRACTS:
        features = (
            ("f_base",) if head.layer == "base"
            else ("f_meta", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES)
        )
        params = (
            {"objective": "multiclass", "num_class": 3, "n_estimators": 4}
            if head.layer == "base" else {"objective": "huber", "n_estimators": 4}
        )
        selector = {"selected_feature_contract": list(features), "best_params": params}
        cells.append(StageIWinnerCell(
            contract=head, selected_feature_names=features, lgbm_params=params,
            selector_manifest=selector, selector_manifest_sha256=_digest(selector),
            source_manifest=source, source_manifest_sha256=_digest(source),
        ))
    bundle = StageIProductionWinnerBundle(
        cells=tuple(cells), code_revision="a" * 40,
        calendar=StageIOOSCalendar(
            "2024-01-01T00:00:00Z", "2026-07-10T21:00:00Z"
        ),
        feature_selection_exception=StageIFeatureSelectionReuseException(
            approved=True,
            selection_reference_start_utc="2023-12-01T00:00:00Z",
            selection_reference_end_utc="2026-07-10T21:00:00Z",
            rationale="approved test fixture reused backward",
        ),
    )
    return input_root, bundle


def _loader_factory(calls):
    def factory(contract, verify):
        def load(ledger, fields):
            calls.append((len(ledger), bool(verify), tuple(fields)))
            out = ledger[["candidate_id", "__ts__", "__symbol__"]].copy()
            base = pd.to_datetime(ledger["__ts__"], utc=True).astype("int64") // 3_600_000_000_000
            out["f_base"] = (base.to_numpy() % 101).astype(np.float32)
            out["f_meta"] = ((base.to_numpy() * 3 + np.arange(len(out))) % 137).astype(np.float32)
            return out[["candidate_id", "__ts__", "__symbol__", *fields]]
        return load
    return factory


def test_selected_panel_materialization_is_monthly_bound_and_restartable(tmp_path) -> None:
    input_root, bundle = _fixture(tmp_path)
    panel = tmp_path / "panel"
    calls = []
    manifest = materialize_stage_i_selected_panels(
        bundle, input_contract_dir=input_root, output_dir=panel,
        loader_factory=_loader_factory(calls),
    )
    assert len(calls) == 3
    assert [call[1] for call in calls] == [True, False, False]
    assert len(manifest["parts"]) == 6
    assert "2024-12" in manifest["calendar_gaps"]
    assert manifest["calendar_gap_disposition"].endswith("no_fabrication_or_backfill")
    coverage = pd.read_parquet(panel / "selected_raw_feature_coverage.parquet")
    assert coverage.status.eq("pass").all()
    assert set(coverage.scope) == {"whole_side", "evaluation_window"}

    # Simulate interruption after every monthly checkpoint but before the root
    # completion manifest. Resume must not touch the PIT loader.
    (panel / "manifest.json").unlink()
    resumed = materialize_stage_i_selected_panels(
        bundle, input_contract_dir=input_root, output_dir=panel, resume=True,
        loader_factory=lambda *_: (_ for _ in ()).throw(AssertionError("must reuse monthly checkpoints")),
    )
    assert resumed["status"] == "complete"

    inputs = load_stage_i_side_production_inputs(bundle, selected_panel_dir=panel)
    assert [source.side for source in inputs] == ["long", "short"]
    assert all(len(source.frame.columns) == 2 for source in inputs)
    assert all(
        pd.to_datetime(source.decision_timestamps, utc=True).equals(
            pd.to_datetime(source.signal_close_timestamps, utc=True) + pd.Timedelta(hours=1)
        ) for source in inputs
    )
    assert all(source.panel_manifest_sha256 == bundle.cells[0].source_manifest_sha256 for source in inputs)
    assert all(len(source.materialized_panel_manifest_sha256 or "") == 64 for source in inputs)
    assert all(len(source.materialized_panel_content_sha256 or "") == 64 for source in inputs)


def test_strict_oof_side_cache_reuses_hash_bound_result(tmp_path) -> None:
    input_root, bundle = _fixture(tmp_path)
    panel = tmp_path / "panel"
    materialize_stage_i_selected_panels(
        bundle, input_contract_dir=input_root, output_dir=panel,
        loader_factory=_loader_factory([]),
    )
    inputs = load_stage_i_side_production_inputs(bundle, selected_panel_dir=panel)
    plans, _ = build_stage_i_production_plans(bundle, inputs)
    calls = []

    def generate(plan):
        calls.append(plan.side)
        n = len(plan.frame)
        prediction = pd.DataFrame({"candidate_id": plan.candidate_ids, "side_name": plan.side})
        provenance = pd.DataFrame({"side": [plan.side], "strict_prior_resolved": [True]})
        return StageIStrictOOFResult(
            plan.side, prediction, provenance,
            {"side": plan.side, "strict_oof": True},
            {"side": plan.side, "rows": n},
        )

    cached = make_cached_stage_i_strict_generator(
        bundle=bundle, selected_panel_dir=panel,
        cache_dir=tmp_path / "oof_cache", generate=generate,
    )
    first = cached(plans[0])
    second = cached(plans[0])
    assert calls == ["long"]
    assert first.plan_summary == second.plan_summary
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


def test_production_coverage_uses_readiness_not_warmup_prefix(tmp_path) -> None:
    _, bundle = _fixture(tmp_path)
    ts = pd.date_range("2023-01-01", periods=100, freq="10D", tz="UTC")
    path = tmp_path / "warmup.parquet"
    pd.DataFrame({
        "__ts__": ts,
        "oi_dominance": np.r_[np.full(15, np.nan), np.arange(85)],
    }).to_parquet(path, index=False)
    audit = _coverage_records(
        {"long": [path], "short": []},
        {"long": ["oi_dominance"], "short": []},
        {"long": {"oi_dominance": ts[15].isoformat()}, "short": {}},
        bundle,
    )
    whole = audit.loc[audit.scope.eq("whole_side")].iloc[0]
    assert whole.finite_coverage == 0.85
    assert whole.post_readiness_finite_coverage == 1.0
    assert whole.required_evaluation_finite_coverage == 1.0
    assert whole.status == "pass"


def test_production_coverage_rejects_sporadic_post_readiness_gap(tmp_path) -> None:
    _, bundle = _fixture(tmp_path)
    ts = pd.date_range("2023-01-01", periods=100, freq="10D", tz="UTC")
    values = np.arange(100, dtype=float)
    values[20::4] = np.nan
    path = tmp_path / "sporadic.parquet"
    pd.DataFrame({"__ts__": ts, "sporadic": values}).to_parquet(path, index=False)
    with pytest.raises(StageIProductionExecutionError, match="coverage audit failed"):
        _coverage_records(
            {"long": [path], "short": []},
            {"long": ["sporadic"], "short": []},
            {"long": {"sporadic": ts[0].isoformat()}, "short": {}},
            bundle,
        )
