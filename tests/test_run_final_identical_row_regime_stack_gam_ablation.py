import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_oof_stack import RegimeOOFStackError
from scripts.run_final_identical_row_regime_stack_gam_ablation import (
    ARMS,
    REGIME_SOURCE,
    SIDECAR_SCHEMA,
    TRANSITION_SOURCE,
    _scores as load_scores,
    run,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scores(start: str, periods: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="12h", tz="UTC")
    n = np.arange(periods)
    net = np.where(n % 3, .01, -.012)
    return pd.DataFrame({
        "candidate_id": [f"{pd.Timestamp(t).isoformat()}|{'long' if i % 2 else 'short'}" for i, t in enumerate(ts)],
        "__ts__": ts, "__symbol__": np.where(n % 3, "BTC/USD:USD", "ETH/USD:USD"), "side_name": np.where(n % 2, "long", "short"),
        "execution_label_end_utc": ts + pd.Timedelta(hours=12), "execution_net_ev_12h": net,
        "execution_gross_ev_12h": net + .01, "execution_cost_return": .01,
        "__first_touch_target_soft__": (n % 7) / 7., "score_base_alpha": (n % 11) / 11.,
        "score_residual_expected_ev": net + (n % 5 - 2) / 1000.,
    })


def _sidecars(scores: pd.DataFrame, historical: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame({"source_utc": scores["__ts__"].drop_duplicates().sort_values().to_numpy()})
    x = np.arange(len(out), dtype=float)
    regime = out.copy()
    for i, field in enumerate(REGIME_SOURCE.values()):
        regime[field] = np.sin(x / (i + 2))
    regime["bocpd_regime_available"] = True
    regime["bocpd_ood_available"] = False
    regime["bocpd_ood_score"] = np.nan
    transition = out.copy()
    for i, field in enumerate(TRANSITION_SOURCE.values()):
        transition[field] = 0.1 + (i + 1) * .01 + np.cos(x / (i + 2)) * .01
    transition["lgbm_transition_available"] = True
    transition["lgbm_ood_available"] = False
    transition["lgbm_ood_score"] = np.nan
    # BOCPD provenance is intentionally duplicated: the real authority emits
    # it in both hourly sidecars and the runner must check it agrees.
    for frame in (regime, transition):
        frame["provenance_partition_bocpd"] = "blocked_oof_2022_2025" if historical else "untouched_2026_forward"
        frame["train_end_exclusive_utc_bocpd"] = frame.source_utc - pd.Timedelta(hours=1) if historical else pd.Timestamp("2026-01-01", tz="UTC")
        frame["fit_label_resolution_max_utc_bocpd"] = frame.source_utc - pd.Timedelta(hours=2) if historical else pd.NaT
    transition["provenance_partition_lgbm"] = "blocked_oof_2022_2025" if historical else "untouched_2026_forward"
    transition["train_end_exclusive_utc_lgbm"] = transition.source_utc - pd.Timedelta(hours=1) if historical else pd.Timestamp("2026-01-01", tz="UTC")
    transition["fit_label_resolution_max_utc_lgbm"] = transition.source_utc - pd.Timedelta(hours=2) if historical else pd.NaT
    return regime, transition


def _manifest(tmp_path, historical, current):
    regime_h, transition_h = _sidecars(historical, True)
    regime_c, transition_c = _sidecars(current, False)
    regime = pd.concat([regime_h, regime_c], ignore_index=True)
    transition = pd.concat([transition_h, transition_c], ignore_index=True)
    r, t = tmp_path / "soft_regime_hourly.parquet", tmp_path / "soft_transition_hourly.parquet"
    regime.to_parquet(r, index=False); transition.to_parquet(t, index=False)
    payload = {"schema": SIDECAR_SCHEMA, "status": "SEALED_CAUSAL_SOFT_REGIME_TRANSITION_SIDECARS", "model_sample_cadence": "1h", "assessment_sample_cadence": "1h", "exact_replay_bar_cadence": "1m_labels_only", "historical_contract": "blocked-OOF hourly predictions using only fit labels resolved before each fold train end", "forward_contract": "untouched 2026 hourly forward predictions", "outputs_sha256": {r.name: _sha(r), t.name: _sha(t)}}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(payload)); (tmp_path / "manifest.sha256").write_text(f"{_sha(path)}  manifest.json\n"); return path


def test_final_runner_declares_all_requested_placements_and_layers():
    assert {arm.placement for arm in ARMS} == {"baseline", "base", "residual_trust", "additive_bounded_gam"}
    assert {arm.context for arm in ARMS} == {"none", "regime", "transition", "combined"}
    assert all(arm.name != "diagonal_state_id" for arm in ARMS)


def test_final_runner_is_fail_closed_without_sealed_authoritative_manifest(tmp_path):
    scores = _scores("2025-01-01", 100); h, c = tmp_path / "h.parquet", tmp_path / "c.parquet"; scores.to_parquet(h); scores.to_parquet(c)
    bad = tmp_path / "bad.json"; bad.write_text(json.dumps({"schema": "wrong", "status": "SEALED"}))
    with pytest.raises(RegimeOOFStackError, match="authoritative sealed"):
        run(sidecar_manifest=bad, historical_scores=h, current_scores=c, output=tmp_path / "out", min_train_rows=20)


def test_final_runner_uses_fixed_global_book_and_emits_all_metric_surfaces(tmp_path):
    history = _scores("2025-01-01", 520); current = _scores("2026-01-10", 80)
    h, c = tmp_path / "history.parquet", tmp_path / "current.parquet"; history.to_parquet(h, index=False); current.to_parquet(c, index=False)
    manifest = _manifest(tmp_path, history, current)
    out = run(sidecar_manifest=manifest, historical_scores=h, current_scores=c, output=tmp_path / "out", oof_start="2025-04-01T00:00:00Z", min_train_rows=30, max_map_age_days=365)
    summary = pd.read_csv(out / "metrics_summary.csv")
    periods = pd.read_parquet(out / "period_metrics.parquet")
    assert set(summary.arm) == {arm.name for arm in ARMS}
    assert {"week_net_ev_q10", "month_net_ev_q50", "calibration_mae_decile", "mapped_tie_mass", "selected_asset_hhi"}.issubset(summary.columns)
    weekly = periods.loc[periods.period_type.eq("week")]
    assert weekly.groupby("arm").global_selected_rows.sum().eq(8).all()  # ceil(80 * 10%) once, never per-period


def test_final_runner_rejects_raw_state_id_even_when_other_context_is_present(tmp_path):
    history = _scores("2025-01-01", 100); current = _scores("2026-01-10", 30)
    h, c = tmp_path / "h.parquet", tmp_path / "c.parquet"; history.to_parquet(h, index=False); current.to_parquet(c, index=False)
    regime, transition = _sidecars(history, True); regime["regime_state_id"] = 2
    current_regime, current_transition = _sidecars(current, False)
    # Build a valid sealed container, then replace the checksum-bound regime
    # file with a forbidden raw identity so the loader rejects before fitting.
    manifest = _manifest(tmp_path, history, current)
    path = tmp_path / "soft_regime_hourly.parquet"; pd.concat([regime, current_regime], ignore_index=True).to_parquet(path, index=False)
    payload = json.loads(manifest.read_text()); payload["outputs_sha256"][path.name] = _sha(path); manifest.write_text(json.dumps(payload)); (tmp_path / "manifest.sha256").write_text(f"{_sha(manifest)}  manifest.json\n")
    with pytest.raises(RegimeOOFStackError, match="state identity"):
        run(sidecar_manifest=manifest, historical_scores=h, current_scores=c, output=tmp_path / "out", min_train_rows=20)


def test_final_runner_rejects_non_hourly_authoritative_sidecar_rows(tmp_path):
    history = _scores("2025-01-01", 100); current = _scores("2026-01-10", 30)
    h, c = tmp_path / "h.parquet", tmp_path / "c.parquet"; history.to_parquet(h, index=False); current.to_parquet(c, index=False)
    manifest = _manifest(tmp_path, history, current)
    path = tmp_path / "soft_transition_hourly.parquet"; frame = pd.read_parquet(path); frame.loc[0, "source_utc"] = pd.Timestamp("2025-01-01T00:30:00Z"); frame.to_parquet(path, index=False)
    payload = json.loads(manifest.read_text()); payload["outputs_sha256"][path.name] = _sha(path); manifest.write_text(json.dumps(payload)); (tmp_path / "manifest.sha256").write_text(f"{_sha(manifest)}  manifest.json\n")
    with pytest.raises(RegimeOOFStackError, match="exactly one row per 1h"):
        run(sidecar_manifest=manifest, historical_scores=h, current_scores=c, output=tmp_path / "out", min_train_rows=20)


def test_final_runner_rejects_historical_hourly_context_with_forward_provenance(tmp_path):
    history = _scores("2025-01-01", 100); current = _scores("2026-01-10", 30)
    h, c = tmp_path / "h.parquet", tmp_path / "c.parquet"; history.to_parquet(h, index=False); current.to_parquet(c, index=False)
    manifest = _manifest(tmp_path, history, current)
    path = tmp_path / "soft_transition_hourly.parquet"; frame = pd.read_parquet(path)
    frame.loc[frame.source_utc.lt(pd.Timestamp("2026-01-01", tz="UTC")), "provenance_partition_lgbm"] = "untouched_2026_forward"
    frame.to_parquet(path, index=False)
    payload = json.loads(manifest.read_text()); payload["outputs_sha256"][path.name] = _sha(path); manifest.write_text(json.dumps(payload)); (tmp_path / "manifest.sha256").write_text(f"{_sha(manifest)}  manifest.json\n")
    with pytest.raises(RegimeOOFStackError, match="provenance"):
        run(sidecar_manifest=manifest, historical_scores=h, current_scores=c, output=tmp_path / "out", min_train_rows=20)


def test_direct_script_bootstraps_the_repository_import_path():
    source = Path("scripts/run_final_identical_row_regime_stack_gam_ablation.py").read_text()
    assert source.index("sys.path.insert") < source.index(
        "from extreme_price_movements.regime_oof_stack"
    )


def test_ev_mapping_direction_and_global_tie_break_are_economically_oriented():
    source = Path("scripts/run_final_identical_row_regime_stack_gam_ablation.py").read_text()
    assert 'IsotonicRegression(out_of_bounds="clip", increasing=True)' in source
    assert 'sort_values(["mapped_score","raw_score","candidate_id"]' in source


def test_historical_lineage_aliases_are_conflict_checked_and_coalesced(tmp_path):
    frame = _scores("2025-01-01", 20)
    expected_base = frame["score_base_alpha"].copy()
    expected_residual = frame["score_residual_expected_ev"].copy()
    expected_end = frame["execution_label_end_utc"].copy()
    frame["base_oof_score"] = expected_base
    frame["residual_expected_ev"] = expected_residual
    frame["execution_label_available_at"] = expected_end
    frame[["score_base_alpha", "score_residual_expected_ev", "execution_label_end_utc"]] = np.nan
    path = tmp_path / "aliases.parquet"
    frame.to_parquet(path, index=False)
    loaded = load_scores(path)
    assert np.allclose(loaded["score_base_alpha"], expected_base)
    assert np.allclose(loaded["score_residual_expected_ev"], expected_residual)
    assert loaded["execution_label_end_utc"].equals(pd.to_datetime(expected_end, utc=True))
