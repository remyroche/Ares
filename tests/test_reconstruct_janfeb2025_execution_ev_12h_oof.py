from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.reconstruct_janfeb2025_execution_ev_12h_oof import (
    BASE_SCORE,
    BASE_TARGET,
    DIRECT_SCORE,
    TARGET,
    add_candidate_context,
    coverage_by_side_month,
    deterministic_fit_sample,
    eligible_raw_features,
    generate_base_oof,
    generate_direct_ev_oof,
    load_external_deployed_policy_labels,
    select_features,
    source_paths,
    topk_metrics,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_feature_inventory_excludes_all_double_underscore_fields(tmp_path) -> None:
    paths = []
    for number in range(2):
        path = tmp_path / f"{number}.parquet"
        table = pa.table(
            {
                "raw_float": pa.array([1.0], type=pa.float32()),
                "raw_int": pa.array([1], type=pa.int16()),
                "side": pa.array([1], type=pa.int8()),
                "candidate_id": ["x"],
                "__outcome__": pa.array([1.0], type=pa.float32()),
                f"only_{number}": pa.array([1.0], type=pa.float32()),
            }
        )
        pq.write_table(table, path)
        paths.append(path)
    assert eligible_raw_features(paths, minimum_features=1) == [
        "raw_float",
        "raw_int",
    ]


def test_source_paths_expand_requested_months_for_both_sides(tmp_path) -> None:
    expected = []
    for side in ("long", "short"):
        for month in ("03", "04"):
            path = tmp_path / f"train_global_{side}_5_2025_{month}.parquet"
            path.touch()
            expected.append(path)
    assert source_paths(
        tmp_path, start_month="2025-03", end_month="2025-04"
    ) == expected


def test_coverage_summary_keeps_path_and_atr_attrition_separate() -> None:
    coverage = pd.DataFrame(
        {
            "candidate_month": ["2025-03"] * 4,
            "side_name": ["long"] * 4,
            "candidate_id": ["a", "b", "c", "d"],
            "complete_exact_1m_path": [1, 1, 1, 1],
            "complete_causal_atr": [1, 1, 1, 0],
            "complete_exact_label": [1, 1, 1, 0],
        }
    )
    result = coverage_by_side_month(coverage).iloc[0]
    assert result["exact_1m_path_coverage"] == 1.0
    assert result["causal_atr_coverage"] == 0.75
    assert result["exact_label_coverage"] == 0.75


def test_external_policy_labels_require_hashes_and_exact_timing(tmp_path) -> None:
    policy = tmp_path / "policy.json"
    spread = tmp_path / "spread.csv"
    policy.write_text('{"policy": "fixture"}\n')
    spread.write_text("symbol,p90_spread_bps\nBTC/USD:USD,2\n")
    ts = pd.Timestamp("2025-02-01T00:00:00Z")
    candidates = pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "candidate_id": ["row"],
            "candidate_month": ["2025-02"],
            "raw": [1.0],
        }
    )
    labels_path = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            "__ts__": [ts],
            "__symbol__": ["BTC/USD:USD"],
            "side_name": ["long"],
            "candidate_id": ["row"],
            "execution_decision_utc": [ts + pd.Timedelta(hours=1)],
            "execution_label_end_utc": [ts + pd.Timedelta(hours=13)],
            "execution_gross_ev_12h": [0.02],
            TARGET: [0.01],
            "execution_cost_return": [0.01],
            "execution_exit_reason": ["timeout"],
            "execution_exit_hour": [12.0],
            "execution_mfe_return_12h": [0.03],
            "execution_mae_return_12h": [0.01],
        }
    ).to_parquet(labels_path, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "accounting": {
                    "simulator": (
                        "extreme_price_movements.simple_policy_optimiser."
                        "simulate_and_score"
                    ),
                    "spread_baseline_sha256": _sha(spread),
                },
                "source": {"policy_sha256": _sha(policy)},
                "geometry": {
                    "fallback_rate": 1.0,
                    "side_archetype_rows": 0,
                },
                "exit_policy_contract": {"horizon_minutes": 720},
            }
        )
    )
    labels, coverage, contract = load_external_deployed_policy_labels(
        candidates,
        labels_path=labels_path,
        manifest_path=manifest,
        policy_path=policy,
        spread_baseline_path=spread,
    )
    assert len(labels) == 1
    assert labels[BASE_TARGET].between(0.0, 1.0).all()
    assert coverage["complete_exact_label"].all()
    assert contract["panels"][0]["checks"]["side_parent_fallback"] is True

    bad = json.loads(manifest.read_text())
    bad["geometry"]["fallback_rate"] = 0.5
    manifest.write_text(json.dumps(bad))
    with np.testing.assert_raises_regex(ValueError, "side_parent_fallback"):
        load_external_deployed_policy_labels(
            candidates,
            labels_path=labels_path,
            manifest_path=manifest,
            policy_path=policy,
            spread_baseline_path=spread,
        )


def test_feature_selection_and_fit_sample_are_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC"),
            "__symbol__": ["BTC_USD:USD"] * 200,
            BASE_TARGET: np.linspace(0.0, 1.0, 200),
            "strong": np.linspace(0.0, 1.0, 200),
            "weak": np.tile([0.0, 1.0], 100),
        }
    )
    selected, scores = select_features(frame, ["weak", "strong"], maximum=1)
    assert selected == ["strong"]
    assert scores["strong"] > scores["weak"]
    first = deterministic_fit_sample(frame, 17)
    second = deterministic_fit_sample(frame, 17)
    pd.testing.assert_frame_equal(first, second)


def test_candidate_context_is_timestamp_relative_and_finite() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2025-01-01T00:00Z"] * 3 + ["2025-01-01T01:00Z"] * 2,
                utc=True,
            ),
            "side_name": ["long"] * 5,
            BASE_SCORE: [0.1, 0.2, 0.3, 0.4, 0.4],
        }
    )
    result = add_candidate_context(frame)
    assert result.loc[:2, "candidate_group_size"].eq(3).all()
    assert result.loc[3:, "candidate_group_size"].eq(2).all()
    assert np.isfinite(
        result[
            [
                "base_margin_to_cutoff",
                "base_margin_to_cutoff_z",
                "base_score_z_within_timestamp",
                "base_score_rank_pct_within_timestamp",
            ]
        ].to_numpy()
    ).all()


def test_metric_slices_do_not_rerank_the_single_global_book() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-01-01", periods=20, freq="h", tz="UTC"),
            "candidate_month": ["2025-01"] * 10 + ["2025-02"] * 10,
            "side_name": ["long"] * 10 + ["short"] * 10,
            DIRECT_SCORE: np.arange(20, dtype=float),
            TARGET: np.arange(20, dtype=float) / 100.0,
        }
    )
    metrics = topk_metrics(frame)
    assert metrics["global"]["selected_rows"] == 2
    assert set(metrics["global_book_by_month"]) == {"2025-02"}
    assert set(metrics["global_book_by_side"]) == {"short"}
    assert metrics["diagnostic_month_local_pooled_top10"]["2025-01"][
        "selected_rows"
    ] == 1


def _synthetic_labels() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", "2025-02-28 23:00", freq="h", tz="UTC")
    rows = []
    for side_number, side in enumerate(("long", "short")):
        for number, ts in enumerate(timestamps):
            raw = np.sin(number / 13.0 + side_number)
            target = 0.01 * np.tanh(raw)
            rows.append(
                {
                    "__ts__": ts,
                    "__symbol__": f"S{number % 4}_USD:USD",
                    "side_name": side,
                    "candidate_id": f"{side}-{number}",
                    "execution_decision_utc": ts + pd.Timedelta(hours=1),
                    "execution_label_end_utc": ts + pd.Timedelta(hours=13),
                    "candidate_month": ts.strftime("%Y-%m"),
                    TARGET: target,
                    BASE_TARGET: 1.0 / (1.0 + np.exp(-target / 0.01)),
                    "raw_a": raw,
                    "raw_b": float(number % 9),
                }
            )
    return pd.DataFrame(rows)


def test_two_layer_oof_is_forward_and_uses_only_inner_oof_base_scores(
    monkeypatch,
) -> None:
    import scripts.reconstruct_janfeb2025_execution_ev_12h_oof as module

    monkeypatch.setattr(module, "MIN_BASE_TRAIN_ROWS", 10)
    monkeypatch.setattr(module, "MIN_META_TRAIN_ROWS", 10)
    monkeypatch.setattr(module, "MAX_FIT_ROWS", 500)
    labels = _synthetic_labels()
    base, base_audit = generate_base_oof(labels, ["raw_a", "raw_b"])
    direct, direct_audit = generate_direct_ev_oof(base)
    assert direct["__ts__"].min() == pd.Timestamp("2025-01-15T00:00:00Z")
    assert set(direct["candidate_month"]) == {"2025-01", "2025-02"}
    assert np.isfinite(direct[DIRECT_SCORE]).all()
    for fold in [*base_audit, *direct_audit]:
        if fold["status"] == "trained":
            assert pd.Timestamp(fold["max_train_label_end_utc"]) <= pd.Timestamp(
                fold["fold_start_utc"]
            )
    assert direct["base_oof_train_cutoff_utc"].le(direct["__ts__"]).all()
    assert direct["direct_oof_train_cutoff_utc"].le(direct["__ts__"]).all()
