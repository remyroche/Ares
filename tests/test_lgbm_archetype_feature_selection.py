from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements import lgbm_pipeline as lp


def _univariate_stats(columns: list[str], selected: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": columns,
            "passed": [name == selected for name in columns],
            "univariate_j": [1.0 if name == selected else 0.1 for name in columns],
            "J_pos_median": [1.0, 0.1],
            "J_neg_median": [0.1, 0.1],
            "direction": [1, 1],
            "direction_stability": [1.0, 1.0],
            "direction_margin_median": [1.0, 0.1],
            "precision20_norm_median": [1.0, 0.1],
            "lift20_median": [1.0, 0.1],
            "monotonicity_median": [1.0, 0.1],
            "pass_precision": [name == selected for name in columns],
            "pass_lift": [name == selected for name in columns],
            "pass_monotonicity": [name == selected for name in columns],
        }
    )


def test_archetype_prescreen_config_defaults_and_normalizes_booleans() -> None:
    assert lp._resolve_lgbm_archetype_prescreen_config(None) == {
        "archetype_univariate_prescreen_enabled": True,
        "archetype_relief_prescreen_enabled": True,
    }
    assert lp._resolve_lgbm_archetype_prescreen_config(
        {
            "archetype_univariate_prescreen_enabled": "false",
            "archetype_relief_prescreen_enabled": "YES",
        }
    ) == {
        "archetype_univariate_prescreen_enabled": False,
        "archetype_relief_prescreen_enabled": True,
    }


def test_univariate_prescreen_switches_between_global_and_archetype_modes(
    monkeypatch,
) -> None:
    x = pd.DataFrame({"feature_a": [0.0, 1.0], "feature_b": [1.0, 0.0]})
    y = np.asarray([0.0, 1.0], dtype=np.float32)
    calls: list[str] = []

    def fake_global(*args, **kwargs):
        calls.append("global")
        return ["feature_a"], _univariate_stats(list(x.columns), "feature_a")

    def fake_union(*args, **kwargs):
        calls.append("archetype_union")
        return ["feature_b"], _univariate_stats(
            list(x.columns), "feature_b"
        ), {"enabled": True, "source": "test"}

    monkeypatch.setattr(lp, "_univariate_directional_filter", fake_global)
    monkeypatch.setattr(lp, "_univariate_directional_filter_archetype_union", fake_union)

    selected_global, stats_global, diag_global = lp._run_lgbm_univariate_prescreen(
        x,
        y,
        archetype_enabled=False,
        classifier=False,
        random_state=3,
    )
    selected_union, stats_union, diag_union = lp._run_lgbm_univariate_prescreen(
        x,
        y,
        archetype_enabled=True,
        classifier=False,
        random_state=3,
    )

    assert calls == ["global", "archetype_union"]
    assert selected_global == ["feature_a"]
    assert selected_union == ["feature_b"]
    assert diag_global["mode"] == "global"
    assert diag_union["mode"] == "archetype_union"
    assert stats_global["archetype_prescreen_mode"].eq("global").all()
    assert stats_union["archetype_prescreen_mode"].eq("archetype_union").all()


def test_relief_prescreen_switches_between_global_and_archetype_modes(
    monkeypatch,
) -> None:
    x = pd.DataFrame({"feature_a": [0.0, 1.0], "feature_b": [1.0, 0.0]})
    y = np.asarray([0.0, 1.0], dtype=np.float32)
    calls: list[str] = []
    relief_stats = pd.DataFrame(
        {
            "feature": list(x.columns),
            "relief_score": [0.1, 1.0],
            "relief_rescued": [False, True],
        }
    )

    def fake_global(*args, **kwargs):
        calls.append("global")
        return ["feature_b"], relief_stats

    def fake_union(*args, **kwargs):
        calls.append("archetype_union")
        return ["feature_b"], relief_stats, {"enabled": True, "source": "test"}

    monkeypatch.setattr(lp, "_relief_rescue_filter", fake_global)
    monkeypatch.setattr(lp, "_relief_rescue_filter_archetype_union", fake_union)
    monkeypatch.setattr(
        lp,
        "_feature_selection_archetype_labels",
        lambda *args, **kwargs: (np.asarray(["a", "b"], dtype=object), {"source": "test"}),
    )

    _, stats_global, diag_global = lp._run_lgbm_relief_prescreen(
        x,
        y,
        ["feature_a"],
        archetype_enabled=False,
        classifier=False,
        random_state=5,
    )
    _, stats_union, diag_union = lp._run_lgbm_relief_prescreen(
        x,
        y,
        ["feature_a"],
        archetype_enabled=True,
        classifier=False,
        random_state=5,
    )

    assert calls == ["global", "archetype_union"]
    assert diag_global["mode"] == "global"
    assert diag_union["mode"] == "archetype_union"
    assert stats_global["archetype_prescreen_mode"].eq("global").all()
    assert stats_union["archetype_prescreen_mode"].eq("archetype_union").all()


def test_archetype_univariate_scores_economic_target_within_each_slice(
    monkeypatch,
) -> None:
    rows = 1_200
    x = pd.DataFrame(
        {
            "feature_a": np.linspace(-1.0, 1.0, rows, dtype=np.float32),
            "feature_b": np.linspace(1.0, -1.0, rows, dtype=np.float32),
        }
    )
    y = np.concatenate(
        [np.linspace(-0.5, 0.5, 600), np.linspace(2.0, 3.0, 600)]
    ).astype(np.float32)
    labels = np.array(["arch_a"] * 600 + ["arch_b"] * 600, dtype=object)
    calls: list[tuple[pd.DataFrame, np.ndarray, bool]] = []

    monkeypatch.setattr(
        lp,
        "_feature_selection_archetype_labels",
        lambda *args, **kwargs: (labels, {"source": "test", "usable": True}),
    )

    def fake_filter(frame, target, *, classifier, **kwargs):
        calls.append((frame.copy(), np.asarray(target).copy(), bool(classifier)))
        selected = "feature_a" if len(calls) == 1 else "feature_b"
        return [selected], _univariate_stats(list(frame.columns), selected)

    monkeypatch.setattr(lp, "_univariate_directional_filter", fake_filter)
    selected, stats, diag = lp._univariate_directional_filter_archetype_union(
        x,
        y,
        classifier=False,
        random_state=7,
    )

    assert len(calls) == 2
    assert all(len(frame) == 600 for frame, _, _ in calls)
    np.testing.assert_allclose(calls[0][1], y[:600])
    np.testing.assert_allclose(calls[1][1], y[600:])
    assert all(classifier is False for _, _, classifier in calls)
    assert set(selected) == {"feature_a", "feature_b"}
    assert stats["archetype_pass_count"].sum() == 2
    assert diag["archetype_union_features"] == 2


def test_archetype_relief_scores_economic_target_within_each_slice(monkeypatch) -> None:
    rows = 1_200
    x = pd.DataFrame(
        {
            "feature_a": np.linspace(-1.0, 1.0, rows, dtype=np.float32),
            "feature_b": np.linspace(1.0, -1.0, rows, dtype=np.float32),
        }
    )
    y = np.concatenate(
        [np.linspace(-0.5, 0.5, 600), np.linspace(2.0, 3.0, 600)]
    ).astype(np.float32)
    labels = np.array(["arch_a"] * 600 + ["arch_b"] * 600, dtype=object)
    calls: list[tuple[int, np.ndarray, bool]] = []

    def fake_relief(frame, target, uni_features, *, classifier, **kwargs):
        calls.append((len(frame), np.asarray(target).copy(), bool(classifier)))
        selected = "feature_b"
        stats = pd.DataFrame(
            {
                "feature": list(frame.columns),
                "relief_score": [0.1, 1.0],
                "relief_presence": [0.0, 1.0],
                "relief_present_runs": [0, 3],
                "relief_selected": [False, True],
                "relief_rescued": [False, True],
            }
        )
        return [selected], stats

    monkeypatch.setattr(lp, "_relief_rescue_filter", fake_relief)
    rescued, stats, diag = lp._relief_rescue_filter_archetype_union(
        x,
        y,
        ["feature_a"],
        classifier=False,
        random_state=11,
        archetype_labels=labels,
        archetype_diag={"source": "test"},
    )

    assert [row_count for row_count, _, _ in calls] == [600, 600]
    np.testing.assert_allclose(calls[0][1], y[:600])
    np.testing.assert_allclose(calls[1][1], y[600:])
    assert all(classifier is False for _, _, classifier in calls)
    assert rescued == ["feature_b"]
    assert (
        stats.loc[
            stats["feature"].eq("feature_b"), "relief_archetype_selected_count"
        ].iloc[0]
        == 2
    )
    assert diag["relief_rescued_features"] == 1
