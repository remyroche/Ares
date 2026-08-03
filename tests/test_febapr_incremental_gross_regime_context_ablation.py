import pandas as pd

from scripts.run_febapr_incremental_gross_regime_context_ablation import run_side, top


def test_top_is_one_pooled_global_book_not_per_timestamp():
    frame = pd.DataFrame({"candidate_id": ["a", "b", "c", "d"], "score": [4.0, 3.0, 2.0, 1.0]})
    assert top(frame, "score")["candidate_id"].tolist() == ["a"]


def test_fold_purge_and_april_holdout_isolation():
    ts = pd.date_range("2025-03-01", periods=25, freq="D", tz="UTC")
    train = pd.DataFrame(
        {
            "candidate_id": [f"m{i}" for i in range(25)],
            "side_name": ["long"] * 25,
            "__symbol__": ["BTC_USD:USD"] * 25,
            "__ts__": ts,
            "execution_label_end_utc": ts + pd.Timedelta(hours=12),
            "execution_net_ev_12h": list(range(25)),
            "f": list(range(25)),
        }
    )
    test = train.iloc[:3].copy()
    test["candidate_id"] = ["a", "b", "c"]
    test["__ts__"] = pd.date_range("2025-04-01", periods=3, freq="D", tz="UTC")
    _, first, contract = run_side(train, test, ["f"])
    altered = test.copy()
    altered["execution_net_ev_12h"] = [-1e9, 1e9, -1e9]
    _, second, _ = run_side(train, altered, ["f"])
    assert contract["purge"].endswith("2025-03-21T00:00:00Z")
    assert first["raw_score"].tolist() == second["raw_score"].tolist()
