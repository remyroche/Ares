import numpy as np
import pandas as pd

from extreme_price_movements import features


def test_hvn_uses_single_process_fallback_when_process_pool_unavailable(monkeypatch):
    idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    cols = ["AAA/USDT", "BBB/USDT"]

    def _frame(base):
        data = np.tile(np.asarray(base, dtype=np.float32).reshape(-1, 1), (1, len(cols)))
        return pd.DataFrame(data, index=idx, columns=cols)

    o = _frame([1.0, 2.0, 3.0, 4.0])
    h = _frame([1.5, 2.5, 3.5, 4.5])
    l = _frame([0.5, 1.5, 2.5, 3.5])
    c = _frame([1.2, 2.2, 3.2, 4.2])
    v = _frame([10.0, 11.0, 12.0, 13.0])

    calls = []

    def fake_compute_col(col, o_col, h_col, l_col, c_col, v_col):
        calls.append(col)
        return col, pd.DataFrame(
            {
                "poc_touchrate": np.full(len(c_col), 0.25, dtype=np.float32),
                "profile_entropy": np.full(len(c_col), 0.75, dtype=np.float32),
            },
            index=c_col.index,
        )

    monkeypatch.setattr(features.os, "sysconf", lambda name: (_ for _ in ()).throw(PermissionError("blocked")))

    results = features._compute_hvn_feature_frames(
        o,
        h,
        l,
        c,
        v,
        ["poc_touchrate", "profile_entropy"],
        compute_col_fn=fake_compute_col,
    )

    assert calls == cols
    assert set(results.keys()) == {"poc_touchrate", "profile_entropy"}
    for key, expected in [("poc_touchrate", 0.25), ("profile_entropy", 0.75)]:
        df = results[key]
        assert list(df.columns) == cols
        assert df.shape == (4, 2)
        assert np.allclose(df.values, expected)
