import re
with open('extreme_price_movements/features.py', 'r') as f:
    content = f.read()

# Replace the broken dispersion lines, formatted by black
dispersion_136 = """    feats["trend_dispersion_1_3_6"] = (
        pd.DataFrame(
            {
                "a": zr_1h.values.ravel(),
                "b": zr_3h.values.ravel(),
                "c": zr_6h.values.ravel(),
            },
            index=c.index,
        )
        .std(axis=1)
        .astype(np.float32)
    )"""

new_dispersion_136 = '    feats["trend_dispersion_1_3_6"] = pd.DataFrame(np.std([zr_1h.values, zr_3h.values, zr_6h.values], axis=0), index=c.index, columns=c.columns).astype(np.float32)'

dispersion_3612 = """    feats["trend_dispersion_3_6_12"] = (
        pd.DataFrame(
            {
                "a": zr_3h.values.ravel(),
                "b": zr_6h.values.ravel(),
                "c": zr_12h.values.ravel(),
            },
            index=c.index,
        )
        .std(axis=1)
        .astype(np.float32)
    )"""

new_dispersion_3612 = '    feats["trend_dispersion_3_6_12"] = pd.DataFrame(np.std([zr_3h.values, zr_6h.values, zr_12h.values], axis=0), index=c.index, columns=c.columns).astype(np.float32)'

content = content.replace(dispersion_136, new_dispersion_136)
content = content.replace(dispersion_3612, new_dispersion_3612)

# Fix the fallback for ret1h
content = content.replace(
    'zr_1h = (\n        feats.get(\n            "ret1h",\n            ff.numba_rolling_sum(feats["ret1h"], 1)\n            if "ret1h" in feats\n            else c.pct_change(1),\n        )\n        / nATR_36h_eps\n    )',
    'zr_1h = feats.get("ret1h", c.pct_change(1)) / nATR_36h_eps'
)

with open('extreme_price_movements/features.py', 'w') as f:
    f.write(content)
