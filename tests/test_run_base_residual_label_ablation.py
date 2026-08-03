import pandas as pd

from scripts.run_base_residual_label_ablation import spread_exclusion_mask


def test_spread_exclusion_uses_inference_symbol_normalization() -> None:
    symbols = pd.Series(["BTC_USD:USD", "ETH/USD:USD", "AAVE_USD:USD"])
    excluded = {"BTC/USD:USD", "AAVE/USD:USD"}
    assert spread_exclusion_mask(symbols, excluded).tolist() == [
        True,
        False,
        True,
    ]
