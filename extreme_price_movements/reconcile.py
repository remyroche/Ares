from __future__ import annotations

import pandas as pd


def reconcile_ledgers(ledger_a, ledger_b, key_cols=("asset", "t_entry", "t_exit", "side")):
    a = pd.DataFrame(list(ledger_a))
    b = pd.DataFrame(list(ledger_b))
    return a.merge(b, on=list(key_cols), suffixes=("_opt", "_sim"), how="outer", indicator=True)
