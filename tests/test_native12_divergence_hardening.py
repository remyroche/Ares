import pandas as pd
from scripts.harden_native12_divergence_artifacts import identity_hash

def test_identity_hash_is_order_independent_and_sensitive():
    assert identity_hash(pd.Series(['b','a'])) == identity_hash(pd.Series(['a','b']))
    assert identity_hash(pd.Series(['a','b'])) != identity_hash(pd.Series(['a','c']))
