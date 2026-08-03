import pandas as pd
from scripts.attribute_native12_execution_ev_failures import select

def test_global_top10_count_is_ceil_and_single_book():
    x=pd.DataFrame({'score':range(11)})
    assert int(select(x,'score').sum()) == 2
