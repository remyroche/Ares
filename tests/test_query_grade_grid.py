import numpy as np
import pandas as pd

from extreme_price_movements.query_grade_grid import grade_columns, triple_barrier_grade


def _frame():
    x=pd.DataFrame({"label_valid":[True,True,True,False],"entry_price":[100.]*4,"atr_bps":[100.]*4,
                    "terminal_gross_bps":[-50.,50.,250.,250.],"terminal_net_bps":[-150.,-50.,150.,150.],
                    "mfe_atr":[.5,1.,3.,3.],"mae_atr":[2.,.5,.5,.5]})
    for t in (1.,1.5,2.,3.,4.,4.5,5.,6.):
        n=(f'{t:g}').replace('.','p'); x[f'fav_touch_{n}atr_minute']=[-1,-1,2,2]; x[f'adv_touch_{n}atr_minute']=[2,-1,-1,-1]
    return x


def test_grade_grid_preserves_invalid_and_economic_guardrails():
    x=grade_columns(_frame())
    grades=[c for c in x if c.startswith('grade_')]
    assert len(grades)==18
    assert (x.loc[3,grades]==0).all()
    assert triple_barrier_grade(x,lower_atr=2.,upper_atr=2.).tolist()==[0,1,4,0]
