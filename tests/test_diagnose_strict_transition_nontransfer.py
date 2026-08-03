import pandas as pd
from scripts.diagnose_strict_transition_nontransfer import zero_positive_status
def test_zero_positive_requires_materialized_active_labels():
 d=pd.DataFrame({'target__transition_active':[0,0]});a=pd.Series(pd.to_datetime(['2026-01-01','2026-01-01'],utc=True),index=d.index);close=pd.Timestamp('2026-01-02',tz='UTC');assert zero_positive_status(d,a,close)=='GENUINE_ZERO_ACTIVE_LABELS_IN_MATERIALIZED_CATALOGUE'
 d.loc[1,'target__transition_active']=None;assert zero_positive_status(d,a,close)=='UNRESOLVED_OR_MISSING_ACTIVE_LABEL'
