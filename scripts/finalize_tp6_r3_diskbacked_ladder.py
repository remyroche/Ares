#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);a=p.parse_args();parts=sorted((a.fit/'prediction_parts').glob('*.parquet'));x=pd.concat([pd.read_parquet(z) for z in parts],ignore_index=True);x.to_parquet(a.fit/'predictions.parquet',index=False);rows=[]
 for f in (.01,.05,.10):
  z=x.nlargest(int(np.ceil(len(x)*f)),'score_bps');rows.append({'top_fraction':f,'rows':len(z),'gross_bps':z.gross_bps.mean(),'net_bps':z.net_bps.mean()})
 pd.DataFrame(rows).to_parquet(a.fit/'results.parquet',index=False);state=json.loads((a.fit/'fit_state.json').read_text());(a.fit/'manifest.json').write_text(json.dumps({'source_matrix':state['source_matrix'],'selected_features':state['selected_features'],'selection':'strict pre-March only covariance ranking; configured base pool only','test_rows':len(x),'metrics':rows},indent=2,default=lambda v:float(v) if hasattr(v,'item') else str(v)));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
