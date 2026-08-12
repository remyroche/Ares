"""Predeclared 0--4 relevance-grade grid for LambdaRank query screening."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _touch(frame: pd.DataFrame, prefix: str, threshold: float) -> np.ndarray:
    name=(f'{threshold:g}').replace('.','p')
    return pd.to_numeric(frame[f'{prefix}_touch_{name}atr_minute'],errors='coerce').fillna(-1).to_numpy(int)


def _first(fav: np.ndarray, adv: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    positive=(fav>=0)&((adv<0)|(fav<adv)); negative=(adv>=0)&((fav<0)|(adv<=fav))
    return positive,negative,~positive&~negative


def _base(frame: pd.DataFrame) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    valid=frame.label_valid.fillna(False).to_numpy(bool)
    gross=pd.to_numeric(frame.terminal_gross_bps,errors='coerce').to_numpy(float)
    net=pd.to_numeric(frame.terminal_net_bps,errors='coerce').to_numpy(float)
    grade=np.zeros(len(frame),np.int8)
    # Grade 0 is reserved for gross-negative paths.  A valid gross-positive
    # path begins at grade 1 and is promoted only after it clears the explicit
    # path/economic thresholds below.  This includes timeouts and avoids
    # encoding a positive but unresolved path as a negative outcome.
    grade[valid&(gross>0)]=1
    return grade,valid,gross


def atr_spacing_grade(frame: pd.DataFrame, spacing: float) -> np.ndarray:
    """Symmetric first-touch grades at 1/1.5/2 ATR spacing, H12 only."""
    grade,valid,gross=_base(frame)
    # Grade thresholds are spacing, 2*spacing and 3*spacing.  The adverse
    # boundary is the same magnitude and therefore does not impose a limit
    # other than the declared competing first touch.
    for level, threshold, floor in ((2,spacing,100.),(3,2*spacing,150.),(4,3*spacing,200.)):
        fav=_touch(frame,'fav',threshold); adv=_touch(frame,'adv',threshold); positive,_,timeout=_first(fav,adv)
        grade[valid&positive&(gross>=floor)]=level
        grade[valid&timeout&(gross>0)&(gross<floor)&(grade==0)]=1
    return grade


def absolute_spacing_grade(frame: pd.DataFrame, spacing_pct: float) -> np.ndarray:
    """Absolute 1/1.5/2% spacing using exact H12 favourable/adverse extrema."""
    grade,valid,gross=_base(frame)
    entry=pd.to_numeric(frame.entry_price,errors='coerce').to_numpy(float)
    atr_bps=pd.to_numeric(frame.atr_bps,errors='coerce').to_numpy(float)
    mfe=pd.to_numeric(frame.mfe_atr,errors='coerce').to_numpy(float)*atr_bps
    mae=pd.to_numeric(frame.mae_atr,errors='coerce').to_numpy(float)*atr_bps
    # Boundaries are in bps and compare the H12 path extrema; terminal gross
    # remains the economic guardrail for assigning a high relevance grade.
    for level, threshold, floor in ((2,spacing_pct*100.,100.),(3,2*spacing_pct*100.,150.),(4,3*spacing_pct*100.,200.)):
        positive=(mfe>=threshold)&((mae<threshold)|(mfe>=mae))
        grade[valid&positive&(gross>=floor)]=level
    return grade


def triple_barrier_grade(frame: pd.DataFrame, *, lower_atr: float, upper_atr: float) -> np.ndarray:
    """Exact TP/SL first-touch grade with adverse ties and H12 timeout."""
    if lower_atr>upper_atr: raise ValueError('lower ATR boundary cannot exceed upper ATR boundary')
    grade,valid,gross=_base(frame)
    fav=_touch(frame,'fav',upper_atr); adv=_touch(frame,'adv',lower_atr); positive,negative,timeout=_first(fav,adv)
    grade[valid&timeout&(gross>0)&(grade==0)]=1
    grade[valid&negative]=0
    for level,floor in ((2,100.),(3,150.),(4,200.)):
        grade[valid&positive&(gross>=floor)]=level
    return grade


def grade_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the complete, small requested grid without invalid supervision."""
    out=frame.copy()
    for spacing in (1.,1.5,2.):
        out[f'grade_atr_spacing_{str(spacing).replace(".","p")}']=atr_spacing_grade(out,spacing)
        out[f'grade_absolute_spacing_{str(spacing).replace(".","p")}pct']=absolute_spacing_grade(out,spacing)
    for lower in (2.,3.,4.):
        for upper in (2.,3.,4.,5.,6.):
            if lower<=upper: out[f'grade_tb_sl{lower:g}_tp{upper:g}']=triple_barrier_grade(out,lower_atr=lower,upper_atr=upper)
    return out
