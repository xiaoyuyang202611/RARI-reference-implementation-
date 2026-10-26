"""Export manuscript-shaped tables from actual evaluation logs; never inject supplied paper numbers."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def summ(g):
    gain=g.final_dice-g.initial_dice
    active=g.active_repairs>0
    harmed=gain<-.02
    return pd.Series({'records':len(g),'patients':g.patient_id.nunique(),
                      'initial_dice':g.initial_dice.mean(),
                      'final_dice':g.final_dice.mean(),
                      'mean_gain':gain.mean(),
                      'material_loss_percent':100*harmed.mean(),
                      'any_negative_percent':100*(gain<0).mean(),
                      'active_repairs':int(active.sum()),
                      'auto_coverage_percent':100*(~g.referred).mean(),
                      'mean_calls':g.active_repairs.mean()})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case-csv',required=True)
    ap.add_argument('--manifest',required=True);ap.add_argument('--output',default='runs/tables')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.case_csv).fillna({'phase':''})
    def grouped_table(frame, keys):
        if frame.empty:
            return pd.DataFrame(columns=keys+list(summ(df.iloc[:1]).index))
        output=[]
        for key, group in frame.groupby(keys, sort=False):
            vals=key if isinstance(key,tuple) else (key,)
            row=dict(zip(keys,vals))
            row.update(summ(group).to_dict())
            output.append(row)
        return pd.DataFrame(output)
    t2=grouped_table(df,['method'])
    t3=grouped_table(df,['method','cohort'])
    t4=grouped_table(df[df.phase.astype(str).str.len()>0],['method','phase'])
    src=pd.read_json(a.manifest,lines=True)
    t1=src.groupby(['cohort','split'],sort=False).agg(records=('case_id','count'),patients=('patient_id','nunique')).reset_index()
    for name,t in [('Table1_cohorts',t1),('Table2_policies',t2),('Table3_cross_domain',t3),('Table4_phase',t4)]:
        t.to_csv(out/f'{name}.csv',index=False)
        print(name,'rows',len(t))
    print('Wrote derived tables to',out.resolve())
if __name__=='__main__':main()
