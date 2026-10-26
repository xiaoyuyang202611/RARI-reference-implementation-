"""Validation-only risk temperature and shared automatic coverage triage."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import ACTIONS
from .data import read_manifest, assert_patient_disjoint, TrajectoryDataset, load_array, current_mask, action_mask
from .metrics import dice
from .policy import load_model, model_predict


def fit_temperature(logits,labels):
    logits=np.asarray(logits,dtype=float); labels=np.asarray(labels,dtype=float)
    if not len(logits) or len(np.unique(labels)) < 2: return 1.0
    temps=np.geomspace(.35,5.0,121)
    def nll(t):
        z=np.clip(logits/t,-30,30)
        return float(np.mean(np.logaddexp(0,z)-labels*z))
    return float(min(temps,key=nll))


def calibrate(checkpoint,manifest,outfile,target_coverage=.8,device='cpu'):
    cases=read_manifest(manifest)
    assert_patient_disjoint(cases)
    val=[c for c in cases if c.get('split')=='val']
    if not val:raise ValueError('Need validation cases')
    model,cfg=load_model(checkpoint,device)
    logits=[];labels=[];qualities=[]
    for c in val:
        pred=model_predict(model,c,(),cfg['side'],device)
        qualities.append(float(pred['quality']))
        g=load_array(c['gt'])>.5
        base=dice(current_mask(c,()),g)
        for j,a in enumerate(ACTIONS):
            if a in c['trajectories']:
                logits.append(float(pred['harm_logits'][j]))
                labels.append(float(dice(action_mask(c,(),a),g)-base < -.02))
    temp=fit_temperature(logits,labels)
    # Shared quality triage: achieve target coverage approximately, not guaranteed exactly for ties.
    if not (0 < target_coverage <= 1):raise ValueError('Coverage must be in (0,1]')
    threshold=float(np.quantile(qualities,1-target_coverage)) if target_coverage<1 else 0.0
    achieved=float(np.mean(np.asarray(qualities)>=threshold))
    obj={'temperature':temp,'quality_referral_threshold':threshold,
         'target_coverage':target_coverage,'val_coverage':achieved,
         'val_patients':len({c['patient_id'] for c in val}),'val_actions':len(logits),
         'val_harm_events':int(sum(labels)),'source_checkpoint':str(checkpoint),
         'method':'shared_predicted_quality_triage_and_temperature_on_validation'}
    Path(outfile).parent.mkdir(parents=True,exist_ok=True)
    Path(outfile).write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps(obj,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',default='runs/calibration.json');ap.add_argument('--coverage',type=float,default=.8)
    ap.add_argument('--device',default='cpu');a=ap.parse_args()
    calibrate(a.checkpoint,a.manifest,a.output,a.coverage,a.device)
if __name__=='__main__':main()
