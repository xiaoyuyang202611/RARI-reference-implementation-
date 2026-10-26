"""Predict an action sequence on a single case; no GT is required."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import read_manifest,current_mask
from .policy import load_model,PolicyConfig,plan_case


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True)
    ap.add_argument('--case-id',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--calibration');ap.add_argument('--device',default='cpu')
    ap.add_argument('--method',default='rari_full');ap.add_argument('--output',default='runs/predicted_mask.npy')
    a=ap.parse_args();cases=read_manifest(a.manifest)
    matches=[c for c in cases if c['case_id']==a.case_id]
    if len(matches)!=1:raise ValueError(f'Expected exactly one {a.case_id}, got {len(matches)}')
    case=matches[0]
    model,mc=load_model(a.checkpoint,a.device)
    cfg=PolicyConfig(max_steps=mc['max_steps'])
    if a.calibration:
        d=json.loads(Path(a.calibration).read_text(encoding='utf-8'))
        cfg.quality_referral_threshold=float(d['quality_referral_threshold'])
        cfg.temperature=float(d['temperature'])
    result=plan_case(model,case,a.method,cfg,side=mc['side'],device=a.device)
    if not result['referred']:
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        np.save(out,current_mask(case,result['history']).astype('uint8'))
        result['saved_mask']=str(out.resolve())
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
