"""Held-out evaluation: patient-level outcomes; risk calibration; verified comparators."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score
from . import ACTIONS
from .data import (read_manifest,assert_patient_disjoint,load_array,current_mask,
                   action_mask,available_actions)
from .metrics import dice,summarize_cases,paired_patient_bootstrap
from .policy import PolicyConfig,load_model,plan_case,model_predict

METHODS=['preserve','fixed_boundary','quality_only','regain_mean_proxy','rari_mean',
         'rari_distribution','rari_risk','rari_1step','rari_full',
         'oracle_1step','oracle_2step']


def oracle_history(case,gt,max_steps):
    """Retrospective, not deployable: use only actually executed trajectory masks."""
    paths=[()]
    for key in case['trajectories']:
        path=tuple(key.split('|'))
        if 1<=len(path)<=max_steps and all(a in ACTIONS for a in path):
            paths.append(path)
    return list(max(paths,key=lambda p: dice(current_mask(case,p),gt)))


def case_result(model,case,method,config,side,device,include_calibration=False):
    gt=load_array(case['gt'])>.5
    ini=current_mask(case,())
    if gt.shape!=ini.shape:raise ValueError(f'Ground truth shape error: {case["case_id"]}')
    before=dice(ini,gt)
    if method.startswith('oracle_'):
        history=oracle_history(case,gt,1 if method=='oracle_1step' else config.max_steps)
        referred=False; pquality=None
    elif method=='regain_logged':
        if not case.get('regain_final'):raise ValueError('regain_final not provided')
        history=[]; referred=bool(case.get('regain_referred',False));pquality=None
    else:
        plan=plan_case(model,case,method,config,side,device)
        history=plan['history'];referred=plan['referred'];pquality=plan['predicted_quality']
    final=(load_array(case['regain_final'])>.5) if method=='regain_logged' else current_mask(case,history)
    if final.shape!=gt.shape:raise ValueError('Final/GT mismatch')
    after=dice(final,gt)
    intermediate=[];step_dice=before
    for i in range(len(history)):
        new_d=dice(current_mask(case,history[:i+1]),gt)
        intermediate.append({'step':i+1,'action':history[i],
                             'gain':new_d-step_dice,'harm':bool(new_d-step_dice<-.02)})
        step_dice=new_d
    calls = int(case.get('regain_calls',-1)) if method=='regain_logged' else len(history)
    return {'case_id':case['case_id'],'patient_id':case['patient_id'],
            'cohort':case.get('cohort','unspecified'),'phase':case.get('phase',''),
            'method':method,'initial_dice':before,'final_dice':after,
            'gain':after-before,'referred':referred,'history':'|'.join(history),
            'active_repairs': calls if calls>=0 else 0,
            'calls_unavailable':method=='regain_logged' and calls<0,
            'predicted_quality':pquality,'intermediate_harm':any(a['harm'] for a in intermediate)}


def calculate_action_calibration(model,cases,config,side,device):
    preds=[];truth=[];gain_preds=[];gain_actual=[];lower=[];upper=[]
    for c in cases:
        ref=load_array(c['gt'])>.5
        start_dice=dice(current_mask(c,()),ref)
        out=model_predict(model,c,(),side,device,config.temperature)
        for j,a in enumerate(ACTIONS):
            if a not in available_actions(c,()):continue
            g=dice(action_mask(c,(),a),ref)-start_dice
            preds.append(float(out['harm'][j]));truth.append(float(g<-.02))
            gain_preds.append(float(out['mean'][j]));gain_actual.append(g)
            lower.append(float(out['q10'][j]));upper.append(float(out['q90'][j]))
    if not truth:return {'n_actions':0},[]
    result={'n_actions':len(truth),'harm_events':int(sum(truth)),
            'brier':float(brier_score_loss(truth,preds)),
            'gain_mae':float(np.mean(np.abs(np.array(gain_preds)-np.array(gain_actual)))),
            'q10_q90_empirical_coverage':float(np.mean((np.array(gain_actual)>=np.array(lower)) & (np.array(gain_actual)<=np.array(upper)))),
            'mean_interval_width':float(np.mean(np.array(upper)-np.array(lower)))}
    if len(set(truth))==2:
        result['auprc']=float(average_precision_score(truth,preds))
        result['auroc']=float(roc_auc_score(truth,preds))
    else:
        result['auprc']=None;result['auroc']=None
    rows=[{'predicted_harm_probability':p,'observed_harm':int(y),
           'predicted_gain':gp,'observed_gain':ga,'q10':qlo,'q90':qhi}
          for p,y,gp,ga,qlo,qhi in zip(preds,truth,gain_preds,gain_actual,lower,upper)]
    return result,rows


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def evaluate(checkpoint,manifest,output,split='test',calibration=None,device='cpu',risk_threshold=.1):
    cases=read_manifest(manifest);assert_patient_disjoint(cases)
    selected=[c for c in cases if c.get('split')==split]
    if not selected:raise ValueError(f'No {split} records')
    model,mc=load_model(checkpoint,device)
    cfg=PolicyConfig(max_steps=mc['max_steps'],risk_threshold=risk_threshold)
    calib={}
    if calibration:
        calib=json.loads(Path(calibration).read_text(encoding='utf-8'))
        cfg.temperature=float(calib['temperature'])
        cfg.quality_referral_threshold=float(calib['quality_referral_threshold'])
    methods=list(METHODS)
    if all(c.get('regain_final') for c in selected):methods.append('regain_logged')
    rows=[];by_method={}
    for method in methods:
        by_method[method]=[case_result(model,c,method,cfg,mc['side'],device) for c in selected]
        rows.extend(by_method[method])
    scores={m:summarize_cases(r) for m,r in by_method.items()}
    cohorts=defaultdict(list)
    for r in rows:cohorts[(r['method'],r['cohort'])].append(r)
    cohort_scores={f'{m}::{cohort}':summarize_cases(r) for (m,cohort),r in cohorts.items()}
    phase_groups=defaultdict(list)
    for r in rows:
        if r['phase']: phase_groups[(r['method'],r['phase'])].append(r)
    phase_scores={f'{m}::{phase}':summarize_cases(r) for (m,phase),r in phase_groups.items()}
    cal,cal_rows=calculate_action_calibration(model,selected,cfg,mc['side'],device)
    paired={}
    baseline='regain_logged' if 'regain_logged' in methods else 'regain_mean_proxy'
    paired[f'rari_full_minus_{baseline}']=paired_patient_bootstrap(by_method['rari_full'],by_method[baseline])
    report={'note':'Only selected manifest cases are evaluated. Not a reconstruction of published table values.',
            'split':split,'calibration':calib,'applied_config':vars(cfg),
            'baseline_notice':('Genuine logged ReGain outputs provided by user' if baseline=='regain_logged'
                else 'regain_mean_proxy uses the RARI model mean head; it is NOT original ReGain'),
            'summary':scores,'by_cohort':cohort_scores,'by_phase':phase_scores,
            'risk_gain_calibration_initial_actions':cal,'paired':paired}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'case_results.csv',rows)
    write_csv(output/'action_calibration.csv',cal_rows)
    (output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'output':str(output.resolve()),'methods':methods,
                      'rari_full':scores['rari_full'],'calibration':cal},indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',default='runs/evaluation')
    ap.add_argument('--split',default='test');ap.add_argument('--calibration')
    ap.add_argument('--risk-threshold',type=float,default=.1);ap.add_argument('--device',default='cpu')
    a=ap.parse_args()
    evaluate(a.checkpoint,a.manifest,a.output,a.split,a.calibration,a.device,a.risk_threshold)
if __name__=='__main__':main()
