"""RARI and controlled comparator decision rules; no reference mask is accessed."""
from dataclasses import dataclass
import numpy as np
import torch
from . import ACTIONS
from .data import build_state, available_actions

@dataclass
class PolicyConfig:
    risk_threshold: float = 0.10
    gain_threshold: float = 0.002
    cost_per_action: float = 0.005  # normalized utility units, not milliseconds
    risk_weight: float = 0.25
    max_steps: int = 2
    quality_referral_threshold: float = 0.0  # derived on validation if coverage matching
    temperature: float = 1.0
    repairability_min_gain: float = 0.02  # reconstruction-specific gating choice


def model_predict(model,case,history,side=96,device='cpu',temperature=1.0):
    x=build_state(case,history,side=side)
    model.eval()
    with torch.no_grad():
        out=model(x['image'][None].to(device),x['features'][None].to(device))
    values={k:v[0].cpu().numpy() for k,v in out.items()}
    values['harm']=1/(1+np.exp(-np.clip(values['harm_logits']/max(temperature,1e-3),-30,30)))
    return values


def choose_action(pred,feasible,method,config,use_repairability=True):
    """No-op always available. 'refer' is controlled by initial quality triage."""
    if method=='preserve' or method=='quality_only':return None
    if (method in ('rari_full','rari_1step') and use_repairability and
            float(pred['repairability']) < config.repairability_min_gain):
        return None
    if method=='fixed_boundary':return 'boundary' if 'boundary' in feasible else None
    scores={}
    for a in feasible:
        j=ACTIONS.index(a)
        risk=float(pred['harm'][j])
        mean=float(pred['mean'][j])
        q10=float(pred['q10'][j])
        if method in ('rari_full','rari_1step','rari_risk') and risk > config.risk_threshold:
            continue
        if method=='rari_full' or method=='rari_1step':
            utility=.75*mean+.25*q10-config.risk_weight*risk-config.cost_per_action
        elif method=='rari_risk': utility=mean-config.risk_weight*risk-config.cost_per_action
        elif method=='rari_distribution': utility=.5*mean+.5*q10-config.cost_per_action
        else: # regain_mean_proxy, rari_mean (NOT authors' original ReGain)
            utility=mean-config.cost_per_action
        scores[a]=utility
    if not scores:return None
    best=max(scores,key=scores.get)
    return best if scores[best]>config.gain_threshold else None


def plan_case(model,case,method='rari_full',cfg=None,side=96,device='cpu',return_predictions=False):
    """Route using inference-visible state; action outputs must already exist in trajectories.
    Does NOT peek at ground truth. The logged trajectories are executed-action data.
    """
    cfg=cfg or PolicyConfig()
    initial=model_predict(model,case,(),side,device,cfg.temperature)
    q=float(initial['quality'])
    if q < cfg.quality_referral_threshold:
        res={'history':[],'referred':True,'reason':'quality_triage','predicted_quality':q,'calls':0}
        if return_predictions:res['initial_predictions']=initial
        return res
    if method=='fixed_boundary' or method=='preserve' or method=='quality_only':steps=1
    elif method=='rari_1step':steps=1
    else:steps=cfg.max_steps
    hist=[];pred=initial
    for step in range(steps):
        available=available_actions(case,hist)
        a=choose_action(pred,available,method,cfg,
                        use_repairability=bool(getattr(model,'use_repairability',True)))
        if a is None:break
        hist.append(a)
        if step+1<steps and available_actions(case,hist):
            pred=model_predict(model,case,hist,side,device,cfg.temperature)
    result={'history':hist,'referred':False,'reason':'auto','predicted_quality':q,'calls':len(hist)}
    if return_predictions:result['initial_predictions']=initial
    return result


def load_model(checkpoint,device='cpu'):
    from .model import RARI
    try:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    except TypeError: # older PyTorch, only trusted checkpoint files
        saved=torch.load(checkpoint,map_location='cpu')
    model=RARI(dimension=saved['config']['dimension'],
               use_repairability=saved['config']['use_repairability'])
    model.load_state_dict(saved['state_dict'])
    model.to(device).eval()
    return model,saved['config']
