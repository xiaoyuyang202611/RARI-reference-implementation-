"""Pre-submission manifest audit; execute from project root as python scripts/check_manifest.py ..."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from collections import Counter
from pathlib import Path
from rari.data import read_manifest,assert_patient_disjoint,load_array,current_mask,action_mask
from rari import ACTIONS


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--strict-two-step',action='store_true');a=p.parse_args()
    cases=read_manifest(a.manifest);assert_patient_disjoint(cases)
    phases={}
    for c in cases:
        image=load_array(c['image']);initial=current_mask(c,())
        if image.shape!=initial.shape:raise ValueError(f'Bad image/initial geometry: {c["case_id"]}')
        if c.get('gt') and load_array(c['gt']).shape!=image.shape:
            raise ValueError(f'Bad GT geometry: {c["case_id"]}')
        for key,path in c['trajectories'].items():
            if load_array(path).shape!=image.shape:raise ValueError(f'Bad trajectory {key}: {c["case_id"]}')
            path_actions=key.split('|')
            if any(x not in ACTIONS for x in path_actions) or len(path_actions)>2:
                raise ValueError(f'Invalid action path {key}')
        if a.strict_two_step:
            missing=[f'{x}|{y}' for x in ACTIONS for y in ACTIONS if f'{x}|{y}' not in c['trajectories']]
            missing += [x for x in ACTIONS if x not in c['trajectories']]
            if missing:raise ValueError(f'Incomplete counterfactual bank for {c["case_id"]}: {missing}')
        key=(c['patient_id'],c.get('phase',''))
        if key in phases:raise ValueError(f'Duplicate patient/phase {key}; review scan identities')
        phases[key]=c['case_id']
    print(json.dumps({'verified_records':len(cases),'unique_patients':len({c['patient_id'] for c in cases}),
                      'splits':dict(Counter(c.get('split') for c in cases)),
                      'all_paths_exist':True,'spatial_shapes_match':True},indent=2))
if __name__=='__main__':main()
