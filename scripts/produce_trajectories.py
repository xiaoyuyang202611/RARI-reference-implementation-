"""Execute every candidate repair trajectory with a frozen user-supplied adapter.

User adapter: --adapter my_adapter:MedSAMRepairBackend
Its `execute(image,current_mask,action,case)` must return a spatially aligned
mask and MUST NOT access test reference masks. Use `demo` ONLY for smoke tests.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rari import ACTIONS
from rari.data import read_manifest,load_array,current_mask
from rari.backbone_adapter import DemoHeuristicBackend


def create_backend(name):
    if name=='demo':return DemoHeuristicBackend()
    module,cls=name.split(':',1)
    return getattr(importlib.import_module(module),cls)()


def produce(input_manifest,output_dir,adapter):
    cases=read_manifest(input_manifest)
    backend=create_backend(adapter)
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in cases:
        image=load_array(case['image'])
        mask=current_mask(case,())
        folder=out/case['case_id']
        folder.mkdir(parents=True,exist_ok=True)
        trajectories={}
        for a in ACTIONS:
            next_mask=np.asarray(backend.execute(image,mask,a,case),dtype=bool)
            if next_mask.shape!=image.shape:raise ValueError(f'Bad action shape {a} {case["case_id"]}')
            filename=folder/f'{a}.npy';np.save(filename,next_mask.astype('uint8'))
            trajectories[a]=str(filename.relative_to(out))
            for b in ACTIONS:
                next_next=np.asarray(backend.execute(image,next_mask,b,case),dtype=bool)
                if next_next.shape!=image.shape:raise ValueError(f'Bad second step {a}|{b}')
                filename=folder/f'{a}_{b}.npy';np.save(filename,next_next.astype('uint8'))
                trajectories[f'{a}|{b}']=str(filename.relative_to(out))
        # Preserve absolute source paths; processed trajectories are relative to new manifest.
        row={**case,'trajectories':trajectories}
        rows.append(json.dumps(row,ensure_ascii=False))
    dest=out/'manifest_with_trajectories.jsonl'
    dest.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    (out/'backend_metadata.json').write_text(json.dumps({
        'backend':adapter,'WARNING': 'demo backend outputs are NOT experimental medical results' if adapter=='demo' else 'verify checkpoint/source and prompt protocol',
        'all_candidate_actions_executed':True,'max_steps':2},indent=2))
    print('Saved',dest)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',default='prepared_trajectories');ap.add_argument('--adapter',required=True,
        help='module:Class to load or demo (non-medical software test only)')
    a=ap.parse_args();produce(a.manifest,a.output,a.adapter)
if __name__=='__main__':main()
