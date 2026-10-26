"""Generate explicitly synthetic phantom data for a reproducible software SMOKE TEST.
Never report resulting numbers as medical experimental results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from . import ACTIONS
from .actions_demo import demo_action


def make_demo(root,patients=16,seed=2026,side=64):
    if patients<6:raise ValueError('Generate at least 6 phantoms for train/val/test')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[:side,:side]
    lines=[]
    for i in range(patients):
        split='train' if i<round(patients*.60) else ('val' if i<round(patients*.80) else 'test')
        cdir=root/f'synthetic_{i:03d}';cdir.mkdir(exist_ok=True)
        cx=side/2+rng.uniform(-8,8);cy=side/2+rng.uniform(-8,8)
        rx=rng.uniform(8,14);ry=rng.uniform(8,14)
        gt=(((x-cx)/rx)**2+((y-cy)/ry)**2)<1
        # kept explicit to prevent any reference label from entering the model feature builder
        image=(ndi.gaussian_filter(gt.astype(np.float32),1.5)*.8 +
               rng.normal(.11,.08,(side,side))).astype(np.float32)
        px=cx+rng.integers(-4,5);py=cy+rng.integers(-4,5)
        initial=(((x-px)/(rx*rng.uniform(.78,1.10)))**2+
                 ((y-py)/(ry*rng.uniform(.78,1.10)))**2)<1
        np.save(cdir/'image.npy',image)
        np.save(cdir/'gt.npy',gt.astype('uint8'))
        np.save(cdir/'initial.npy',initial.astype('uint8'))
        paths={}
        for a in ACTIONS:
            ma=demo_action(image,initial,a)
            np.save(cdir/f'{a}.npy',ma.astype('uint8'))
            paths[a]=str((cdir/f'{a}.npy').relative_to(root))
            for b in ACTIONS:
                mb=demo_action(image,ma,b)
                np.save(cdir/f'{a}_{b}.npy',mb.astype('uint8'))
                paths[f'{a}|{b}']=str((cdir/f'{a}_{b}.npy').relative_to(root))
        case={'patient_id':f'SYNTHETIC_{i:03d}','case_id':f'SYNTHETIC_{i:03d}',
              'cohort':'SYNTHETIC_PHANTOM_NOT_MEDICAL', 'split':split,
              'image':str((cdir/'image.npy').relative_to(root)),
              'initial':str((cdir/'initial.npy').relative_to(root)),
              'gt':str((cdir/'gt.npy').relative_to(root)),
              'prompt_box':[max(0,int(cx-rx)),max(0,int(cy-ry)),
                            min(side,int(cx+rx)),min(side,int(cy+ry))],
              'trajectories':paths}
        lines.append(json.dumps(case))
    path=root/'manifest_demo.jsonl';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'README_DEMO.txt').write_text('ONLY SYNTHETIC SOFTWARE TEST DATA. NO PATIENT DATA. NEVER USE FOR PAPER RESULTS.\n')
    print('Generated synthetic-only manifest:',path)
    return path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='demo_data')
    ap.add_argument('--patients',type=int,default=16);ap.add_argument('--side',type=int,default=64)
    ap.add_argument('--seed',type=int,default=2026);a=ap.parse_args()
    make_demo(a.output,a.patients,a.seed,a.side)
if __name__=='__main__':main()
