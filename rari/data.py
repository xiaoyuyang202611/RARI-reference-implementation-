"""Reference mask only used offline for training labels and post-hoc metrics.

Manifest path fields are relative to the .jsonl file. NumPy volumes are (Z,Y,X);
NIfTI (X,Y,Z) is transposed on load. No implicit resampling or co-registration.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import ACTIONS
from .metrics import dice

FEATURE_DIM = 17


def load_array(path):
    path = Path(path)
    if str(path).lower().endswith('.npy'):
        arr = np.load(path, allow_pickle=False)
    elif str(path).lower().endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        arr = np.asarray(nib.load(str(path)).dataobj)
        if arr.ndim == 3:
            arr = np.moveaxis(arr, -1, 0)  # Z,Y,X; same rule for all files
    else:
        raise ValueError(f'Unsupported file type: {path}')
    if arr.ndim not in (2, 3):
        raise ValueError(f'Only 2D or 3D supported, got {arr.shape} in {path}')
    if not np.isfinite(arr).all():
        raise ValueError(f'NaN/Inf in {path}')
    return np.asarray(arr)


def read_manifest(path, split=None, allow_empty=False):
    path = Path(path).resolve()
    cases = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if split is not None and c.get('split') != split:
            continue
        for k in ('patient_id', 'case_id', 'image', 'initial', 'trajectories'):
            if k not in c:
                raise ValueError(f'Missing key {k} in {c}')
        for key in ('image', 'initial', 'prob', 'gt', 'regain_final'):
            if c.get(key):
                c[key] = str((path.parent / c[key]).resolve())
        c['trajectories'] = {k: str((path.parent / v).resolve()) for k, v in c['trajectories'].items()}
        cases.append(c)
    if not cases and not allow_empty:
        raise ValueError(f'No records for split {split!r}')
    return cases


def assert_patient_disjoint(cases):
    assignments = {}
    for c in cases:
        pid, split = c['patient_id'], c.get('split')
        if pid in assignments and assignments[pid] != split:
            raise ValueError(f'PATIENT DATA LEAKAGE: {pid} in {assignments[pid]} and {split}')
        assignments[pid] = split
    return True


def history_key(history):
    return '|'.join(history)


def available_actions(case, history):
    return [a for a in ACTIONS if history_key(list(history) + [a]) in case['trajectories']]


def current_mask(case, history):
    key = history_key(history)
    return load_array(case['initial'] if not history else case['trajectories'][key]) > 0.5


def action_mask(case, history, action):
    k = history_key(list(history) + [action])
    if k not in case['trajectories']:
        raise KeyError(f'Not executed / missing action path {k} for {case["case_id"]}')
    return load_array(case['trajectories'][k]) > 0.5


def _shape_hw(arr):
    return tuple(arr.shape[-2:])


def _resize(arr, side=96, kind='bilinear'):
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None, None]
    if kind == 'nearest':
        return F.interpolate(t, size=(side,side), mode='nearest')[0,0]
    return F.interpolate(t, size=(side,side), mode='bilinear', align_corners=False)[0,0]


def build_state(case, history=(), side=96, max_steps=2):
    """No GT data accessed here; valid at inference. Image/probability/prompt/history only."""
    image = load_array(case['image']).astype(np.float32)
    mask = current_mask(case, history)
    if image.shape != mask.shape:
        raise ValueError(f'Image/mask spatial mismatch for {case["case_id"]}: {image.shape} {mask.shape}')
    if image.ndim == 2:
        slices = [image, image, image]
        mid_mask = mask
    else:
        mid = int(case.get('slice_index', image.shape[0]//2))
        if not 0 <= mid < image.shape[0]:
            raise ValueError(f'Invalid slice_index {mid}')
        slices = [image[max(mid-1,0)],image[mid],image[min(mid+1,image.shape[0]-1)]]
        mid_mask = mask[mid]
    lo, hi = np.percentile(image, [1,99])
    scale = max(float(hi-lo),1e-6)
    norm = [np.clip((s-lo)/scale,0,1) for s in slices]
    prob = load_array(case['prob']).astype(np.float32) if (case.get('prob') and not history) else mask.astype(np.float32)
    if prob.shape != mask.shape:
        raise ValueError('Probability map must match current mask shape')
    prob2 = prob if prob.ndim == 2 else prob[mid]
    prompt = np.zeros(_shape_hw(mask),dtype=np.float32)
    box = case.get('prompt_box')
    if box is not None:
        x0,y0,x1,y1=map(int,box)
        h,w=prompt.shape
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):
            raise ValueError(f'Invalid prompt_box {box} on {prompt.shape}')
        prompt[y0:y1,x0:x1]=1
    channels=norm+[mid_mask.astype(np.float32),prob2,prompt]
    tensor=torch.stack([_resize(c,side,'nearest' if i in (3,5) else 'bilinear') for i,c in enumerate(channels)],dim=0)
    p=np.clip(prob2,1e-5,1-1e-5)
    binary=mid_mask
    m_area=int(binary.sum())
    arr=norm[1]
    inner=arr[binary] if m_area else np.asarray([0],dtype=float)
    outer=arr[~binary] if (~binary).any() else np.asarray([0],dtype=float)
    bd = (np.abs(np.diff(binary.astype(float),axis=0)).sum() + np.abs(np.diff(binary.astype(float),axis=1)).sum()) / max(binary.size,1)
    last_onehot = [float(bool(history) and history[-1]==a) for a in ACTIONS]
    features=np.asarray([
        float(binary.mean()),float(bd),float(inner.mean()),float(inner.std()),
        float(outer.mean()),float(outer.std()),
        float((-p*np.log(p)-(1-p)*np.log(1-p)).mean()),
        float((np.abs(p-.5)*2).mean()),float(prompt.mean()),
        float(len(history)/max_steps),float((max_steps-len(history))/max_steps),
        *last_onehot,float(np.mean(arr)),float(np.std(arr))
    ],dtype=np.float32)
    assert len(features)==FEATURE_DIM, (len(features), FEATURE_DIM)
    return {'image':tensor,'features':torch.from_numpy(features)}


class TrajectoryDataset(Dataset):
    def __init__(self, cases, side=96, max_steps=2):
        self.cases=cases; self.side=side; self.max_steps=max_steps
        self.samples=[]
        for ci,c in enumerate(cases):
            if not c.get('gt'):
                raise ValueError('Training/evaluation requires GT for offline labels')
            histories=[()]
            if max_steps>1:
                histories.extend((a,) for a in ACTIONS if a in c['trajectories'])
            for history in histories:
                if available_actions(c,history):
                    self.samples.append((ci,history))
        if not self.samples:
            raise ValueError('No candidate actions found: build actual action trajectories first')

    def __len__(self): return len(self.samples)

    def __getitem__(self,idx):
        ci,history=self.samples[idx]; case=self.cases[ci]
        state=build_state(case,history,self.side,self.max_steps)
        gt=load_array(case['gt'])>0.5
        curr=current_mask(case,history)
        if curr.shape!=gt.shape: raise ValueError(f'Shape mismatch for {case["case_id"]}')
        start_dice=dice(curr,gt)
        gain=np.zeros((len(ACTIONS),),dtype=np.float32)
        valid=np.zeros(len(ACTIONS),dtype=np.float32)
        for j,a in enumerate(ACTIONS):
            if a in available_actions(case,history):
                gain[j]=dice(action_mask(case,history,a),gt)-start_dice
                valid[j]=1.
        return {**state,
                'gain':torch.tensor(gain),
                'harm':torch.tensor((gain < -.02).astype(np.float32)),
                'valid':torch.tensor(valid),
                'repairability':torch.tensor(max(0,float(gain.max())),dtype=torch.float32),
                'quality':torch.tensor(start_dice,dtype=torch.float32)}
