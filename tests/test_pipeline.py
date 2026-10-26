import numpy as np
import torch
import pytest
from rari.data import build_state,read_manifest,assert_patient_disjoint,TrajectoryDataset
from rari.demo import make_demo
from rari.model import RARI,rari_loss
from rari.metrics import dice
from rari.policy import PolicyConfig,choose_action


def test_dice():
    assert dice(np.zeros((3,3)),np.zeros((3,3)))==1
    assert dice(np.ones((3,3)),np.zeros((3,3)))==0


def test_no_gt_in_state_and_splits(tmp_path):
    path=make_demo(tmp_path/'data',patients=7,side=48)
    cases=read_manifest(path)
    assert assert_patient_disjoint(cases)
    a=build_state(cases[0],(),48)
    # Changing GT cannot alter deployment input (image, masks, prompt, history).
    clone=dict(cases[0]);clone.pop('gt')
    b=build_state(clone,(),48)
    assert torch.equal(a['image'],b['image'])
    assert torch.equal(a['features'],b['features'])
    altered=[dict(c) for c in cases];altered[0]['split']='test'
    altered.append(dict(cases[0]))
    with pytest.raises(ValueError):assert_patient_disjoint(altered)


def test_model_and_policy(tmp_path):
    path=make_demo(tmp_path/'data',patients=7,side=48)
    train=[c for c in read_manifest(path) if c['split']=='train']
    ds=TrajectoryDataset(train,side=48)
    sample=ds[0]
    b={k:v.unsqueeze(0) for k,v in sample.items()}
    model=RARI(dimension=64)
    out=model(b['image'],b['features'])
    loss,_=rari_loss(out,b)
    assert torch.isfinite(loss)
    loss.backward()
    assert out['mean'].shape==(1,4)
    assert torch.all(out['q10']<=out['q50'])
    assert torch.all(out['q50']<=out['q90'])
    assert choose_action({'mean':np.zeros(4),'q10':np.zeros(4),'harm':np.zeros(4),'repairability':np.array(0.5)},
                         ['prompt'],'rari_full',PolicyConfig()) is None
