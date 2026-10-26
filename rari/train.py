"""Patient-disjoint offline supervision from executed actions and reference masks."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import read_manifest, assert_patient_disjoint, TrajectoryDataset
from .model import RARI, rari_loss


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def train(args):
    seed_everything(args.seed)
    all_cases=read_manifest(args.manifest)
    assert_patient_disjoint(all_cases)
    tr=TrajectoryDataset([c for c in all_cases if c.get('split')=='train'],side=args.side,max_steps=args.max_steps)
    va=TrajectoryDataset([c for c in all_cases if c.get('split')=='val'],side=args.side,max_steps=args.max_steps)
    tl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
    vl=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    device=torch.device('cuda' if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    model=RARI(dimension=args.dimension,use_repairability=not args.no_repairability).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float('inf'); patience=0
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        metrics=[]
        for phase,loader in (('train',tl),('val',vl)):
            model.train(phase=='train')
            losses=[]
            for b in loader:
                b={k:v.to(device) for k,v in b.items()}
                with torch.set_grad_enabled(phase=='train'):
                    pred=model(b['image'],b['features'])
                    loss,_=rari_loss(pred,b,use_repairability=not args.no_repairability)
                    if phase=='train':
                        opt.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                        opt.step()
                losses.append(float(loss.item()))
            avg=float(np.mean(losses));metrics.append(avg)
        print(f'epoch={epoch:03d} train={metrics[0]:.5f} val={metrics[1]:.5f}',flush=True)
        if metrics[1]<best-1e-7:
            best=metrics[1];patience=0
            torch.save({'state_dict':model.state_dict(),
                        'config':{'dimension':args.dimension,'side':args.side,
                                  'max_steps':args.max_steps,'use_repairability':not args.no_repairability},
                        'best_val_loss':best,'epoch':epoch},output)
        else:
            patience+=1
            if patience>=args.patience:break
    print(json.dumps({'checkpoint':str(output.resolve()),'best_val_loss':best,'train_states':len(tr),
                      'val_states':len(va),'device':str(device)},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True);ap.add_argument('--output',default='runs/rari.pt')
    ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--dimension',type=int,default=512);ap.add_argument('--side',type=int,default=96)
    ap.add_argument('--max-steps',type=int,default=2);ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--patience',type=int,default=8);ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--device',default='auto');ap.add_argument('--no-repairability',action='store_true')
    train(ap.parse_args())
if __name__=='__main__':main()
