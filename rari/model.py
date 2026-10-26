"""Action-conditioned RARI heads. Architecture is a documented reconstruction."""
import torch
from torch import nn
from torch.nn import functional as F
from . import ACTIONS
from .data import FEATURE_DIM


class ConvBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(inp,out,3,stride=2,padding=1,bias=False),
                                  nn.BatchNorm2d(out),nn.SiLU(inplace=True),
                                  nn.Conv2d(out,out,3,padding=1,bias=False),
                                  nn.BatchNorm2d(out),nn.SiLU(inplace=True))
    def forward(self,x): return self.layers(x)


class RARI(nn.Module):
    def __init__(self, dimension=512, action_embedding=32, use_repairability=True):
        super().__init__()
        self.use_repairability=use_repairability
        self.cnn=nn.Sequential(ConvBlock(6,32),ConvBlock(32,64),ConvBlock(64,128),
                               ConvBlock(128,256),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.tab=nn.Sequential(nn.Linear(FEATURE_DIM,64),nn.LayerNorm(64),nn.SiLU())
        self.shared=nn.Sequential(nn.Linear(320,dimension),nn.LayerNorm(dimension),nn.SiLU(),
                                  nn.Dropout(.1))
        self.action_emb=nn.Embedding(len(ACTIONS),action_embedding)
        self.head=nn.Sequential(nn.Linear(dimension+action_embedding,128),nn.SiLU(),
                                nn.Linear(128,5))
        self.repair_head=nn.Sequential(nn.Linear(dimension,64),nn.SiLU(),nn.Linear(64,1))
        self.quality_head=nn.Sequential(nn.Linear(dimension,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,image,features):
        encoded=self.shared(torch.cat((self.cnn(image),self.tab(features)),dim=1))
        batch=image.shape[0]
        ae=self.action_emb(torch.arange(len(ACTIONS),device=image.device))[None].expand(batch,-1,-1)
        merged=torch.cat([encoded.unsqueeze(1).expand(-1,len(ACTIONS),-1),ae],dim=-1)
        z=self.head(merged)
        mean=torch.tanh(z[...,0])
        med=torch.tanh(z[...,1])
        q10=(med-F.softplus(z[...,2])).clamp(-1,1)
        q90=(med+F.softplus(z[...,3])).clamp(-1,1)
        return {'mean':mean,'q10':q10,'q50':med,'q90':q90,
                'harm_logits':z[...,4],
                'repairability':torch.sigmoid(self.repair_head(encoded)).squeeze(-1),
                'quality':torch.sigmoid(self.quality_head(encoded)).squeeze(-1)}


def masked_mean(value, mask):
    return (value*mask).sum()/mask.sum().clamp_min(1.0)


def pinball(actual,pred,tau):
    err=actual-pred
    return torch.maximum(tau*err,(tau-1)*err)


def rari_loss(output,batch,weights=None,use_repairability=True):
    w={'gain':1.,'quantile':.5,'harm':.5,'repairability':1.,'quality':.25}
    if weights: w.update(weights)
    m=batch['valid']; gain=batch['gain']; harm=batch['harm'];
    gain_loss=masked_mean(F.smooth_l1_loss(output['mean'],gain,reduction='none'),m)
    q_loss=sum(masked_mean(pinball(gain,output[f'q{int(q*100)}'],q),m) for q in (.1,.5,.9))/3
    h_loss=masked_mean(F.binary_cross_entropy_with_logits(output['harm_logits'],harm,reduction='none'),m)
    r_loss=F.smooth_l1_loss(output['repairability'],batch['repairability']) if use_repairability else 0.0
    quality_loss=F.smooth_l1_loss(output['quality'],batch['quality'])
    loss=w['gain']*gain_loss+w['quantile']*q_loss+w['harm']*h_loss+w['repairability']*r_loss+w['quality']*quality_loss
    return loss, {k:float(v.detach().item()) if isinstance(v,torch.Tensor) else float(v) for k,v in
                  [('gain',gain_loss),('quantile',q_loss),('harm',h_loss),('repairability',r_loss),('quality',quality_loss)]}
