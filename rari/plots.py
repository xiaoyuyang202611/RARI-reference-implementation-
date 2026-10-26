"""Draw diagnostic plots FROM evaluated case and candidate-action source data."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',default='runs/evaluation')
    ap.add_argument('--output',default='runs/plots');a=ap.parse_args()
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(Path(a.input)/'case_results.csv')
    grp=df.groupby('method',sort=False)['final_dice'].mean().sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(11,5));ax.bar(grp.index,grp.values)
    ax.set_ylim(0,1);ax.set_ylabel('Mean final Dice');ax.set_title('Held-out segmentation outcome by policy')
    ax.tick_params(axis='x',rotation=50);fig.tight_layout();fig.savefig(folder/'dice_comparison.png',dpi=200);plt.close(fig)
    cpath=Path(a.input)/'action_calibration.csv'
    if cpath.exists():
        c=pd.read_csv(cpath)
        if len(c):
            probs=c['predicted_harm_probability'].to_numpy();y=c['observed_harm'].to_numpy()
            bins=np.linspace(0,1,11)
            loc=np.digitize(probs,bins[1:-1])
            xs=[];ys=[];counts=[]
            for i in range(10):
                idx=loc==i
                if idx.any():xs.append(float(probs[idx].mean()));ys.append(float(y[idx].mean()));counts.append(int(idx.sum()))
            fig,ax=plt.subplots(figsize=(5,5));ax.plot([0,1],[0,1],linestyle='--',label='Perfect calibration')
            if xs:ax.plot(xs,ys,marker='o',label='Observed')
            ax.set(xlabel='Predicted material-loss risk',ylabel='Observed rate',xlim=(0,1),ylim=(0,1),title='Action risk reliability')
            ax.legend();fig.tight_layout();fig.savefig(folder/'risk_reliability.png',dpi=200);plt.close(fig)
    print('Plots:',folder.resolve())
if __name__=='__main__':main()
