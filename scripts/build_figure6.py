"""Construct Figure 6 using real patient CT and masks only.

Install: pip install numpy matplotlib pillow nibabel
Usage: python build_rari_fig6.py --manifest figure6_cases.json --output Figure6_real.png
Do not use generated or illustrative CT images as experimental findings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLORS = {'ground_truth':'#16A34A','initial':'#F59E0B','regain':'#8B5CF6','rari':'#0284C7'}
HEADERS = ['Original CT', 'Expert reference', 'Initial prediction', 'ReGain correction', 'RARI decision']


def load_array(path, slice_index=None):
    p=Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'File not found: {p}')
    name=p.name.lower()
    if name.endswith('.npy'):
        arr=np.load(p, allow_pickle=False)
    elif name.endswith(('.nii','.nii.gz')):
        try:
            import nibabel as nib
        except ImportError as e:
            raise ImportError('Install nibabel to read NIfTI files') from e
        arr=np.asarray(nib.load(str(p)).dataobj)
        if arr.ndim==3: arr=np.moveaxis(arr,-1,0)
    elif name.endswith(('.png','.tif','.tiff','.jpg','.jpeg')):
        from PIL import Image
        arr=np.asarray(Image.open(p))
        if arr.ndim==3 and arr.shape[-1] in (3,4):
            arr=arr[:,:,:3].astype(np.float32).mean(axis=2)
    else:
        raise ValueError(f'Unsupported image format: {p}')
    if arr.ndim==3 and slice_index is not None:
        # RARI manifest convention: numpy and NIfTI volumes indexed (Z,Y,X)
        # Note: NIfTI input is transposed below into this orientation.
        if not 0<=int(slice_index)<arr.shape[0]:
            raise ValueError(f'slice_index {slice_index} not in [0,{arr.shape[0]-1}] for {p}')
        arr=arr[int(slice_index)]
    if arr.ndim!=2:
        raise ValueError(f'Expected 2D array or 3D volume with slice_index for {p}; got {arr.shape}')
    if not np.isfinite(arr).all():
        raise ValueError(f'Nonfinite pixels in {p}')
    return arr


def compute_dice(pred, reference):
    pred=np.asarray(pred,dtype=bool)
    reference=np.asarray(reference,dtype=bool)
    n=int(pred.sum())+int(reference.sum())
    return (2*float((pred & reference).sum())/n) if n else 1.0


def set_extent(ax, ct, mask=None, color=None, roi=None, vmin=None, vmax=None):
    ax.imshow(ct,cmap='gray',vmin=vmin,vmax=vmax,interpolation='nearest')
    if mask is not None and mask.any():
        rgba=np.zeros((mask.shape[0],mask.shape[1],4),dtype=np.float32)
        from matplotlib.colors import to_rgb
        rgba[:,:,:3]=to_rgb(color)
        rgba[:,:,3]=np.where(mask,0.23,0.0)
        ax.imshow(rgba, interpolation='nearest')
        ax.contour(mask.astype(float),levels=[0.5],colors=[color],linewidths=1.05,origin='upper')
    if roi:
        x,y,w,h=roi
        ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.4,edgecolor='#E11D48',linestyle='--'))
    ax.axis('off')


def validate_case(case):
    required=('case_id','label','ct','ground_truth','initial','regain','action')
    for k in required:
        if k not in case:
            raise ValueError(f'{case.get("case_id", "(unknown)")}: missing {k}')
    action=str(case['action']).strip().lower()
    if action not in ('repair','preserve','refer'):
        raise ValueError(f'{case["case_id"]}: action must be repair, preserve or refer')
    z=case.get('slice_index')
    ct=load_array(case['ct'],z)
    gt=load_array(case['ground_truth'],z).astype(bool)
    initial=load_array(case['initial'],z).astype(bool)
    regain=load_array(case['regain'],z).astype(bool)
    if action=='preserve':
        final=initial.copy() if not case.get('rari') else load_array(case['rari'],z).astype(bool)
        if not np.array_equal(final,initial):
            raise ValueError(f'{case["case_id"]}: preserve action requires RARI mask identical to initial mask')
    elif action=='repair':
        if not case.get('rari'):
            raise ValueError(f'{case["case_id"]}: repair requires actual RARI output mask')
        final=load_array(case['rari'],z).astype(bool)
    else: # Referral: not an automated final repaired segmentation
        final=None
    for tag,arr in [('ground_truth',gt),('initial',initial),('regain',regain),('rari',final)]:
        if arr is not None and arr.shape !=ct.shape:
            raise ValueError(f'{case["case_id"]}: {tag} shape {arr.shape} != CT {ct.shape}; do not silently resample')
    if not gt.any():
        raise ValueError(f'{case["case_id"]}: reference mask empty; verify eligibility, lesion type and annotation')
    roi=case.get('roi_xywh')
    if roi is not None:
        if len(roi)!=4:
            raise ValueError('roi_xywh must be [x,y,width,height]')
        x,y,w,h=map(int,roi)
        if x<0 or y<0 or w<=0 or h<=0 or x+w>ct.shape[1] or y+h>ct.shape[0]:
            raise ValueError(f'{case["case_id"]}: zoom ROI exceeds image dimensions')
        roi=(x,y,w,h)
    return ct,gt,initial,regain,final,roi,action


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',default='Figure6_real.png')
    args=ap.parse_args()
    obj=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    cases=obj.get('cases',[])
    if len(cases)!=3:
        raise ValueError('Figure 6 requires exactly three selected, verified patient cases')
    if len({c['case_id'] for c in cases})!=3:
        raise ValueError('Repeated case_id is not allowed')
    fig,axes=plt.subplots(3,5,figsize=(15.8,8.9),dpi=220)
    fig.patch.set_facecolor('white')
    for j,h in enumerate(HEADERS):
        axes[0,j].set_title(h,fontsize=11.5,fontweight='bold',pad=13)
    for i,case in enumerate(cases):
        ct,gt,initial,regain,final,roi,action=validate_case(case)
        center=np.nanmedian(ct)
        vmin=float(case.get('vmin',np.percentile(ct,1)))
        vmax=float(case.get('vmax',np.percentile(ct,99)))
        if vmax<=vmin: raise ValueError('vmax must exceed vmin')
        masks=[None,gt,initial,regain,final]
        colors=[None,COLORS['ground_truth'],COLORS['initial'],COLORS['regain'],COLORS['rari']]
        for j,ax in enumerate(axes[i]):
            mask=masks[j]
            set_extent(ax,ct,mask,color=colors[j],roi=roi,vmin=vmin,vmax=vmax)
            if j==4 and action=='refer':
                ax.text(0.5,0.05,'REFER TO EXPERT — no automatic output',color='white',
                    weight='bold',fontsize=8.4,ha='center',transform=ax.transAxes,
                    bbox=dict(facecolor='black',alpha=0.65,pad=5))
            elif j>=2 and mask is not None:
                d=compute_dice(mask,gt)
                ax.text(0.02,0.04,f'Dice: {d:.3f}',color='white',fontsize=10.1,
                        ha='left',weight='bold',transform=ax.transAxes,
                        bbox=dict(facecolor='black',alpha=0.5,pad=2,edgecolor='none'))
            if j==4 and action=='preserve':
                ax.text(0.98,0.96,'PRESERVE',color='white',fontsize=8.5,ha='right',va='top',
                        transform=ax.transAxes,bbox=dict(facecolor='#14532D',alpha=0.9,pad=3))
            if roi:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                x,y,w,h=roi
                inset=inset_axes(ax,width='37%',height='37%',loc='lower right',borderpad=0.25)
                sub_ct=ct[y:y+h,x:x+w]
                sub_mask=mask[y:y+h,x:x+w] if mask is not None else None
                set_extent(inset,sub_ct,sub_mask,colors[j],None,vmin,vmax)
                for spine in inset.spines.values():
                    spine.set_visible(True);spine.set_edgecolor('#E11D48');spine.set_linewidth(1.5)
        label=case['label']
        phase=f" | {case['phase']}" if case.get('phase') else ''
        axes[i,0].text(-0.03,0.5,f'{chr(65+i)}. {label}{phase}',fontsize=10.4,
                       weight='bold',va='center',ha='right',rotation=90,
                       transform=axes[i,0].transAxes)
    fig.subplots_adjust(left=0.11,right=0.99,top=0.92,bottom=0.045,wspace=0.025,hspace=0.16)
    outfile=Path(args.output)
    outfile.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outfile,dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('Wrote',outfile.resolve())
    for c in cases:
        ct,gt,ini,rg,fin,roi,action=validate_case(c)
        print(c['case_id'], 'initial',f'{compute_dice(ini,gt):.3f}',
              'regain',f'{compute_dice(rg,gt):.3f}',
              'RARI',f'{compute_dice(fin,gt):.3f}' if fin is not None else 'REFERRAL')

if __name__=='__main__':
    main()
