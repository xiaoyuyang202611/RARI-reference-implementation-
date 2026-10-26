"""Illustrative NON-MedSAM actions used ONLY to test the reference pipeline.

DO NOT claim they are the manuscript's actual learned prompt/local/boundary/volume
operators. For real evaluation replace with logged outputs from real segmentation
backbones and validated repair algorithms.
"""
import numpy as np
from scipy import ndimage as ndi


def demo_action(image,mask,name):
    img=np.asarray(image,dtype=np.float32)
    mask=np.asarray(mask,dtype=bool)
    if img.shape!=mask.shape:raise ValueError('Image/mask mismatch')
    local=ndi.gaussian_filter(img,1.0)
    if name=='prompt':
        region=ndi.binary_dilation(mask,iterations=3)
        anchor=np.median(img[mask]) if mask.any() else float(np.median(img))
        return (region & (local > anchor*.70)) | ndi.binary_erosion(mask,iterations=1)
    if name=='local':
        region=ndi.binary_dilation(mask,iterations=2)
        threshold=np.percentile(local[mask],35) if mask.any() else np.percentile(local,80)
        return ndi.binary_fill_holes(region & (local>threshold))
    if name=='boundary':
        return ndi.binary_closing(ndi.binary_opening(mask,iterations=1),iterations=1)
    if name=='volume':
        if mask.ndim==3:
            return ndi.gaussian_filter(mask.astype(float),(1,1,1))>.45
        return ndi.gaussian_filter(mask.astype(float),1.2)>.43
    raise ValueError(name)
