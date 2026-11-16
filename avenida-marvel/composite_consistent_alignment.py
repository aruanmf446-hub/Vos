#!/usr/bin/env python3
"""Conservative background registration for existing character video layers."""
import cv2
import numpy as np


def warp_image(image, matrix, size, nearest=False):
    return cv2.warpAffine(image,matrix,size,flags=(cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT_101)


def normalized_cost(a,b,mask):
    x=a[mask].astype(np.float32);y=b[mask].astype(np.float32)
    if len(x)<100:return float('inf')
    x=(x-x.mean())/max(float(x.std()),8.)
    y=(y-y.mean())/max(float(y.std()),8.)
    return float(np.minimum((x-y)**2,4).mean())


def fit_background(source,generated,old_mask,new_mask):
    """Fit a bounded affine transform; reject fits without measured improvement.

    Coordinates map target/source pixels to generated input pixels, as expected
    by warpAffine's WARP_INVERSE_MAP. No region outside the person union is used
    to define a character's identity or expression.
    """
    h,w=source.shape[:2]
    union=np.maximum(old_mask,new_mask).astype(np.uint8)
    background=cv2.dilate(union,np.ones((17,17),np.uint8))==0
    interior=np.zeros_like(background);interior[16:h-16,16:w-16]=True
    background &= interior
    identity=np.eye(2,3,dtype=np.float32)
    if background.mean()<.10:return identity,{'accepted':False,'reason':'too_little_background'}
    a=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(generated,cv2.COLOR_BGR2GRAY)
    mask=(background*255).astype(np.uint8)
    before=normalized_cost(a,b,background)
    best=identity.copy();best_cost=before;kind='identity';correlation=None
    for method,name in [(cv2.MOTION_TRANSLATION,'translation'),(cv2.MOTION_AFFINE,'affine')]:
        trial=best.copy()
        try:
            correlation,trial=cv2.findTransformECC(a,b,trial,method,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,70,1e-5),mask,5)
        except cv2.error:continue
        linear=trial[:,:2]
        sx=np.linalg.norm(linear[:,0]);sy=np.linalg.norm(linear[:,1])
        angle=np.degrees(np.arctan2(linear[1,0],linear[0,0]))
        shear=abs(float(np.dot(linear[:,0],linear[:,1])))
        if max(abs(float(trial[0,2])),abs(float(trial[1,2])))>15:continue
        if not (.95<=sx<=1.05 and .95<=sy<=1.05 and abs(angle)<=2 and shear<.035):continue
        aligned=warp_image(b,trial,(w,h))
        cost=normalized_cost(a,aligned,background)
        if cost<best_cost*.97:best=trial;best_cost=cost;kind=name
    accepted=kind!='identity' and best_cost<before*.95
    if not accepted:best=identity
    return best,{'accepted':accepted,'method':kind if accepted else 'identity','before':before,'after':best_cost if accepted else before,'matrix':best.tolist(),'background_fraction':float(background.mean())}


def fit_color(source,generated,old_mask,new_mask):
    """Conservative per-shot exposure/color match measured only on background."""
    union=np.maximum(old_mask,new_mask).astype(np.uint8)
    bg=cv2.dilate(union,np.ones((19,19),np.uint8))==0
    if bg.mean()<.1:return np.ones(3,np.float32),np.zeros(3,np.float32)
    x=generated[bg].astype(np.float32);y=source[bg].astype(np.float32)
    # Percentiles are more robust to mismatched decorative background details
    # than a pixelwise color regression. Tight caps protect skin/costume colors.
    xa,xb=np.percentile(x,[20,80],axis=0);ya,yb=np.percentile(y,[20,80],axis=0)
    gain=np.clip((yb-ya)/np.maximum(xb-xa,20),.93,1.07)
    bias=np.clip(np.median(y,axis=0)-gain*np.median(x,axis=0),-8,8)
    return gain.astype(np.float32),bias.astype(np.float32)


def apply_color(image,gain,bias):
    return np.clip(image.astype(np.float32)*gain[None,None,:]+bias[None,None,:]+.5,0,255).astype(np.uint8)
