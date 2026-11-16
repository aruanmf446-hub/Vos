"""Recover camera-revealed background around a single shot cleanplate."""
from pathlib import Path
import warnings,cv2,numpy as np
from composite_consistent_test import PersonMatte
R=Path(__file__).resolve().parent;W=R/'work';D=W/'clean_sides_v2';cv2.setNumThreads(2)
motion=np.load(D/'camera-translation.npy');pad=128
base=cv2.imread(str(D/'cleanplate-without-encoded-border.png'))
base=cv2.copyMakeBorder(base,pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
H,WW=base.shape[:2];samples=[];segment=None
for n in [76,88,100,112,124,136,148,160,172]:
    src=cv2.imread(str(W/'action'/f'{n:04d}.png'));p=D/'review-mattes'/f'removal-{n:04d}.npy'
    if p.exists():alpha=np.load(p)
    else:
        if segment is None:segment=PersonMatte()
        prob,_=segment(src);m=(prob>.10).astype(np.uint8)
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
        m=cv2.dilate(m,np.ones((29,29),np.uint8));alpha=cv2.GaussianBlur(m.astype(np.float32),(11,11),2)
        np.save(p,alpha)
    visible=(alpha<.005).astype(np.uint8)
    visible[:4]=0;visible[-4:]=0;visible[:,:4]=0;visible[:,-4:]=0
    inverse=cv2.invertAffineTransform(motion[n-76]);inverse[:,2]+=pad
    aligned=cv2.warpAffine(src,inverse,(WW,H)).astype(np.float32)
    valid=cv2.warpAffine(visible,inverse,(WW,H),flags=cv2.INTER_NEAREST)>0
    aligned[~valid]=np.nan;samples.append(aligned)
with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning);observed=np.nanmedian(np.stack(samples),axis=0)
known=np.isfinite(observed).all(2)
distance=cv2.distanceTransform(known.astype(np.uint8),cv2.DIST_L2,3)
blend=np.minimum(distance/4,1)[...,None]
observed[~known]=base[~known]
result=np.clip(base*(1-blend)+observed*blend+.5,0,255).astype(np.uint8)
cv2.imwrite(str(D/'panorama-background.png'),result)
cv2.imwrite(str(D/'panorama-observed-pixels.png'),known.astype(np.uint8)*255)
print('Created camera-aligned panorama from nine original frames.',flush=True)
