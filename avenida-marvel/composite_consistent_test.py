#!/usr/bin/env python3
"""Video matte/compositing tests using already generated character frames."""
from pathlib import Path
import time, json
import cv2
import numpy as np
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'
OUT=WORK/'consistency_test'

class PersonMatte:
    def __init__(self):
        torch.set_num_threads(2)
        self.net=lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT).eval()
        self.mean=torch.tensor([.485,.456,.406])[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225])[None,:,None,None]
    def __call__(self, im):
        x=torch.from_numpy(np.ascontiguousarray(im[...,::-1])).permute(2,0,1)[None].float()/255
        with torch.inference_mode():
            p=self.net((x-self.mean)/self.std)['out'].softmax(1)[0,15].numpy()
        binary=(p>.25).astype(np.uint8)
        count,lab,stats,_=cv2.connectedComponentsWithStats(binary)
        clean=np.zeros_like(binary)
        for i in range(1,count):
            if stats[i,cv2.CC_STAT_AREA]>200: clean[lab==i]=1
        return p,clean

def soft_mask(mask,dilate=3,feather=3):
    m=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate+1,2*dilate+1)))
    if feather: m=cv2.GaussianBlur(m.astype(np.float32),(2*feather+1,2*feather+1),0)
    return np.clip(m,0,1)

def mix(bg, fg, a):
    return np.clip(bg.astype(np.float32)*(1-a[...,None])+fg.astype(np.float32)*a[...,None]+.5,0,255).astype(np.uint8)

def main():
    OUT.mkdir(exist_ok=True,parents=True); model=PersonMatte(); started=time.time()
    records=[]
    panel=[]
    for s in [359,369,389,444]:
        src=cv2.imread(str(WORK/'action'/f'{s:04d}.png'))
        gen=cv2.imread(str(WORK/'review_rife'/f'{2*s:04d}.png'))
        po,mo=model(src);pn,mn=model(gen)
        union=soft_mask(np.maximum(mo,mn),3,3)
        comp=mix(src,gen,union)
        # Foreground-only test with source old actor removed by deterministic video cleanplate.
        back=cv2.inpaint(src,cv2.dilate(mo,np.ones((7,7),np.uint8))*255,3,cv2.INPAINT_TELEA)
        only=mix(back,gen,soft_mask(mn,2,2))
        maskvis=np.dstack([mo*255,mn*255,np.zeros_like(mo)])
        row=np.concatenate([src,gen,maskvis,comp,only],axis=1)
        panel.append(cv2.resize(row,(1440,182)))
        for label,im in [('union',comp),('foreground',only),('mask',maskvis)]:
            cv2.imwrite(str(OUT/f'{s:04d}-{label}.png'),im)
        outside=union==0
        records.append({'source':s,'old_person_coverage':float(mo.mean()),'new_person_coverage':float(mn.mean()),'union_coverage':float((union>0).mean()),'original_background_mae':float(np.abs(comp.astype(float)-src.astype(float))[outside].mean())})
    cv2.imwrite(str(OUT/'matte-comparison.jpg'),np.concatenate(panel,axis=0))
    (OUT/'test-metrics.json').write_text(json.dumps({'runtime':time.time()-started,'frames':records},indent=2))
    print(json.dumps({'runtime':time.time()-started,'frames':records},indent=2))

if __name__=='__main__':main()
