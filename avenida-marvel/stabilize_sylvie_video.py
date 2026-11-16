#!/usr/bin/env python3
"""Hold an approved seated-background face behind the moving foreground layer."""
from pathlib import Path
import json,subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw

R=Path(__file__).resolve().parent;W=R/'work';Q=W/'stand_fix_qa'

def foreground_head(cx,cy,fw,fh=183):
    p=np.array([[cx-.50*fw,cy-.98*fh],[cx+.70*fw,cy-.98*fh],
                [cx+.95*fw,cy+.95*fh],[cx-.90*fw,cy+1.20*fh],
                [cx-.90*fw,cy+.65*fh],[cx-.64*fw,cy-.35*fh],
                [cx-.40*fw,cy-.70*fh]],np.int32)
    m=np.zeros((364,576),np.float32);cv2.fillPoly(m,[p],1)
    return cv2.GaussianBlur(m,(9,9),0)

def main():
    meta={x['frame']:x for x in json.loads((Q/'metrics.json').read_text())}
    approved=cv2.imread(str(W/'anchors/0055.png'))
    clean=np.zeros((364,576),np.float32)
    cv2.ellipse(clean,(272,73),(91,125),0,0,360,1,-1)
    clean*=1-foreground_head(*meta[55]['generated_head'])
    clean=cv2.GaussianBlur(clean,(15,15),0)
    out=Q/'sylvie_stable';out.mkdir(exist_ok=True)
    gallery=Image.new('RGB',(1152,486))
    for n in range(56,65):
        current=cv2.imread(str(Q/'final'/f'{n:04d}.png'))
        dy=-(n-55)*1.5
        matrix=np.array([[1,0,0],[0,1,dy]],np.float32)
        layer=cv2.warpAffine(approved,matrix,(576,364),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        alpha=cv2.warpAffine(clean,matrix,(576,364),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        fg=foreground_head(*meta[n]['generated_head'])
        alpha*=1-fg
        a=alpha[...,None]
        result=current.copy();base=current[230:594]
        result[230:594]=np.clip(base.astype(np.float32)*(1-a)+layer.astype(np.float32)*a+.5,0,255).astype(np.uint8)
        cv2.imwrite(str(out/f'{n:04d}.png'),result,[cv2.IMWRITE_PNG_COMPRESSION,2])
        if n in [60,62]:
            row=[60,62].index(n)
            for col,(im,title) in enumerate([(current[230:594],'before'),(result[230:594],'stable Sylvie')]):
                p=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ImageDraw.Draw(p).text((6,6),f'{n} {title}',fill='red',stroke_width=1,stroke_fill='black');gallery.paste(p,(col*576,row*243))
    gallery.save(Q/'sylvie-stable-test.jpg')

if __name__=='__main__':main()
