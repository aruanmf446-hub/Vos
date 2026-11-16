#!/usr/bin/env python3
"""Stabilize the approved seated layer, occluding it with clean Wanda hair."""
from pathlib import Path
import json,subprocess
import cv2
import numpy as np
from PIL import Image
from composite_consistent_test import PersonMatte

ROOT=Path(__file__).resolve().parent;W=ROOT/'work';Q=W/'stand_fix_qa'

def main():
    cv2.setNumThreads(1)
    meta={r['frame']:r for r in json.loads((Q/'metrics.json').read_text())}
    src=cv2.imread(str(W/'anchors/0065.png'))
    src[:4]=src[4:5];src[-4:]=src[-5:-4];src[:,:4]=src[:,4:5];src[:,-4:]=src[:,-5:-4]
    _,person=PersonMatte()(src)
    roi=np.zeros((364,576),np.float32)
    poly=np.array([[230,0],[520,0],[555,190],[520,295],[435,300],[310,300],[235,260],[210,170]],np.int32)
    cv2.fillPoly(roi,[poly],1)
    vertical=np.clip((300-np.arange(364))/45,0,1).astype(np.float32)[:,None]
    matte=cv2.GaussianBlur(person.astype(np.float32)*roi*vertical,(9,9),0)
    before={n:cv2.imread(str(Q/'final'/f'{n:04d}.png')) for n in [60,62]}
    for n in range(56,65):
        base=cv2.imread(str(Q/'sylvie_stable'/f'{n:04d}.png'))
        cx,cy,fw=meta[n]['generated_head'];sx=fw/145.9;sy=183.5/183.8
        matrix=np.array([[sx,0,cx-sx*373.85],[0,sy,cy-sy*146.4]],np.float32)
        layer=cv2.warpAffine(src,matrix,(576,364),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        alpha=cv2.warpAffine(matte,matrix,(576,364),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        alpha=cv2.GaussianBlur(alpha,(17,17),0)
        x0,y0=cx-fw/2,cy-183.5/2
        face=np.zeros_like(alpha)
        pts=np.array([[x0+.15*fw,y0],[x0+.85*fw,y0],[x0+fw,y0+.3*183.5],[x0+.9*fw,y0+.7*183.5],[cx,y0+183.5],[x0+.1*fw,y0+.7*183.5],[x0,y0+.3*183.5]],np.int32)
        cv2.fillPoly(face,[pts],1);face=cv2.GaussianBlur(face,(13,13),0)
        alpha*=1-face
        a=alpha[...,None]
        out=base.copy();out[230:594]=np.clip(base[230:594].astype(np.float32)*(1-a)+layer.astype(np.float32)*a+.5,0,255).astype(np.uint8)
        cv2.imwrite(str(Q/'final'/f'{n:04d}.png'),out,[cv2.IMWRITE_PNG_COMPRESSION,2])
        if n in [60,62]:
            grid=Image.new('RGB',(1152,364));grid.paste(Image.fromarray(cv2.cvtColor(before[n][230:594],cv2.COLOR_BGR2RGB)),(0,0));grid.paste(Image.fromarray(cv2.cvtColor(out[230:594],cv2.COLOR_BGR2RGB)),(576,0));grid.save(Q/f'approved-occlusion-{n}.jpg')
    p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size','576x854','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(W/'stand_fix_30fps.mp4')],stdin=subprocess.PIPE)
    for n in range(39,76):p.stdin.write(cv2.imread(str(Q/'final'/f'{n:04d}.png')).tobytes())
    p.stdin.close()
    if p.wait():raise RuntimeError('encoder failed')
    print('Updated 9 occlusion frames and complete 37-frame patch',flush=True)

if __name__=='__main__':main()
