#!/usr/bin/env python3
"""Refine only approved standing-shot video layers; no anchors are modified."""
from pathlib import Path
from bisect import bisect_left
import json,subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw
from render_rife_backend import Rife
from composite_consistent import OriginalBackgroundCompositor
from composite_consistent_alignment import apply_color
from composite_consistent_test import soft_mask,mix

ROOT=Path(__file__).resolve().parent;W=ROOT/'work';Q=W/'stand_fix_qa'

def clean(a):
    a=a.copy();p=4
    a[:p]=a[p:p+1];a[-p:]=a[-p-1:-p]
    a[:,:p]=a[:,p:p+1];a[:,-p:]=a[:,-p-1:-p]
    return a

def main():
    cv2.setNumThreads(1)
    keys=[39,41,43,45,47,49,51,53,55,65,67,69,72,75]
    anchors={i:clean(cv2.imread(str(W/'anchors'/f'{i:04d}.png'))) for i in keys}
    rife=Rife(threads=1)
    meta={r['frame']:r for r in json.loads((Q/'metrics.json').read_text())}
    face_a=cv2.resize(anchors[55][116:299,334:487],(192,232),interpolation=cv2.INTER_LANCZOS4)
    face_b=cv2.resize(anchors[65][54:240,300:448],(192,232),interpolation=cv2.INTER_LANCZOS4)
    mask=np.zeros((232,192),np.float32)
    polygon=np.array([[24,1],[167,1],[191,68],[176,164],[143,216],[96,231],[48,216],[15,164],[0,68]],np.int32)
    cv2.fillConvexPoly(mask,polygon,1);mask=cv2.GaussianBlur(mask,(15,15),0)
    c=OriginalBackgroundCompositor(registration=False,segmentation_stride=2)
    p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size','576x854','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(W/'stand_fix_30fps.mp4')],stdin=subprocess.PIPE)
    samples=[53,55,57,60,64,65,75];gallery=Image.new('RGB',(1152,243*len(samples)))
    for n in range(39,76):
        if n in anchors:gen=anchors[n].copy()
        else:
            pos=bisect_left(keys,n);a,b=keys[pos-1],keys[pos]
            gen=rife.between(anchors[a],anchors[b],(n-a)/(b-a),('body',a,b))
        r=meta[n]
        if 55<n<65:
            face=rife.between(face_a,face_b,(n-55)/10,('face',55,65))
            cx,cy,fw=r['generated_head'];fh=183+(n-55)*.3
            x0,y0=int(round(cx-fw/2)),int(round(cy-fh/2));ww,hh=int(round(fw)),int(round(fh))
            patch=cv2.resize(face,(ww,hh),interpolation=cv2.INTER_LANCZOS4)
            alpha=cv2.resize(mask,(ww,hh))[...,None]
            xa,ya=max(0,x0),max(0,y0);xb,yb=min(576,x0+ww),min(364,y0+hh)
            if xa<xb and ya<yb:
                px,py=xa-x0,ya-y0;a=alpha[py:py+yb-ya,px:px+xb-xa]
                gen[ya:yb,xa:xb]=np.clip(gen[ya:yb,xa:xb].astype(np.float32)*(1-a)+patch[py:py+yb-ya,px:px+xb-xa].astype(np.float32)*a,0,255).astype(np.uint8)
        # Keep the complete approved body image. Large translations would
        # expose body regions the generated crop does not contain.
        aligned=gen
        source=cv2.imread(str(W/'source'/f'{n:04d}.png'))
        fixed,_=c.process(source[230:594],aligned,n,'stand')
        union=np.maximum(c.source_mask,c.previous_raw_mask).astype(np.uint8)
        points=cv2.findNonZero(union)
        if points is not None:
            hull=np.zeros_like(union);cv2.fillConvexPoly(hull,cv2.convexHull(points),1)
            alpha=soft_mask(hull,3,6)
            fixed=mix(source[230:594],apply_color(aligned,c.gain,c.bias),alpha)
        full=source.copy();full[230:594]=fixed
        cv2.imwrite(str(Q/'final'/f'{n:04d}.png'),full,[cv2.IMWRITE_PNG_COMPRESSION,2])
        p.stdin.write(full.tobytes())
        if n in samples:
            row=samples.index(n)
            for col,(im,label) in enumerate([(source[230:594],'source'),(gen,'clean isolated face'),(fixed,'final composition')]):
                a=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((384,243));ImageDraw.Draw(a).text((4,4),f'{n} {label}',fill='red',stroke_width=1,stroke_fill='black');gallery.paste(a,(384*col,243*row))
        print(n,flush=True)
    p.stdin.close()
    if p.wait():raise RuntimeError('encoding')
    gallery.save(Q/'refined-comparison.jpg')
    flash=cv2.imread(str(W/'anchors/0746.png'))
    fp=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size','576x854','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','1','-pix_fmt','yuv420p',str(W/'flash_fix_30fps.mp4')],stdin=subprocess.PIPE)
    for n in [747,748]:
        source=cv2.imread(str(W/'source'/f'{n:04d}.png'))
        fixed,_=c.process(source[230:594],flash,n,'final_flash')
        full=source.copy();full[230:594]=fixed
        cv2.imwrite(str(Q/f'flash-{n}.png'),full)
        fp.stdin.write(full.tobytes())
    fp.stdin.close()
    if fp.wait():raise RuntimeError('flash encoder')
    print('Finished refined patch',flush=True)

if __name__=='__main__':main()
