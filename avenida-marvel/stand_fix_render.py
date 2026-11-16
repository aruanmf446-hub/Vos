#!/usr/bin/env python3
"""Isolated stand-shot repair; never changes project anchors or main outputs."""
from pathlib import Path
from bisect import bisect_left
import json,subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d,median_filter
from PIL import Image,ImageDraw
from render_rife_backend import Rife
from composite_consistent import OriginalBackgroundCompositor

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'
OUT=WORK/'stand_fix_qa'

def encode(path,frames):
    p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size','576x854','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    for f in frames:p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait():raise RuntimeError('encoder')

def main():
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    keys=[39,41,43,45,47,49,51,53,55,65,67,69,72,75]
    anchors={i:cv2.imread(str(WORK/'anchors'/f'{i:04d}.png')) for i in keys}
    rife=Rife(threads=1)
    sources=[];generated=[]
    for n in range(39,76):
        src=cv2.imread(str(WORK/'source'/f'{n:04d}.png'));sources.append(src)
        if n in anchors:gen=anchors[n].copy()
        else:
            pos=bisect_left(keys,n);a,b=keys[pos-1],keys[pos]
            gen=rife.between(anchors[a],anchors[b],(n-a)/(b-a),(a,b))
        generated.append(gen)
    detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet.onnx'),'',(576,364),.45,.3,5000)
    def face(im):
        _,found=detector.detect(im)
        if found is None:return None
        found=[f for f in found if f[2]>45 and f[3]>55]
        if not found:return None
        f=max(found,key=lambda r:r[1]+r[3]/2+.08*(r[0]+r[2]/2))
        return np.array([f[0]+f[2]/2,f[1]+f[3]/2,f[2]],np.float32)
    sf=[face(s[230:594]) for s in sources]
    gf=[face(g) for g in generated]
    for values in [sf,gf]:
        valid=[j for j,v in enumerate(values) if v is not None]
        for j,v in enumerate(values):
            if v is None:values[j]=np.array([np.interp(j,valid,[values[k][c] for k in valid]) for c in range(3)],np.float32)
    transforms=[]
    for j,n in enumerate(range(39,76)):
        s,g=sf[j],gf[j]
        strength=float(np.clip((n-45)/10,0,1))
        scale=1+strength*(float(np.clip(s[2]/max(g[2],1),.91,1.12))-1)
        target=g[:2]+strength*(s[:2]-g[:2])
        transforms.append([scale,target[0]-scale*g[0],target[1]-scale*g[1]])
    transforms=np.asarray(transforms)
    for c in range(3):transforms[:,c]=gaussian_filter1d(median_filter(transforms[:,c],size=3,mode='nearest'),.55,mode='nearest')
    baseline=OriginalBackgroundCompositor(registration=True,segmentation_stride=1)
    aligned_compositor=OriginalBackgroundCompositor(registration=False,segmentation_stride=1)
    frames_base=[];frames_aligned=[];records=[]
    snapshots=[53,55,57,60,64,65,75]
    gallery=Image.new('RGB',(1152,243*len(snapshots)))
    final_dir=OUT/'final';final_dir.mkdir(exist_ok=True)
    for j,n in enumerate(range(39,76)):
        source=sources[j];gen=generated[j]
        base,_=baseline.process(source[230:594],gen,n,'stand')
        scale,dx,dy=transforms[j];matrix=np.array([[scale,0,dx],[0,scale,dy]],np.float32)
        aligned=cv2.warpAffine(gen,matrix,(576,364),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        fixed,diag=aligned_compositor.process(source[230:594],aligned,n,'stand')
        a=source.copy();a[230:594]=base;frames_base.append(a)
        b=source.copy();b[230:594]=fixed;frames_aligned.append(b)
        cv2.imwrite(str(final_dir/f'{n:04d}.png'),b,[cv2.IMWRITE_PNG_COMPRESSION,2])
        records.append({'frame':n,'source_head':sf[j].tolist(),'generated_head':gf[j].tolist(),'transform':transforms[j].tolist(),'coverage':diag['coverage']})
        if n in snapshots:
            row=snapshots.index(n)
            for col,(pic,label) in enumerate([(source[230:594],'source'),(base,'filtered RIFE'),(fixed,'head aligned')]):
                im=Image.fromarray(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)).resize((384,243));ImageDraw.Draw(im).text((5,5),f'{n} {label}',fill='red',stroke_width=1,stroke_fill='black');gallery.paste(im,(col*384,row*243))
        print(n,flush=True)
    gallery.save(OUT/'comparison.jpg')
    (OUT/'metrics.json').write_text(json.dumps(records,indent=2))
    encode(WORK/'stand_fix_30fps.mp4',frames_aligned)
    encode(OUT/'baseline.mp4',frames_base)
    print('Finished',flush=True)

if __name__=='__main__':main()
