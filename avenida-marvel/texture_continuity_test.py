#!/usr/bin/env python3
"""Foreground-only source-camera texture matching; changes no timing or poses."""
from pathlib import Path
import json, subprocess, time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from composite_consistent_test import PersonMatte

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'
OUT=WORK/'texture_test'
INPUT=Path('/workspace/shared/downloads/video-continuo-em-revisao-60fps.mp4')


def encoder(path,width,height):
    return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24',
        '-video_size',f'{width}x{height}','-framerate','60','-i','pipe:0','-an','-c:v','libx264',
        '-preset','fast','-crf','17','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)


class TextureMatch:
    def __init__(self):
        self.person=PersonMatte()
        cv2.setNumThreads(2)
        self.dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.grid=None
        self.previous_gray=None
        self.previous_alpha=None
        self.grain=None
        self.rng=np.random.default_rng(1739)
        self.number=0

    def process(self,roi):
        h,w=roi.shape[:2];small=cv2.resize(roi,(384,243),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        if self.grid is None:
            yy,xx=np.mgrid[:243,:384].astype(np.float32);self.grid=np.dstack([xx,yy])
        flow=None
        if self.previous_gray is not None:flow=self.dis.calc(gray,self.previous_gray,None)
        transported=None
        if self.previous_alpha is not None:
            m=self.grid+flow
            transported=cv2.remap(self.previous_alpha,m[...,0],m[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        if self.number%6==0 or transported is None:
            probability,binary=self.person(small)
            current=cv2.GaussianBlur(np.clip((probability-.18)/.55,0,1),(5,5),0)
            current*=cv2.dilate(binary,np.ones((3,3),np.uint8))
            alpha=current if transported is None else .8*current+.2*transported
        else:alpha=transported
        alpha[alpha<.015]=0
        self.previous_alpha=alpha
        full_alpha=cv2.resize(alpha,(w,h),interpolation=cv2.INTER_LINEAR)
        full_alpha[full_alpha<.015]=0
        # Match the source camera's softer transfer function only on actors.
        # Native equivalent sigma=.45px; this is not a global blur or frame blend.
        softened=cv2.GaussianBlur(roi,(0,0),.56)
        noise=self.rng.normal(0,1,(h,w)).astype(np.float32)
        if self.grain is None:self.grain=noise
        else:
            field=cv2.resize(flow,(w,h));field[...,0]*=w/384;field[...,1]*=h/243
            yy,xx=np.mgrid[:h,:w].astype(np.float32)
            prior=cv2.remap(self.grain,xx+field[...,0],yy+field[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            self.grain=.88*prior+np.sqrt(1-.88**2)*noise
        # Subtle correlated luma grain, not colored noise or random flicker.
        textured=softened.astype(np.float32)+self.grain[...,None]*.65
        result=np.clip(roi.astype(np.float32)*(1-full_alpha[...,None])+textured*full_alpha[...,None]+.5,0,255).astype(np.uint8)
        self.previous_gray=gray;self.number+=1
        outside=full_alpha==0
        return result,{'coverage':float((full_alpha>0).mean()),
            'outside_mask_mae':float(np.abs(result.astype(float)-roi.astype(float))[outside].mean()) if outside.any() else None,
            'foreground_change_mae':float(np.abs(result.astype(float)-roi.astype(float))[full_alpha>.75].mean())}


def run(label,start,length):
    cap=cv2.VideoCapture(str(INPUT));cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    comparison=OUT/f'{label}-texture-comparison.mp4';fixed=OUT/f'{label}-texture-only.mp4'
    # Active picture only for legible side-by-side assessment; standalone retains full canvas.
    e=encoder(comparison,1440,492);f=encoder(fixed,720,1068)
    head=Image.new('RGB',(1440,36),(18,20,25));d=ImageDraw.Draw(head)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    d.text((12,4),'Atual',font=font,fill='white');d.text((732,4),'Textura da câmera (só personagens)',font=font,fill='white')
    header=np.asarray(head)[...,::-1].copy()
    match=TextureMatch();stats=[];began=time.monotonic()
    for i in range(length):
        ok,frame=cap.read()
        if not ok:raise RuntimeError(f'Missing frame {start+i}')
        roi=frame[288:743]
        corrected,metrics=match.process(roi);stats.append(metrics)
        full=frame.copy();full[288:743]=corrected
        panel=np.zeros((492,1440,3),np.uint8);panel[:36]=header;panel[36:491,:720]=roi;panel[36:491,720:]=corrected
        e.stdin.write(panel.tobytes());f.stdin.write(full.tobytes())
        if i in [0,60,90,120,length-1]:cv2.imwrite(str(OUT/f'{label}-{start+i:04d}.jpg'),panel)
        if i%60==0:print(label,i,length,round(time.monotonic()-began,1),flush=True)
    cap.release()
    for enc in [e,f]:
        enc.stdin.close()
        if enc.wait():raise RuntimeError('Encoder failed')
    summary={'input':str(INPUT),'start_frame_60fps':start,'frames':length,'duration':length/60,
        'fps':60,'timing_poses_or_interpolation_changed':False,'foreground_only':True,
        'blur_sigma_output_pixels':.56,'grain_sigma_luma':.65,
        'outside_mask_mae':float(np.mean([r['outside_mask_mae'] for r in stats if r['outside_mask_mae'] is not None])),
        'foreground_change_mae':float(np.mean([r['foreground_change_mae'] for r in stats])),
        'comparison':str(comparison),'standalone':str(fixed),'runtime':time.monotonic()-began}
    (OUT/f'{label}-texture-metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    run('wanda',888,240)
    run('strange',152,60)
