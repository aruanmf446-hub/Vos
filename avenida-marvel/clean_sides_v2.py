"""One camera-tracked cleanplate and independently interpolated foreground."""
from pathlib import Path
import json, subprocess, time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from composite_consistent_test import PersonMatte
from render_rife_backend import Rife

R=Path(__file__).resolve().parent; W=R/'work'; D=W/'clean_sides_v2'
D.mkdir(exist_ok=True); OUT=Path('/workspace/shared/downloads')
cv2.setNumThreads(2); segment=PersonMatte(); started=time.monotonic()
def matte(im,expand=1):
    p,_=segment(im)
    m=(p>.16).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    if expand:m=cv2.dilate(m,np.ones((2*expand+1,2*expand+1),np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32),(7,7),1.2)
def warp(im,m,inverse=False):
    return cv2.warpAffine(im,m,(576,364),flags=cv2.INTER_LINEAR|(cv2.WARP_INVERSE_MAP if inverse else 0),borderMode=cv2.BORDER_REFLECT_101)
source=[cv2.imread(str(W/'action'/f'{n:04d}.png')) for n in range(76,173)]
generated=[cv2.imread(str(W/'generated_native_v3'/f'{n:04d}.png')) for n in range(76,173)]
ref=source[24]; gray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
union=np.zeros((364,576),np.float32)
for j in [0,12,24,36,48,60,72,84,96]:
    union=np.maximum(union,matte(source[j],16))
union=(union>.01).astype(np.uint8)*255
# The same reconstruction is used throughout the shot. Padding lets the cloning
# mask include actors touching the bottom edge without leaving an actor strip.
plate=cv2.resize(cv2.imread('/workspace/generated_images/exec-b35e0d07-a69b-47d1-9a36-b81a619b75ae.png'),(576,364))
pad=40
pp=cv2.copyMakeBorder(plate,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
rr=cv2.copyMakeBorder(ref,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
mm=cv2.copyMakeBorder(union,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
mm[:2]=0;mm[-2:]=0;mm[:,:2]=0;mm[:,-2:]=0
x,y,w,h=cv2.boundingRect(mm)
plate=cv2.seamlessClone(pp,rr,mm,(x+w//2,y+h//2),cv2.NORMAL_CLONE)[pad:-pad,pad:-pad]
cv2.imwrite(str(D/'single-cleanplate.png'),plate)
cv2.imwrite(str(D/'replacement-region.png'),union)
bg=(union==0).astype(np.uint8)*255
transforms=[]
for im in source:
    m=np.eye(2,3,dtype=np.float32)
    try:
        score,m=cv2.findTransformECC(gray,cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),m,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,45,1e-5),bg,5)
        if abs(m[0,2])>35 or abs(m[1,2])>35 or not .93<np.linalg.det(m[:,:2])<1.07:m=np.eye(2,3,dtype=np.float32)
    except cv2.error:m=np.eye(2,3,dtype=np.float32)
    transforms.append(m)
transforms=np.array(transforms)
for a in range(2):
    for b in range(3):transforms[:,a,b]=cv2.GaussianBlur(transforms[:,a,b][None,:],(0,0),1.4)[0]
det=cv2.FaceDetectorYN.create(str(R/'models/yunet.onnx'),'',(576,364),.55,.3,5000)
def eye(im):
    _,f=det.detect(im)
    if f is None:return None
    p=max(f,key=lambda p:p[2]*p[3]);return float((p[4]+p[6])/2)
dx=[]
for s,g in zip(source,generated):
    a,b=eye(s),eye(g);dx.append(np.nan if a is None or b is None else np.clip(a-b,-96,96))
dx=np.array(dx,np.float32);valid=np.isfinite(dx)
dx=np.interp(np.arange(len(dx)),np.flatnonzero(valid),dx[valid]).astype(np.float32)
dx=cv2.GaussianBlur(dx[None,:],(0,0),1.5)[0]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
head=Image.new('RGB',(1728,36),(18,20,25));draw=ImageDraw.Draw(head)
for x,t in [(12,'Original'),(588,'Versão anterior'),(1164,'Novo: laterais estáveis + transição a 60 FPS')]:draw.text((x,5),t,font=font,fill='white')
header=np.asarray(head)[...,::-1].copy()
def encoder(path,size):
    return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',size,'-r','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
comparison=encoder(OUT/'comparacao-laterais-corrigidas-60fps.mp4','1728x400')
standalone=encoder(OUT/'teste-laterais-corrigidas-60fps.mp4','576x854')
old=cv2.VideoCapture(str(W/'continuacao_corrigida_30fps.mp4'));old.set(cv2.CAP_PROP_POS_FRAMES,76)
rife=Rife(threads=2)
metrics=[];prev_bg=None
for j,g in enumerate(generated):
    ok,previous=old.read()
    if not ok:raise RuntimeError('Missing comparison frame')
    for half in range(2):
        k=min(j+1,96);f=half*.5
        fg=g if half==0 or j==96 else rife.between(g,generated[k],.5,('sides',j,k))
        alpha=matte(fg,1)
        shift=np.array([[1,0,(1-f)*dx[j]+f*dx[k]],[0,1,0]],np.float32)
        rgb=cv2.GaussianBlur(fg,(0,0),.55).astype(np.float32)*alpha[...,None]
        rgb=cv2.warpAffine(rgb,shift,(576,364),borderMode=cv2.BORDER_CONSTANT)
        alpha=cv2.warpAffine(alpha,shift,(576,364),borderMode=cv2.BORDER_CONSTANT)
        matrix=(1-f)*transforms[j]+f*transforms[k]
        back=warp(plate,matrix)
        fixed=np.clip(rgb+back*(1-alpha[...,None])+.5,0,255).astype(np.uint8)
        panel=np.empty((400,1728,3),np.uint8);panel[:36]=header
        panel[36:,:576]=source[j];panel[36:,576:1152]=previous[230:594];panel[36:,1152:]=fixed
        full=cv2.imread(str(W/'source'/f'{j+76:04d}.png'));full[230:594]=fixed
        comparison.stdin.write(panel.tobytes());standalone.stdin.write(full.tobytes())
        if half==0:
            cv2.imwrite(str(D/f'{j+76:04d}.png'),fixed)
            if j in [0,12,24,36,48,72,96]:cv2.imwrite(str(D/f'qa-{j+76:04d}.jpg'),panel)
        if prev_bg is not None:metrics.append(float(np.abs(back.astype(float)-prev_bg.astype(float)).mean()))
        prev_bg=back
    if j%12==0:print(f'{j}/97 native frames; {time.monotonic()-started:.1f}s',flush=True)
for p in [comparison,standalone]:
    p.stdin.close()
    if p.wait():raise RuntimeError('Encoding failed')
(D/'metrics.json').write_text(json.dumps({'frames':194,'fps':60,'seconds':194/60,'background_plates':1,'foreground_interpolation':'RIFE 4.25 Lite, adjacent native frames','max_background_step_MAE':max(metrics),'average_background_step_MAE':float(np.mean(metrics)),'limitations':'Original acting remains approximate; only source frames 76 to 172 treated by this test.'},indent=2))
print('Finished independent foreground and fixed background comparison.',flush=True)
