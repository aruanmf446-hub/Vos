"""Stable original-video background mosaic plus independent character alpha."""
from pathlib import Path
import json,subprocess,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from composite_consistent_test import PersonMatte

R=Path(__file__).resolve().parent;W=R/'work';D=W/'clean_sides_test';D.mkdir(exist_ok=True)
cv2.setNumThreads(2);segment=PersonMatte()
def matte(im):
    p,b=segment(cv2.resize(im,(384,243)))
    p=cv2.resize(p,(576,364));m=(p>.18).astype(np.uint8)
    m=cv2.dilate(m,np.ones((9,9),np.uint8));return cv2.GaussianBlur(m.astype(np.float32),(7,7),0)
def warp(im,m,inv=True):
    return cv2.warpAffine(im,m,(576,364),flags=cv2.INTER_LINEAR|(cv2.WARP_INVERSE_MAP if inv else 0),borderMode=cv2.BORDER_REFLECT_101)
ref=cv2.imread(str(W/'action/0100.png'));refmask=matte(ref);refgray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
generated_plate=cv2.imread('/workspace/generated_images/exec-b35e0d07-a69b-47d1-9a36-b81a619b75ae.png')
generated_plate=cv2.resize(generated_plate,(576,364))
def camera(src,mask):
    mat=np.eye(2,3,dtype=np.float32);bg=((refmask<.02)&(mask<.02)).astype(np.uint8)*255
    try:
        score,m=cv2.findTransformECC(refgray,cv2.cvtColor(src,cv2.COLOR_BGR2GRAY),mat,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-5),bg,5)
        if abs(m[0,2])<18 and abs(m[1,2])<18 and .96<np.linalg.det(m[:,:2])<1.04:return m
    except cv2.error:pass
    return mat
print('Building background mosaic from original visible pixels',flush=True)
samples=[]
for n in sorted(set([76,88,100,112,124,136,148,160,172])):
    src=cv2.imread(str(W/'action'/f'{n:04d}.png'));mask=matte(src);m=camera(src,mask)
    aligned=warp(src,m).astype(np.float32);valid=warp(mask,m)<.02
    aligned[~valid]=np.nan;samples.append(aligned)
stack=np.stack(samples)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning);observed=np.nanmedian(stack,axis=0)
known=np.isfinite(observed).all(2)
plate=generated_plate.copy();plate[known]=np.clip(observed[known],0,255).astype(np.uint8)
plate[refmask<.02]=ref[refmask<.02]
cv2.imwrite(str(D/'fixed-original-background.png'),plate)
cv2.imwrite(str(D/'observed-background-coverage.png'),known.astype(np.uint8)*255)
detector=cv2.FaceDetectorYN.create(str(R/'models/yunet.onnx'),'',(576,364),.55,.3,5000)
def eye(im):
    _,f=detector.detect(im)
    if f is None:return None
    p=max(f,key=lambda r:r[2]*r[3]);return float((p[4]+p[6])/2)
records=[];inputs=[]
for n in range(76,173):
    src=cv2.imread(str(W/'action'/f'{n:04d}.png'));gen=cv2.imread(str(W/'generated_native_v3'/f'{n:04d}.png'))
    if gen is None:raise RuntimeError(f'Missing raw generated frame {n}')
    a,b=eye(src),eye(gen);records.append(0 if a is None or b is None else np.clip(a-b,-96,96));inputs.append((src,gen))
dx=cv2.GaussianBlur(np.array(records,np.float32)[None,:],(0,0),1.0)[0]
def enc(path,size):return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',size,'-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
out=Path('/workspace/shared/downloads');e=enc(out/'comparacao-laterais-e-fundo-continuo.mp4','1728x400');f=enc(out/'teste-laterais-fundo-estavel.mp4','576x854')
head=Image.new('RGB',(1728,36),(18,20,25));draw=ImageDraw.Draw(head);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
for x,text in [(12,'Original'),(588,'Versão anterior'),(1164,'Novo: fundo contínuo + personagem separado')]:draw.text((x,5),text,font=font,fill='white')
header=np.asarray(head)[...,::-1].copy();old=cv2.VideoCapture(str(W/'continuacao_corrigida_30fps.mp4'));old.set(cv2.CAP_PROP_POS_FRAMES,76)
source_masks=[]
for j,(src,gen) in enumerate(inputs):
    n=j+76;oldmask=matte(src);alpha=matte(gen)
    m=camera(src,oldmask);back=warp(plate,m,False)
    base=src.astype(np.float32)*(1-oldmask[...,None])+back.astype(np.float32)*oldmask[...,None]
    shift=np.array([[1,0,dx[j]],[0,1,0]],np.float32)
    premul=cv2.GaussianBlur(gen,(0,0),.45).astype(np.float32)*alpha[...,None]
    fg=cv2.warpAffine(premul,shift,(576,364),borderMode=cv2.BORDER_CONSTANT)
    a=cv2.warpAffine(alpha,shift,(576,364),borderMode=cv2.BORDER_CONSTANT)
    fixed=np.clip(fg+base*(1-a[...,None]),0,255).astype(np.uint8)
    ok,previous=old.read()
    if not ok:raise RuntimeError('Missing reference video frame')
    panel=np.empty((400,1728,3),np.uint8);panel[:36]=header;panel[36:,:576]=src;panel[36:,576:1152]=previous[230:594];panel[36:,1152:]=fixed
    full=cv2.imread(str(W/'source'/f'{n:04d}.png'));full[230:594]=fixed
    e.stdin.write(panel.tobytes());f.stdin.write(full.tobytes())
    if n in [76,88,100,112,124,148,172]:cv2.imwrite(str(D/f'{n:04d}.jpg'),panel)
    if j%20==0:print(f'{j}/97 frames',flush=True)
for proc in [e,f]:
    proc.stdin.close()
    if proc.wait():raise RuntimeError('Encoder failed')
old.release()
(D/'metrics.json').write_text(json.dumps({'frames':97,'fps':30,'duration':97/30,'observed_original_background_fraction':float(known.mean()),'foreground_x_shift_min':float(dx.min()),'foreground_x_shift_max':float(dx.max()),'no_generated_background_used_per_frame':True},indent=2))
print('Stable background and independent-character comparison finished.',flush=True)
