"""Complete silent full-length edit from approved character assets and source motion."""
from pathlib import Path
from bisect import bisect_left
import json,subprocess,time
import cv2,numpy as np
from PIL import Image
from composite_consistent_test import PersonMatte
from render_source import SourceMotion
from render_rife_backend import Rife

R=Path(__file__).resolve().parent;W=R/'work';D=W/'delivery';O=Path('/workspace/shared/downloads')
CUTS=[0,39,76,173,241,272,349,404,666,749,778]
NATIVE=D/'complete-native-30fps.mp4';FINAL=O/'video-marvel-completo-60fps.mp4'
for name in ['raw','source-mattes','qa','camera']: (D/name).mkdir(exist_ok=True,parents=True)
cv2.setNumThreads(2);segment=PersonMatte();rife=Rife(threads=2)
flow=SourceMotion(W/'action',W/'flow_full',384,'medium')
det=cv2.FaceDetectorYN.create(str(R/'models/yunet.onnx'),'',(576,364),.40,.3,5000)
grid=np.dstack(np.meshgrid(np.arange(576,dtype=np.float32),np.arange(364,dtype=np.float32)))
started=time.monotonic()
def log(t):print(f'{time.monotonic()-started:.1f}s {t}',flush=True)
def src(n):return cv2.imread(str(W/'action'/f'{n:04d}.png'))
def gray(im):return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
def probability(im):
    p,_=segment(cv2.resize(im,(384,243)))
    return cv2.resize(p,(576,364))
def clean_frame(im):
    im=im.copy();im[-4:]=im[-5:-4];im[:2]=im[2:3]
    im[:,:2]=im[:,2:3];im[:,-2:]=im[:,-3:-2]
    return im
def mask(im,removal=False,point=None,hulk=False):
    p=probability(im);m=(p>(.10 if removal else .18)).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9 if removal else 5,)*2,np.uint8))
    if point is not None:
        count,labels,stats,centers=cv2.connectedComponentsWithStats(m)
        choices=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>250]
        if choices:
            chosen=min(choices,key=lambda i:np.linalg.norm(centers[i]-point)-.0007*stats[i,cv2.CC_STAT_AREA])
            m=(labels==chosen).astype(np.uint8)
    if removal:m=cv2.dilate(m,np.ones((27,27),np.uint8))
    else:
        small=cv2.resize(im,(384,243));pp=cv2.resize(p,(384,243));hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        labels=np.full((243,384),cv2.GC_PR_BGD,np.uint8)
        labels[pp<.025]=cv2.GC_BGD;labels[pp>.22]=cv2.GC_PR_FGD
        white=(small.min(2)>155)&(hsv[...,1]<55)
        core=cv2.erode((pp>.92).astype(np.uint8),np.ones((3,3),np.uint8))>0
        labels[core&~white]=cv2.GC_FGD
        # Add the green background character's fingers when the semantic model
        # does not recognize them as part of a human silhouette.
        green=(hsv[...,0]>=28)&(hsv[...,0]<=95)&(hsv[...,1]>30)&(small[...,1]>small[...,2].astype(float)*1.025)
        green[:,190:]=False
        if hulk:labels[green]=cv2.GC_FGD
        if np.any(labels==cv2.GC_FGD) and np.any(labels==cv2.GC_BGD):
            cv2.grabCut(small,labels,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),2,cv2.GC_INIT_WITH_MASK)
            m=cv2.resize(((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)).astype(np.uint8),(576,364),interpolation=cv2.INTER_NEAREST)
        count,components,stats,_=cv2.connectedComponentsWithStats(m)
        keep=sorted(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_AREA],reverse=True)[:2 if hulk else 1]
        m=np.isin(components,keep).astype(np.uint8)
        bottom=np.max(m[-20:-8],axis=0)>0;m[-10:,bottom]=1
    return cv2.GaussianBlur(m.astype(np.float32),(5,5),.9)
def original_mask(n):
    p=D/'source-mattes'/f'{n:04d}.npy'
    if p.exists():a=np.load(p).astype(np.float32)/255
    else:
        a=mask(src(n),True,point=np.array([400,210]) if 272<=n<349 else None)
        np.save(p,np.rint(a*255).astype(np.uint8))
    extra=np.zeros_like(a)
    if n<39:extra[135:257,125:260]=1;extra[135:257,410:520]=1
    if 173<=n<241:extra[30:240,:135]=1
    if 272<=n<349:extra[295:364,200:575]=1
    return np.maximum(a,cv2.GaussianBlur(extra,(15,15),3))
def face(im,near=None):
    _,f=det.detect(im)
    if f is None:return None
    if near is None:p=max(f,key=lambda v:v[2]*v[3])
    else:p=min(f,key=lambda v:np.linalg.norm((v[:2]+v[2:4]/2)-near[:2])/140+abs(np.log(max(v[2],1)/max(near[2],1))))
    return np.array([(p[4]+p[6])/2,(p[5]+p[7])/2,p[2],p[3]],np.float32)
def remap(im,m):return cv2.remap(im,m[...,0],m[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
def smooth(a,sigma=1.2):
    a=np.asarray(a,np.float32)
    for k in range(a.shape[1]):
        valid=np.isfinite(a[:,k])
        a[:,k]=np.interp(np.arange(len(a)),np.flatnonzero(valid),a[valid,k]) if valid.any() else 0
        a[:,k]=cv2.GaussianBlur(a[:,k][None],(0,0),sigma)[0]
    return a

anchors={int(p.stem):p for p in (W/'anchors').glob('*.png')};keys=sorted(anchors)
edge_maps={}
for a,targets in [(706,list(range(666,706)))]:
    log(f'Following source motion for {len(targets)} edge frames')
    edge_maps.update({i:(a,f) for i,(f,_) in flow.anchor_maps(a,targets).items()})
def raw(n):
    for p in [D/'raw'/f'{n:04d}.png',W/'generated_native_v3'/f'{n:04d}.png',W/'review_rife'/f'{2*n:04d}.png']:
        if p.exists():return clean_frame(cv2.imread(str(p)))
    if n in anchors:g=cv2.resize(cv2.imread(str(anchors[n])),(576,364))
    elif n in edge_maps:
        a,f=edge_maps[n];g=flow.warp(raw(a),f)
    else:
        j=bisect_left(keys,n);a,b=keys[j-1:j+1]
        g=rife.between(raw(a),raw(b),(n-a)/(b-a),('delivery',a,b))
    cv2.imwrite(str(D/'raw'/f'{n:04d}.png'),g,[cv2.IMWRITE_PNG_COMPRESSION,2]);return g

sheet=Image.open('/workspace/generated_images/exec-889cdef7-b715-4ce3-b16c-353734ce59d6.png').convert('RGB')
plates={}
for j,n in enumerate([0,210,302,500]):
    x,y=j%2,j//2
    p=sheet.crop((round(x*sheet.width/2)+2,round(y*sheet.height/2)+2,round((x+1)*sheet.width/2)-2,round((y+1)*sheet.height/2)-2)).resize((576,364),Image.Resampling.LANCZOS)
    plates[n]=cv2.cvtColor(np.asarray(p),cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(D/f'cleanplate-{n:04d}.png'),plates[n])
panorama=cv2.resize(cv2.imread('/workspace/generated_images/exec-34cced39-e1f6-4f17-98e1-a5b3d1da8dce.png'),(832,620),interpolation=cv2.INTER_AREA)
panorama=cv2.GaussianBlur(panorama,(0,0),.8)
central=panorama[128:492,128:704];ref100=src(100);bg=(original_mask(100)<.01)&(ref100.min(2)>100)&(central.min(2)>100)
panorama=np.clip(panorama.astype(float)+np.clip(np.median(ref100[bg].astype(float)-central[bg],axis=0),-30,30),0,255).astype(np.uint8)

def pair_camera(a,b,ma,mb):
    valid=((ma<.01)*255).astype(np.uint8)
    pts=cv2.goodFeaturesToTrack(gray(a),220,.005,5,mask=valid)
    if pts is None or len(pts)<5:return np.eye(3,dtype=np.float32)
    dst,ok,_=cv2.calcOpticalFlowPyrLK(gray(a),gray(b),pts,None,winSize=(25,25),maxLevel=4)
    rev,ok2,_=cv2.calcOpticalFlowPyrLK(gray(b),gray(a),dst,None,winSize=(25,25),maxLevel=4)
    q=dst[:,0];ix=np.clip(np.rint(q[:,0]).astype(int),0,575);iy=np.clip(np.rint(q[:,1]).astype(int),0,363)
    good=(ok.ravel()==1)&(ok2.ravel()==1)&(np.linalg.norm((rev-pts)[:,0],axis=1)<1.5)&(mb[iy,ix]<.02)
    if good.sum()<5:return np.eye(3,dtype=np.float32)
    fit,inl=cv2.estimateAffinePartial2D(pts[good],dst[good],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=500)
    if fit is None or inl.sum()<4:return np.eye(3,dtype=np.float32)
    scale=float(np.clip(np.hypot(fit[0,0],fit[1,0]),.97,1.03));use=inl.ravel()==1
    delta=np.median(dst[good,0][use]-scale*pts[good,0][use],axis=0)
    return np.array([[scale,0,delta[0]],[0,scale,delta[1]],[0,0,1]],np.float32)
def cross_glass(reference):
    if reference==100:return np.eye(3,dtype=np.float32)
    sift=cv2.SIFT_create(nfeatures=1800)
    ka,da=sift.detectAndCompute(gray(src(100)),((original_mask(100)<.01)*255).astype(np.uint8))
    kb,db=sift.detectAndCompute(gray(src(reference)),((original_mask(reference)<.01)*255).astype(np.uint8))
    identity=np.eye(3,dtype=np.float32)
    if da is None or db is None:return identity
    matches=cv2.BFMatcher().knnMatch(da,db,k=2);good=[a for a,b in matches if a.distance<.72*b.distance]
    if len(good)<5:return identity
    aa=np.float32([ka[m.queryIdx].pt for m in good]);bb=np.float32([kb[m.trainIdx].pt for m in good])
    fit,inside=cv2.estimateAffinePartial2D(aa,bb,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000)
    if fit is None or inside.sum()<5:return identity
    scale=np.hypot(fit[0,0],fit[1,0]);angle=np.degrees(np.arctan2(fit[1,0],fit[0,0]))
    if not (.75<scale<1.4 and abs(angle)<3 and max(abs(fit[:,2]))<200):return identity
    use=inside.ravel()==1;shift=np.median(bb[use]-scale*aa[use],axis=0)
    return np.array([[scale,0,shift[0]],[0,scale,shift[1]],[0,0,1]],np.float32)
def camera_sequence(start,end,reference,glass=False):
    path=D/'camera'/f'{start:04d}.npy'
    if path.exists():return np.load(path)
    global_m=[np.eye(3,dtype=np.float32)];previous=src(start);ma=original_mask(start)
    for n in range(start+1,end):
        current=src(n);mb=original_mask(n)
        global_m.append(pair_camera(previous,current,ma,mb)@global_m[-1]);previous=current;ma=mb
    offset=np.linalg.inv(global_m[reference-start]);atlas=cross_glass(reference) if glass else np.eye(3,dtype=np.float32)
    result=np.stack([m@offset@atlas for m in global_m])
    flattened=smooth(result[:,:2].reshape(len(result),6),.8)
    result[:,:2]=flattened.reshape(len(result),2,3)
    np.save(path,result);return result

def align_offsets(start,end):
    offsets=[]
    for n in range(start,end):
        original=face(src(n));cg=face(raw(n),original) if original is not None else None
        if original is None or cg is None:offsets.append([np.nan,np.nan])
        else:offsets.append(np.clip(original[:2]-cg[:2],[-100,-64],[100,64]))
    return smooth(offsets,1.4)
def translate_layer(im,alpha,dx,dy):
    premul=cv2.GaussianBlur(im,(0,0),.60).astype(np.float32)*alpha[...,None]
    p=96;premul=cv2.copyMakeBorder(premul,p,p,p,p,cv2.BORDER_REPLICATE)
    alpha=cv2.copyMakeBorder(alpha,p,p,p,p,cv2.BORDER_REPLICATE)
    matrix=np.array([[1,0,dx-p],[0,1,dy-p]],np.float32)
    return cv2.warpAffine(premul,matrix,(576,364)),cv2.warpAffine(alpha,matrix,(576,364))

def actor_transforms(reference,start,end):
    im=src(reference);f=face(im);region=np.zeros((364,576),np.uint8)
    if f is not None:
        x,y=f[:2];ww,hh=f[2:];region[max(0,int(y-hh*.6)):min(364,int(y+hh*.7)),max(0,int(x-ww*.6)):min(576,int(x+ww*.6))]=255
    else:region=(original_mask(reference)*255).astype(np.uint8)
    pts=cv2.goodFeaturesToTrack(gray(im),80,.01,4,mask=region)
    transforms=[]
    for n in range(start,end):
        target=src(n);fit=None
        if pts is not None:
            dst,ok,_=cv2.calcOpticalFlowPyrLK(gray(im),gray(target),pts,None,winSize=(31,31),maxLevel=4)
            valid=ok.ravel()==1
            if valid.sum()>=5:fit,inliers=cv2.estimateAffinePartial2D(pts[valid],dst[valid],method=cv2.RANSAC,ransacReprojThreshold=3)
        if fit is None:fit=np.eye(2,3,dtype=np.float32)
        scale=np.hypot(fit[0,0],fit[1,0]);angle=np.degrees(np.arctan2(fit[1,0],fit[0,0]))
        if not(.75<scale<1.3 and abs(angle)<15 and max(abs(fit[:,2]))<150):fit=np.eye(2,3,dtype=np.float32)
        transforms.append(fit.reshape(-1))
    return smooth(transforms,1.0).reshape(-1,2,3)

log('Preparing Doom motion from the approved single frame')
doom_base=clean_frame(cv2.resize(cv2.imread('/workspace/generated_images/exec-bf3d6e55-f887-4ac8-a0c8-5b855eca5571.png'),(576,364)))
doom_alpha=np.load(D/'doom-full-alpha.npy');doom_premul=cv2.GaussianBlur(doom_base,(0,0),.6).astype(np.float32)*doom_alpha[...,None]
doom_motion=actor_transforms(302,272,349)
doom_maps=flow.anchor_maps(302,list(range(272,349)))
def doom_layer(n):
    inv=cv2.invertAffineTransform(doom_motion[n-272]);rigid=grid@inv[:,:2].T+inv[:,2]
    ff=cv2.resize(doom_maps[n][0],(576,364));ff[...,0]*=576/384;ff[...,1]*=364/243
    source_map=grid+ff
    original_body=np.load(D/'source-mattes/0302.npy').astype(np.float32)/255
    body=np.clip((rigid[...,1]-145)/75,0,1)*remap(original_body,source_map)*.55
    combined=rigid*(1-body[...,None])+source_map*body[...,None]
    return remap(doom_premul,combined),remap(doom_alpha,combined)

ending_base=raw(271);ending_alpha=mask(ending_base)
ending_contour=np.array([[224,0],[327,0],[350,27],[358,90],[356,130],[342,154],[397,177],[363,240],[408,255],[457,280],[485,310],[510,364],[25,364],[34,323],[66,299],[118,273],[128,218],[145,166],[173,155],[173,112],[164,99],[172,67],[181,41],[202,19]],np.int32)
ending_limit=np.zeros((364,576),np.float32);cv2.fillPoly(ending_limit,[ending_contour],1)
ending_alpha*=cv2.GaussianBlur(ending_limit,(5,5),.8)
cg_face=face(ending_base);old_face=face(src(764))
scale=1.0 if cg_face is None or old_face is None else float(np.clip(old_face[2]/cg_face[2],.8,1.4))
translation=np.zeros(2) if cg_face is None or old_face is None else old_face[:2]-scale*cg_face[:2]
fit=np.array([[scale,0,translation[0]],[0,scale,translation[1]]],np.float32)
mono=cv2.cvtColor(gray(ending_base),cv2.COLOR_GRAY2BGR).astype(np.float32)*ending_alpha[...,None]
ending_rgb=cv2.warpAffine(cv2.copyMakeBorder(mono,0,100,0,0,cv2.BORDER_REPLICATE),fit,(576,364))
ending_a=cv2.warpAffine(cv2.copyMakeBorder(ending_alpha,0,100,0,0,cv2.BORDER_REPLICATE),fit,(576,364))
ending_motion=actor_transforms(764,749,778)

log('Preparing source backgrounds and replacement motion')
shot_info={0:(0,False),3:(210,False),4:(241,True),5:(302,False),6:(349,True),7:(500,False),8:(706,True)}
cameras={};offsets={}
for shot,(reference,glass) in shot_info.items():
    a,b=CUTS[shot:shot+2];end=min(b,747)
    cameras[shot]=camera_sequence(a,b,reference,glass)
    if shot!=5:offsets[shot]=align_offsets(a,end)
    log(f'Prepared shot {shot+1}/10; source frames {a}..{b-1}')

stand=cv2.VideoCapture(str(O/'teste-reacao-wanda-mais-denso.mp4'))
side=cv2.VideoCapture(str(O/'trecho-fundo-e-bordas-corrigidos.mp4'))
side_frames=[]
for j in range(194):
    ok,im=side.read()
    if not ok:raise RuntimeError('Missing approved side-correction frame')
    if j%2==0:side_frames.append(im)
side.release()
encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','576x854','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','16','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(NATIVE)],stdin=subprocess.PIPE)
shot=0;last_regular=None
for n in range(778):
    if n>=CUTS[shot+1]:shot+=1
    full=cv2.imread(str(W/'source'/f'{n:04d}.png'));original=full[230:594]
    if shot==1:
        ok,full=stand.read()
        if not ok:raise RuntimeError('Missing standing-motion patch')
    elif shot==2:full=side_frames[n-76].copy()
    elif shot==9:
        matrix=ending_motion[n-749]
        premul=cv2.warpAffine(ending_rgb,matrix,(576,364),borderMode=cv2.BORDER_REPLICATE)
        alpha=cv2.warpAffine(ending_a,matrix,(576,364),borderMode=cv2.BORDER_REPLICATE)
        removed=original_mask(n)
        bg=original.astype(np.float32)*(1-removed[...,None])+np.array([5,5,5])*removed[...,None]
        fixed=premul+bg*(1-alpha[...,None])
        hsv=cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
        lights=((hsv[...,1]>90)&(hsv[...,2]>50)).astype(np.float32)
        lights=cv2.GaussianBlur(lights,(5,5),.7)
        fixed=fixed*(1-lights[...,None])+original*lights[...,None]
        full[230:594]=np.clip(fixed,0,255).astype(np.uint8)
    elif n in [747,748]:
        hsv=cv2.cvtColor(original,cv2.COLOR_BGR2HSV);a=np.clip((hsv[...,2].astype(float)-160)/80,0,1)
        full[230:594]=np.clip(last_regular*(1-a[...,None])+original*a[...,None],0,255).astype(np.uint8)
    else:
        reference,glass=shot_info[shot];matrix=cameras[shot][n-CUTS[shot]][:2]
        if glass:
            matrix=matrix.copy();matrix[:,2]-=matrix[:,:2]@np.array([128,128],np.float32)
            plate=panorama
        else:plate=plates[reference]
        back=cv2.warpAffine(plate,matrix,(576,364),borderMode=cv2.BORDER_REFLECT_101)
        removal=original_mask(n)
        bg=original.astype(np.float32)*(1-removal[...,None])+back*removal[...,None]
        if shot==5:premul,alpha=doom_layer(n)
        else:
            generated=raw(n);alpha=mask(generated,hulk=(shot==3))
            dx,dy=offsets[shot][n-CUTS[shot]]
            premul,alpha=translate_layer(generated,alpha,dx,dy)
        fixed=np.clip(premul+bg*(1-alpha[...,None])+.5,0,255).astype(np.uint8)
        full[230:594]=fixed;last_regular=fixed
    if n in [0,20,38,39,55,75,76,100,172,173,210,240,241,260,271,272,282,302,322,348,349,375,403,404,444,500,575,600,623,624,665,666,706,746,748,749,764,777]:
        cv2.imwrite(str(D/'qa'/f'{n:04d}.jpg'),full,[cv2.IMWRITE_JPEG_QUALITY,95])
    encoder.stdin.write(full.tobytes())
    if n%30==0:log(f'Render {n}/778; shot {shot+1}')
encoder.stdin.close()
if encoder.wait():raise RuntimeError('Native encoder failed')
stand.release()
report={'duration':778/30,'source_frames':778,'output_frames':1556,'output_fps':60,'audio':False,'characters':['Doctor Strange','Wanda','Doctor Doom','Hulk','Sylvie'],'original_actor_shots_pending':[],'source_motion_single_image_ranges':{'Doom':[272,349],'Doctor Strange finale':[749,778]},'limitations':'Doom and the final portrait use motion transfer from single approved character images; gestures and facial acting remain approximate. Existing dense standing-motion and corrected-window-background patches retained.'}
(D/'delivery-report.json').write_text(json.dumps(report,indent=2))
log('Native complete; exporting all 1556 frames at 60 FPS')
subprocess.run(['python',str(R/'export_adjacent60.py'),str(NATIVE),str(FINAL)],check=True)
log(f'Complete: {FINAL}')
