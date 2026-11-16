"""Constrain background tracking to translation and compare actual delivered 60FPS."""
from pathlib import Path
import subprocess,cv2,numpy as np,json,time
from composite_consistent_test import PersonMatte

R=Path(__file__).resolve().parent;W=R/'work';D=W/'clean_sides_v2';O=Path('/workspace/shared/downloads')
cv2.setNumThreads(2); segment=PersonMatte();start=time.monotonic()
plate=cv2.imread(str(D/'single-cleanplate.png'))
# Remove the narrow encoded black image border before moving the cleanplate.
# Replicating that border during an upward camera pan creates a black block.
plate=cv2.copyMakeBorder(plate[3:-3,3:-3],3,3,3,3,cv2.BORDER_REFLECT_101)
cv2.imwrite(str(D/'cleanplate-without-encoded-border.png'),plate)
cache=D/'review-mattes';cache.mkdir(exist_ok=True)
region=cv2.imread(str(D/'replacement-region.png'),0)
ref=cv2.imread(str(W/'action/0100.png'));gray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
bg=(region==0).astype(np.uint8)*255
panorama=cv2.imread('/workspace/generated_images/exec-34cced39-e1f6-4f17-98e1-a5b3d1da8dce.png')
panorama=cv2.resize(panorama,(832,620),interpolation=cv2.INTER_AREA)
panorama=cv2.GaussianBlur(panorama,(0,0),.8)
central=panorama[128:492,128:704]
glass=(bg>0)&(ref.min(2)>100)&(central.min(2)>100)
color_shift=np.clip(np.median(ref[glass].astype(float)-central[glass].astype(float),axis=0),-30,30)
panorama=np.clip(panorama.astype(float)+color_shift,0,255).astype(np.uint8)
cv2.imwrite(str(D/'panorama-for-final.png'),panorama)
motion=[]
points=cv2.goodFeaturesToTrack(gray,250,.004,5,mask=bg)
for n in range(76,173):
    im=cv2.imread(str(W/'action'/f'{n:04d}.png'));m=np.eye(2,3,dtype=np.float32)
    target=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    dst,ok,_=cv2.calcOpticalFlowPyrLK(gray,target,points,None,winSize=(31,31),maxLevel=4)
    rev,ok2,_=cv2.calcOpticalFlowPyrLK(target,gray,dst,None,winSize=(31,31),maxLevel=4)
    valid=(ok.ravel()==1)&(ok2.ravel()==1)&(np.linalg.norm((rev-points).reshape(-1,2),axis=1)<1.5)
    if valid.sum()>=6:
        fit,inliers=cv2.estimateAffinePartial2D(points[valid],dst[valid],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=1000)
        if fit is not None and inliers.sum()>=6:
            use=inliers.ravel()==1
            scale=float(np.clip(np.hypot(fit[0,0],fit[1,0]),.98,1.02))
            translation=np.median(dst[valid].reshape(-1,2)[use]-scale*points[valid].reshape(-1,2)[use],axis=0)
            m=np.array([[scale,0,translation[0]],[0,scale,translation[1]]],np.float32)
    motion.append(m)
motion=np.array(motion)
for a in range(2):
    for b in range(3):motion[:,a,b]=cv2.GaussianBlur(motion[:,a,b][None,:],(0,0),1.0)[0]
np.save(D/'camera-translation.npy',motion)
print('Background feature tracking completed; rotation and shear disabled.',flush=True)
def encoder(name,size):
    return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',size,'-r','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(O/name)],stdin=subprocess.PIPE)
wide=encoder('comparacao-fundo-e-bordas-60fps.mp4','1728x400')
vertical=encoder('comparacao-fundo-e-bordas-vertical.mp4','576x1200')
single=encoder('trecho-fundo-e-bordas-corrigidos.mp4','576x854')
candidate_path=W/'side_tests_intermediarios/comparacao-laterais-corrigidas-60fps.mp4'
if not candidate_path.exists():candidate_path=O/'comparacao-laterais-corrigidas-60fps.mp4'
cap=cv2.VideoCapture(str(candidate_path))
old=cv2.VideoCapture(str(O/'video-continuo-em-revisao-60fps.mp4'));old.set(cv2.CAP_PROP_POS_FRAMES,152)
steps=[];before=None
original_alpha=None
for t in range(194):
    ok,frame=cap.read();ok2,prior=old.read()
    if not ok or not ok2:raise RuntimeError(f'Missing review frame {t}')
    candidate=frame[36:,1152:]
    # Re-extract only the person. No camera-affine pixels are retained outside
    # this matte. RIFE interpolation of the person has already been completed.
    alpha_file=cache/f'foreground-{t:04d}.npy'
    if alpha_file.exists():alpha=np.load(alpha_file)
    else:
        p,_=segment(candidate)
        mask=(p>.20).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        alpha=cv2.GaussianBlur(mask.astype(np.float32),(5,5),.9)
        np.save(alpha_file,alpha)
    j=t//2;k=min(j+1,96);f=.5*(t%2)
    matrix=(1-f)*motion[j]+f*motion[k]
    panoramic_matrix=matrix.copy()
    panoramic_matrix[:,2]-=matrix[:,:2]@np.array([128,128],np.float32)
    back=cv2.warpAffine(panorama,panoramic_matrix,(576,364),borderMode=cv2.BORDER_REFLECT_101)
    source=cv2.imread(str(W/'source'/f'{j+76:04d}.png'))
    if t%2==0:
        removal_file=cache/f'removal-{j+76:04d}.npy'
        if removal_file.exists():original_alpha=np.load(removal_file)
        else:
            person,_=segment(source[230:594])
            removal=(person>.10).astype(np.uint8)
            removal=cv2.morphologyEx(removal,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
            removal=cv2.dilate(removal,np.ones((29,29),np.uint8))
            original_alpha=cv2.GaussianBlur(removal.astype(np.float32),(11,11),2)
            np.save(removal_file,original_alpha)
    source_background=source[230:594]
    source_removal=original_alpha
    if t%2:
        # Transport the actual visible scenery at the half-frame camera pose;
        # holding its 30FPS pixels would introduce a half-step at the matte edge.
        current=np.vstack([motion[j],[0,0,1]])
        intermediate=np.vstack([matrix,[0,0,1]])
        relative=(intermediate@np.linalg.inv(current))[:2].astype(np.float32)
        source_background=cv2.warpAffine(source_background,relative,(576,364),borderMode=cv2.BORDER_REPLICATE)
        source_removal=cv2.warpAffine(original_alpha,relative,(576,364),borderMode=cv2.BORDER_REPLICATE)
    # Keep actual source scenery outside the removed actor. This also supplies
    # camera-revealed areas beyond the fixed plate's reference crop.
    back=np.clip(source_background*(1-source_removal[...,None])+back*source_removal[...,None],0,255).astype(np.uint8)
    fixed=np.clip(candidate.astype(np.float32)*alpha[...,None]+back*(1-alpha[...,None])+.5,0,255).astype(np.uint8)
    frame[36:,1152:]=fixed
    prior=cv2.resize(prior,(576,854),interpolation=cv2.INTER_AREA)
    frame[36:,576:1152]=prior[230:594]
    full=source.copy();full[230:594]=fixed
    stacked=np.concatenate([frame[:,:576],frame[:,576:1152],frame[:,1152:]],axis=0)
    wide.stdin.write(frame.tobytes());vertical.stdin.write(stacked.tobytes());single.stdin.write(full.tobytes())
    if t in [0,24,48,96,144,192]:cv2.imwrite(str(D/f'final-qa-{j+76:04d}.jpg'),frame)
    if before is not None:steps.append(float(np.abs(back.astype(float)-before.astype(float)).mean()))
    before=back
    if t%30==0:print(f'Review {t}/194; {time.monotonic()-start:.1f}s',flush=True)
if cap.read()[0]:raise RuntimeError('Unexpected extra frames')
for proc in [wide,vertical,single]:
    proc.stdin.close()
    if proc.wait():raise RuntimeError('Review encoder failed')
report={'frames':194,'fps':60,'seconds':194/60,'audio':False,'source_interval_frames':[76,173],'background_plates':1,'camera_model':'source background feature tracking; translation and uniform scale; no rotation or shear','background_compositing':'original source pixels outside old actor; fixed cleanplate inside removal region','foreground_interpolation':'adjacent-frame RIFE 4.25 Lite','comparison_previous':'actual delivered 60FPS file; source frames 152 to 345','background_step_mean_MAE':float(np.mean(steps)),'background_step_max_MAE':float(max(steps)),'limitations':'This is a 3.233s test. Acting is approximate. Does not replace the entire 25.933s video.'}
(D/'final-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('Review exports completed',flush=True)
