"""Camera-locked background atlas, registered foreground, recurrent matting.

The background never passes through character-frame interpolation. Character
position and scale are registered in scene coordinates before layered 60FPS.
"""
from pathlib import Path
import sys,argparse,json,time,subprocess
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont

R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
args=argparse.ArgumentParser();args.add_argument('--full',action='store_true');args.add_argument('--only-shot',type=int);args=args.parse_args()
prefix=(R/'deliver_full_video.py').read_text().split("log('Preparing Doom motion")[0]
exec(compile(prefix,str(R/'deliver_full_video.py'),'exec'),globals())
V=W/'continuity-v2';V.mkdir(exist_ok=True)
for name in ['raw-overrides','atlases','canonical','qa','tracks']: (V/name).mkdir(exist_ok=True)
torch.set_num_threads(2);cv2.setNumThreads(2)
rvm=torch.jit.load(str(R/'models/rvm/rvm_mobilenetv3_fp32.torchscript'),map_location='cpu').eval()
t0=time.monotonic()
def status(s):print(f'{time.monotonic()-t0:.1f}s {s}',flush=True)
def picture(n):
    p=V/'raw-overrides'/f'{n:04d}.png'
    if p.exists():return clean_frame(cv2.imread(str(p)))
    if 575<=n<=580:return clean_frame(cv2.imread(str(V/'dense-anchors'/f'{n:04d}.png')))
    return raw(n)

# New genuine images cover the first six consecutive frames. The blocked
# continuation is bridged using existing approved poses, explicitly recorded.
closed=clean_frame(cv2.imread(str(V/'dense-anchors/0580.png')))
relaxed=raw(624)
for n in [581,582]:cv2.imwrite(str(V/'raw-overrides'/f'{n:04d}.png'),closed)
for n in [583,584,585]:
    p=V/'raw-overrides'/f'{n:04d}.png'
    if not p.exists():cv2.imwrite(str(p),rife.between(closed,relaxed,(n-582)/4,('blink-reopen',580,624)))
cv2.imwrite(str(V/'raw-overrides/0586.png'),relaxed)
for n in range(587,594):
    p=V/'raw-overrides'/f'{n:04d}.png'
    if not p.exists():cv2.imwrite(str(p),rife.between(relaxed,raw(594),(n-586)/8,('blink-return',624,594)))

SHOT={2:(76,173,100),3:(173,241,210),4:(241,272,241),5:(272,349,302),6:(349,404,349),7:(404,666,500),8:(666,747,706)}
def camera_raw(start,end,reference):
    path=V/'tracks'/f'camera-{start:04d}.npy'
    if path.exists():return np.load(path)
    matrices=[np.eye(3,dtype=np.float32)];prev=src(start);old=original_mask(start)
    for n in range(start+1,end):
        cur=src(n);new=original_mask(n)
        matrices.append(pair_camera(prev,cur,old,new)@matrices[-1]);prev=cur;old=new
    correction=np.linalg.inv(matrices[reference-start])
    result=np.stack([m@correction for m in matrices]).astype(np.float32)
    np.save(path,result);return result

def initial_plate(shot,ref):
    if shot in [2,4,6,8]:
        if shot==2:m=np.eye(3,dtype=np.float32)
        else:
            existing=np.load(D/'camera'/f'{SHOT[shot][0]:04d}.npy');m=existing[ref-SHOT[shot][0]].copy()
        transform=m[:2].copy();transform[:,2]-=transform[:,:2]@np.array([128,128],np.float32)
        return cv2.warpAffine(panorama,transform,(576,364),borderMode=cv2.BORDER_REFLECT_101)
    return plates[ref].copy()

def calibrate(ref,plate):
    original=src(ref);remove=original_mask(ref)
    valid=((remove<.01)*255).astype(np.uint8);valid[:4]=0;valid[-4:]=0
    matrix=np.eye(2,3,dtype=np.float32)
    try:
        score,fit=cv2.findTransformECC(gray(original),gray(plate),matrix,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,80,1e-5),valid,5)
        sx=np.linalg.norm(fit[:,0]);sy=np.linalg.norm(fit[:,1]);angle=np.degrees(np.arctan2(fit[1,0],fit[0,0]))
        if .85<sx<1.15 and .85<sy<1.15 and abs(angle)<3 and max(abs(fit[:,2]))<60:matrix=fit
    except cv2.error:score=None
    aligned=cv2.warpAffine(plate,matrix,(576,364),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT_101)
    unknown=remove>.02
    filled=original.copy();filled[unknown]=aligned[unknown]
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    residual=dis.calc(gray(filled),gray(aligned),None)
    hole=(unknown*255).astype(np.uint8)
    for k in range(2):
        residual[...,k]=cv2.inpaint(np.clip(residual[...,k],-14,14).astype(np.float32),hole,5,cv2.INPAINT_NS)
    residual=cv2.GaussianBlur(residual,(0,0),2.5)
    corrected=cv2.remap(aligned,grid[...,0]+residual[...,0],grid[...,1]+residual[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    blend=cv2.GaussianBlur((remove>.01).astype(np.float32),(17,17),3)
    clean=np.clip(original*(1-blend[...,None])+corrected*blend[...,None],0,255).astype(np.uint8)
    return clean,remove,score

def build_atlas(shot,cameras):
    start,end,ref=SHOT[shot];file=V/'atlases'/f'{shot}-continuous-plate.npz'
    if file.exists():
        a=np.load(file);return a['image'],a['to_world']
    if shot in [2,4,6,8]:
        atlas=panorama.copy()
        m=np.eye(3,dtype=np.float32) if shot==2 else np.load(D/'camera'/f'{start:04d}.npy')[ref-start].copy()
        to_world=m@np.array([[1,0,-128],[0,1,-128],[0,0,1]],np.float32)
    else:atlas=plates[ref].copy();to_world=np.eye(3,dtype=np.float32)
    reference=src(ref);valid=((original_mask(ref)<.01)*255).astype(np.uint8)
    valid[:4]=0;valid[-4:]=0;initial=cv2.warpAffine(atlas,to_world[:2],(576,364),borderMode=cv2.BORDER_REFLECT_101)
    correction=np.eye(3,dtype=np.float32);score=None
    try:
        score,fit=cv2.findTransformECC(gray(reference),gray(initial),np.eye(2,3,dtype=np.float32),cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,80,1e-5),valid,5)
        if .85<fit[0,0]<1.15 and .85<fit[1,1]<1.15 and max(abs(fit[:,2]))<60:
            correction[:2]=[[fit[0,0],0,fit[0,2]],[0,fit[1,1],fit[1,2]]]
    except cv2.error:pass
    to_world=np.linalg.inv(correction)@to_world
    aligned=cv2.warpAffine(atlas,to_world[:2],(576,364),borderMode=cv2.BORDER_REFLECT_101)
    known=(valid>0)&(reference.min(2)>35)&(aligned.min(2)>35)
    if known.sum()>500:
        bias=np.clip(np.median(reference[known].astype(float)-aligned[known].astype(float),axis=0),-18,18)
        atlas=np.clip(atlas.astype(float)+bias,0,255).astype(np.uint8)
    np.savez_compressed(file,image=atlas,to_world=to_world)
    cv2.imwrite(str(V/'atlases'/f'{shot}-continuous-background.png'),atlas)
    status(f'Single continuous background {shot}; ECC={score}')
    return atlas,to_world

def faces(im,near=None):
    _,items=det.detect(im)
    if items is None:return None
    if near is None:
        strong=[p for p in items if p[-1]>.6 and p[2]*p[3]>2200 and p[1]<220]
        p=max(strong or list(items),key=lambda p:p[2]*p[3])
    else:p=min(items,key=lambda p:np.linalg.norm((p[:2]+p[2:4]/2)-near[:2])/150+abs(np.log(max(p[3],1)/max(near[2],1))))
    return np.array([(p[4]+p[6])/2,(p[5]+p[7])/2,p[3]],np.float32)

def tracking(shot,cameras):
    start,end,ref=SHOT[shot];source=[];generated=[]
    for n in range(start,end):
        s=faces(src(n));source.append([np.nan]*3 if s is None else s)
        if shot==5:g=np.array([338.5,92.8,84],np.float32)
        else:g=faces(picture(n),s)
        generated.append([np.nan]*3 if g is None else g)
    source=np.asarray(source,np.float32);generated=np.asarray(generated,np.float32)
    # Interpolate missing detections without smoothing the generated-frame
    # coordinates; these measurements cancel frame-to-frame crop/scale jumps.
    for a in [source,generated]:
        for k in range(3):
            valid=np.isfinite(a[:,k]);a[:,k]=np.interp(np.arange(len(a)),np.flatnonzero(valid),a[valid,k]) if valid.any() else [338,100,150][k]
    canonical_eye=np.median(generated[:,:2],axis=0);canonical_h=float(np.median(generated[:,2]))
    source_world=[]
    for j,m in enumerate(cameras):
        eye=np.linalg.inv(m)@np.r_[source[j,:2],1]
        source_world.append([eye[0],eye[1],source[j,2]/m[0,0]])
    world=np.asarray(source_world,np.float32)
    world[:,:2]=smooth(world[:,:2],.55)
    reference_h=float(np.median(world[:,2]));filtered=smooth(world[:,2:3],2)[:,0]
    world[:,2]=reference_h*np.clip(filtered/reference_h,.97,1.03)
    if shot==5:world[:,2]=canonical_h/cameras[ref-start,0,0]
    np.savez_compressed(V/'tracks'/f'faces-{shot}.npz',source=source,generated=generated,world=world,canonical_eye=canonical_eye,canonical_h=canonical_h)
    return generated,world,canonical_eye,canonical_h

def canonical_frames(shot,start,end,generated,canonical_eye,canonical_h):
    rec=[None]*4;results=[];base_start=SHOT[shot][0]
    for n in range(start,end):
        j=n-base_start;g=generated[j];scale=canonical_h/g[2]
        matrix=np.array([[scale,0,canonical_eye[0]-scale*g[0]],[0,scale,canonical_eye[1]-scale*g[1]]],np.float32)
        if shot==5:
            image=clean_frame(cv2.resize(cv2.imread('/workspace/generated_images/exec-bf3d6e55-f887-4ac8-a0c8-5b855eca5571.png'),(576,364)))
            a=np.load(D/'doom-full-alpha.npy');premul=image.astype(np.float32)*a[...,None]
            rgb=cv2.warpAffine(premul,matrix,(576,364),borderMode=cv2.BORDER_REPLICATE)
            alpha=cv2.warpAffine(a,matrix,(576,364),borderMode=cv2.BORDER_REPLICATE)
        else:
            file=V/'canonical'/f'{n:04d}.png'
            if file.exists() and not args.full:
                rgba=cv2.imread(str(file),cv2.IMREAD_UNCHANGED);alpha=rgba[...,3].astype(np.float32)/255;rgb=rgba[...,:3].astype(np.float32)*alpha[...,None]
                # Cached output is reused for full rendering after the pilot.
            else:
                image=cv2.warpAffine(picture(n),matrix,(576,364),borderMode=cv2.BORDER_REPLICATE)
                x=torch.from_numpy(np.ascontiguousarray(image[...,::-1])).permute(2,0,1)[None].float()/255
                with torch.inference_mode():
                    if n==start:
                        for _ in range(3):_,_,*rec=rvm(x,*rec,downsample_ratio=.5)
                    foreground,a,*rec=rvm(x,*rec,downsample_ratio=.5)
                alpha=a[0,0].numpy();color=foreground[0].permute(1,2,0).numpy()[...,::-1]*255
                if shot==3:
                    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                    green=((hsv[...,0]>=28)&(hsv[...,0]<=95)&(hsv[...,1]>35)&(image[...,1]>image[...,2].astype(float)*1.05)).astype(np.float32)
                    green[:,285:]=0;green=cv2.GaussianBlur(green,(5,5),.8)
                    alpha=np.maximum(alpha,green);color[green>.2]=image[green>.2]
                cv2.imwrite(str(file),np.dstack([np.clip(color,0,255).astype(np.uint8),np.clip(alpha*255,0,255).astype(np.uint8)]))
                rgb=cv2.GaussianBlur(color.astype(np.float32),(0,0),.4)*alpha[...,None]
        if shot!=5:
            # The matte model can mistake the room's human-shaped bronze statue
            # for another foreground subject. Keep only the actual character
            # silhouettes, allowing a generous margin for fine hair.
            p=probability(picture(n));seed=(p>.22).astype(np.uint8)
            components_count,labels,stats,_=cv2.connectedComponentsWithStats(seed)
            largest=sorted(range(1,components_count),key=lambda k:stats[k,cv2.CC_STAT_AREA],reverse=True)[:2 if shot==3 else 1]
            seed=np.isin(labels,largest).astype(np.uint8)
            seed=cv2.dilate(seed,np.ones((35,35),np.uint8)).astype(np.float32)
            allowed=cv2.warpAffine(seed,matrix,(576,364),borderMode=cv2.BORDER_REPLICATE)
            allowed=cv2.GaussianBlur(allowed,(7,7),1.2)
            rgb*=allowed[...,None];alpha*=allowed
            if shot==3:
                # Hulk is intentionally behind Wanda. A foreground-only matte
                # can classify him as background, so retain his own semantic
                # silhouette as a separate layer underneath her hair.
                hulkmask=np.clip((p-.08)/.55,0,1)
                cutoff=int(np.clip(g[0]-.38*g[2],260,340));hulkmask[:,cutoff:]=0
                hulkmask=cv2.GaussianBlur(hulkmask,(5,5),.8)
                ha=cv2.warpAffine(hulkmask,matrix,(576,364),borderMode=cv2.BORDER_REPLICATE)
                source_rgb=picture(n).astype(np.float32)*hulkmask[...,None]
                hp=cv2.warpAffine(source_rgb,matrix,(576,364),borderMode=cv2.BORDER_REPLICATE)
                rgb=rgb+hp*(1-alpha[...,None]);alpha=alpha+ha*(1-alpha)
        results.append((rgb,alpha))
        if n%30==0:status(f'Canonical frame {n} / shot {shot}')
    return results

dis_mid=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
gy,gx=np.mgrid[:243,:384].astype(np.float32);small_grid=np.dstack([gx,gy])
def warp_field(a,m):return cv2.remap(a,m[...,0],m[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
def midpoint(a,b):
    ga=cv2.resize(cv2.cvtColor(np.clip(a[0],0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY),(384,243))
    gb=cv2.resize(cv2.cvtColor(np.clip(b[0],0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY),(384,243))
    ab=dis_mid.calc(ga,gb,None);ba=dis_mid.calc(gb,ga,None)
    def inverse(f):
        m=small_grid-.5*f
        for _ in range(3):m=.35*m+.65*(small_grid-.5*warp_field(f,m))
        delta=cv2.resize(m-small_grid,(576,364));delta[...,0]*=576/384;delta[...,1]*=364/243
        return grid+delta
    ma,mb=inverse(ab),inverse(ba)
    return ((warp_field(a[0],ma)+warp_field(b[0],mb))*.5,(warp_field(a[1],ma)+warp_field(b[1],mb))*.5)

def composite(atlas,offset,camera,world,canonical_eye,canonical_h,layer):
    background_matrix=(camera@offset)[:2]
    bg=cv2.warpAffine(atlas,background_matrix,(576,364),borderMode=cv2.BORDER_REFLECT_101)
    s=world[2]/canonical_h
    pose=np.array([[s,0,world[0]-s*canonical_eye[0]],[0,s,world[1]-s*canonical_eye[1]],[0,0,1]],np.float32)
    transform=(camera@pose)[:2];pad=96
    transform[:,2]-=transform[:,:2]@np.array([pad,pad],np.float32)
    # Reflect only the isolated foreground, so scene architecture is never
    # copied or stretched along with the character's cropped body.
    rgb=cv2.copyMakeBorder(layer[0],pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
    a=cv2.copyMakeBorder(layer[1],pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
    fg=cv2.warpAffine(rgb,transform,(576,364));alpha=cv2.warpAffine(a,transform,(576,364))
    return np.clip(fg+bg*(1-alpha[...,None])+.5,0,255).astype(np.uint8)

def encoder(path,size,upscale=False):
    scale=['-vf','scale=720:1068:flags=bicubic,setsar=1'] if upscale else []
    return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',size,'-r','60','-i','pipe:0',*scale,'-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
baseline=cv2.VideoCapture(str(O/'video-marvel-completo-60fps.mp4'))
baseline_index=-1
def old_frame(n):
    global baseline_index
    if n!=baseline_index+1:baseline.set(cv2.CAP_PROP_POS_FRAMES,n)
    ok,im=baseline.read();baseline_index=n
    if not ok:raise RuntimeError(f'Missing delivered frame {n}')
    return cv2.resize(im,(576,854),interpolation=cv2.INTER_AREA)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
header=Image.new('RGB',(576,30),(18,20,25));draw=ImageDraw.Draw(header)
headers=[]
for name in ['Original','Entregue anteriormente','Novo: fundo e personagem no mesmo enquadramento']:
    h=header.copy();ImageDraw.Draw(h).text((9,4),name,font=font,fill='white');headers.append(np.asarray(h)[...,::-1].copy())
patch_mode=args.only_shot is not None
output=(V/f'patch-shot-{args.only_shot}-60fps.mp4') if patch_mode else O/('video-marvel-continuidade-v2-60fps.mp4' if args.full else 'teste-continuidade-enquadramento-60fps.mp4')
enc=encoder(output,'576x854',upscale=args.full or patch_mode)
comparison=None if args.full or patch_mode else encoder(O/'comparacao-continuidade-quadros.mp4','576x1182')
if patch_mode:segments=[(args.only_shot,*SHOT[args.only_shot][:2])]
elif args.full:segments=[(s,*SHOT[s][:2]) if s in SHOT and s!=5 else (s,CUTS[s],CUTS[s+1]) for s in range(10)]
else:segments=[(2,94,130),(7,570,594)]
count=0
for shot,start,end in segments:
    if shot not in SHOT or (args.full and shot==5):
        for n in range(start*2,end*2):enc.stdin.write(old_frame(n).tobytes());count+=1
        continue
    first,last,ref=SHOT[shot];cameras=camera_raw(first,last,ref)
    atlas,offset=build_atlas(shot,cameras)
    generated,world,eye,ch=tracking(shot,cameras)
    canonical=canonical_frames(shot,start,end,generated,eye,ch)
    for n in range(start,end):
        j=n-first;k=min(j+1,len(cameras)-1);local=n-start;nextlocal=min(local+1,len(canonical)-1)
        for half in range(2):
            f=.5*half
            layer=canonical[local] if half==0 or n==end-1 else midpoint(canonical[local],canonical[nextlocal])
            cam=(1-f)*cameras[j]+f*cameras[k];pose=(1-f)*world[j]+f*world[k]
            fixed=composite(atlas,offset,cam,pose,eye,ch,layer)
            original_full=cv2.imread(str(W/'source'/f'{n:04d}.png'));full=original_full.copy();full[230:594]=fixed
            enc.stdin.write(full.tobytes());count+=1
            if comparison:
                old=old_frame(n*2+half)
                panels=[np.vstack([headers[0],original_full[230:594]]),np.vstack([headers[1],old[230:594]]),np.vstack([headers[2],fixed])]
                comparison.stdin.write(np.vstack(panels).tobytes())
            if half==0 and (not args.full or n%12==0 or n in [575,578,580,586,594]):cv2.imwrite(str(V/'qa'/f'{n:04d}.jpg'),full)
        if n%30==0:status(f'Composed source frame {n}')
    status(f'Completed shot {shot}')
    if args.full and shot==8:
        for q in range(747*2,749*2):enc.stdin.write(old_frame(q).tobytes());count+=1
enc.stdin.close()
if enc.wait():raise RuntimeError('Encoding failed')
if comparison:
    comparison.stdin.close()
    if comparison.wait():raise RuntimeError('Comparison encoding failed')
baseline.release()
expected=2*(SHOT[args.only_shot][1]-SHOT[args.only_shot][0]) if patch_mode else (1556 if args.full else 120)
assert count==expected,count
report={'file':str(output),'frames':count,'fps':60,'audio':False,'source_interval_per_native_frame':1/30,'new_genuine_generated_frames':list(range(575,581)),'held_pose_frames':[581,582],'interpolated_reopening_frames':[583,584,585],'new_background_method':'One calibrated continuous empty background image per shot, projected with the same camera used for the character. No original/generated background patchwork or per-frame architecture deformation.','character_registration':'Source face position in scene coordinates and stable apparent head size; generated-frame crop/scale changes canceled before composition.','matting':'Robust Video Matting MobileNetV3 with recurrent temporal state; foreground color and alpha.','interpolation':'Foreground RGBA only, camera/background projected independently at the same sub-frame time.','unchanged_shots_if_full':[0,1,5,9],'limitations':'Only six newly generated native frames were produced before the next image generation was blocked. Other character poses are reused or interpolated; acting remains approximate. The reconstructed background matches the reference scene approximately.'}
(V/(f'patch-{args.only_shot}-report.json' if patch_mode else ('full-report.json' if args.full else 'pilot-report.json'))).write_text(json.dumps(report,ensure_ascii=False,indent=2))
status(f'Finished {output}; {count} frames')
