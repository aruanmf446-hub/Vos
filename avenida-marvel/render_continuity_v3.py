"""Source-driven identity-stable meshes, persistent background and independent Hulk.

This is a 2D motion-retargeting revision, not newly generated video or full 3D.
It never morphs the architecture or blends across the original hard cuts.
"""
from pathlib import Path
import argparse,json,time,subprocess
from bisect import bisect_left
import cv2,numpy as np,torch
from scipy.spatial import Delaunay
from fixed_mesh_v3 import FixedMesh
from scipy.ndimage import gaussian_filter1d
from composite_consistent_test import PersonMatte

R=Path(__file__).resolve().parent;W=R/'work';V=W/'continuity-v3';D=W/'delivery';C=W/'continuity-v2';O=Path('/workspace/shared/downloads')
V.mkdir(exist_ok=True)
for d in ['mattes','qa','atlases']:(V/d).mkdir(exist_ok=True)
CUTS=[0,39,76,173,241,272,349,404,666,749,778]
REFS={0:20,2:100,3:210,4:253,5:302,6:379,7:474,8:706}
SOURCE=np.load(V/'source-geometry.npz');ANCH=np.load(V/'anchor-geometry.npz');AN={int(n):j for j,n in enumerate(ANCH['frames'])}
SF=SOURCE['face'].copy();SP=SOURCE['pose'].copy()
for start,end in zip(CUTS[:-1],CUTS[1:]):
 for arr in [SF,SP]:
  a=arr[start:end];flat=a.reshape(len(a),-1)
  for k in range(flat.shape[1]):
   good=np.isfinite(flat[:,k])
   if good.any():flat[:,k]=np.interp(np.arange(len(a)),np.flatnonzero(good),flat[good,k])
  flat[:]=gaussian_filter1d(flat,.55,axis=0,mode='nearest')
STABLE=np.array([4,5,6,33,133,263,362,168,197,195,127,356,234,454,10,151])
H,WID=364,576
gy,gx=np.mgrid[:H,:WID].astype(np.float32);GRID=np.dstack([gx,gy]);PIX=GRID.reshape(-1,2)
cv2.setNumThreads(2);torch.set_num_threads(2)
model=None;rvm=None
starttime=time.monotonic()
def log(x):print(f'{time.monotonic()-starttime:.1f}s {x}',flush=True)
def src(n):return cv2.imread(str(W/'action'/f'{n:04d}.png'))
def clean(im):
 im=im.copy();im[:6]=im[6:7];im[-8:]=im[-9:-8];im[:,:4]=im[:,4:5];im[:,-4:]=im[:,-5:-4];return im
def raw(n):
 if n==302:return clean(cv2.resize(cv2.imread(str(R.parent/'generated_images/exec-bf3d6e55-f887-4ac8-a0c8-5b855eca5571.png')),(576,364)))
 for directory,idx in [(W/'anchors',n),(D/'raw',n),(W/'generated_native_v3',n),(W/'review_rife',n*2)]:
  p=directory/f'{idx:04d}.png'
  if p.exists():return clean(cv2.resize(cv2.imread(str(p)),(576,364)))
 raise FileNotFoundError(n)
def similarity(a,b,three=False):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
 ca=a.mean(0);cb=b.mean(0);aa=a-ca;bb=b-cb
 u,s,vt=np.linalg.svd(aa.T@bb);rot=u@vt
 if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt
 scale=float(np.clip(np.sum((aa@rot)*bb)/max(np.sum(aa*aa),1e-6),.5,2))
 return scale,rot,cb-scale*ca@rot
def apply(p,fit):s,r,t=fit;return np.asarray(p)@r*s+t
def affine_fit(a,b):
 s,r,t=similarity(a,b);m=np.eye(3,dtype=np.float32);m[:2,:2]=(s*r).T;m[:2,2]=t;return m

def matte(n):
 global model,rvm
 p=V/'mattes'/f'{n:04d}.png'
 if p.exists():return cv2.imread(str(p),0).astype(np.float32)/255
 image=raw(n)
 if n==302:a=np.load(D/'doom-full-alpha.npy')
 else:
  if model is None:model=PersonMatte()
  if rvm is None:rvm=torch.jit.load(str(R/'models/rvm/rvm_mobilenetv3_fp32.torchscript'),map_location='cpu').eval()
  x=torch.from_numpy(np.ascontiguousarray(image[...,::-1])).permute(2,0,1)[None].float()/255;rec=[None]*4
  with torch.inference_mode():
   for _ in range(4):fg,alpha,*rec=rvm(x,*rec,downsample_ratio=.5)
  a=alpha[0,0].numpy()
  pp,_=model(cv2.resize(image,(384,243)));pp=cv2.resize(pp,(576,364));mask=(pp>.16).astype(np.uint8)
  if 173<=n<241:
   # Keep Wanda here. Hulk receives an independent, persistent layer.
   f=ANCH['face'][AN[n]];cut=int(np.nanmin(f[:,0])-24);mask[:,:max(cut,190)]=0
  count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
  if count>1:
   f=ANCH['face'][AN[n]]
   if np.isfinite(f).all():point=f[:,:2].mean(0);ix=int(np.clip(point[0],0,575));iy=int(np.clip(point[1],0,363));label=int(labels[iy,ix])
   else:label=0
   if label==0:label=max(range(1,count),key=lambda k:stats[k,cv2.CC_STAT_AREA])
   allowed=cv2.dilate((labels==label).astype(np.uint8),np.ones((29,29),np.uint8));a*=cv2.GaussianBlur(allowed.astype(np.float32),(7,7),1.2)
 a=np.clip(a,0,1);cv2.imwrite(str(p),np.rint(a*255).astype(np.uint8));return a

def camera(shot):
 start,end=CUTS[shot:shot+2]
 p=C/'tracks'/f'camera-{start:04d}.npy'
 if p.exists():return np.load(p)
 p=D/'camera'/f'{start:04d}.npy'
 if p.exists():return np.load(p)
 return np.repeat(np.eye(3,dtype=np.float32)[None],end-start,axis=0)

def atlas(shot,cam):
 start,end=CUTS[shot:shot+2]
 if shot not in [0,5]:
  p=C/'atlases'/f'{shot}-continuous-plate.npz'
  data=np.load(p);return data['image'],data['to_world']
 # Recover one persistent atlas from the union of visible source background.
 path=V/'atlases'/f'{shot}.npz'
 if path.exists():a=np.load(path);return a['image'],a['to_world']
 reference=0 if shot==0 else 302;refcam=cam[reference-start]
 camera_ref=np.stack([m@np.linalg.inv(refcam) for m in cam])
 corners=np.array([[0,0,1],[575,0,1],[575,363,1],[0,363,1]],np.float32)
 world=np.concatenate([(np.linalg.inv(m)@corners.T).T[:,:2] for m in camera_ref])
 lo=np.floor(world.min(0)-12).astype(int);hi=np.ceil(world.max(0)+12).astype(int);size=tuple((hi-lo+1).tolist())
 shift=np.array([[1,0,-lo[0]],[0,1,-lo[1]],[0,0,1]],np.float32)
 sheet=cv2.imread(str(D/f'cleanplate-{reference:04d}.png'))
 if sheet is None:raise FileNotFoundError('clean plate')
 fill=cv2.warpAffine(sheet,shift[:2],size,borderMode=cv2.BORDER_REFLECT_101)
 total=np.zeros((*fill.shape[:2],3),np.float64);weight=np.zeros(fill.shape[:2],np.float64)
 for n in range(start,end,2):
  image=src(n);mask=np.load(D/'source-mattes'/f'{n:04d}.npy').astype(np.float32)/255
  valid=(mask<.01).astype(np.float32);valid=cv2.erode(valid,np.ones((9,9),np.uint8))
  m=(shift@np.linalg.inv(camera_ref[n-start]))[:2]
  total+=cv2.warpAffine(image.astype(np.float32)*valid[...,None],m,size)
  weight+=cv2.warpAffine(valid,m,size)
 known=(weight>.5).astype(np.float32);known=cv2.GaussianBlur(known,(0,0),.7)
 recovered=total/np.maximum(weight[...,None],1e-6)
 fill=np.clip(recovered*known[...,None]+fill*(1-known[...,None]),0,255).astype(np.uint8)
 offset=refcam@np.linalg.inv(shift)
 np.savez_compressed(path,image=fill,to_world=offset);cv2.imwrite(str(V/'atlases'/f'{shot}.jpg'),fill)
 return fill,offset

def background(a,off,cam):return cv2.warpAffine(a,(cam@off)[:2],(576,364),borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)

class Puppet:
 def __init__(self,ref):
  self.ref=ref;self.image=raw(ref).astype(np.float32);self.alpha=matte(ref);self.alpha[-8:]=self.alpha[-9:-8];self.alpha[:4]=self.alpha[4:5]
  self.face=ANCH['face'][AN[ref]].copy();self.pose=ANCH['pose'][AN[ref]].copy();self.source_face=SF[ref]
  self.hasface=np.isfinite(self.face).all() and np.isfinite(self.source_face).all() and ref!=302 and ref>=39
  if self.hasface:
   self.fit3=similarity(self.face[STABLE],SF[ref][STABLE]);target=apply(self.face,self.fit3)
   self.initial=affine_fit(self.face[STABLE,:2],target[STABLE,:2])
  elif ref==302:
   self.face=np.full((478,3),np.nan,np.float32)
   self.initial=np.eye(3,dtype=np.float32)
   source_eye=np.load(D/'doom-face-track.npy')[ref-272];self.initial[:2,2]=source_eye-np.array([338.5,92.8])
   p=np.zeros((33,4),np.float32);p[:,:3]=np.nan;p[:,3]=1
   for i,xy in {0:(339,95),11:(470,174),12:(400,168),13:(487,280),14:(404,274),15:(365,321),16:(304,317),23:(470,422),24:(406,417)}.items():p[i,:2]=xy;p[i,2]=0
   self.pose=p
  else:
   ids=[0,11,12,23,24];ok=[i for i in ids if np.isfinite(self.pose[i,:2]).all() and np.isfinite(SP[ref,i,:2]).all()]
   self.initial=np.eye(3,dtype=np.float32)
   if ref<39:
    self.initial[:2,2]=SP[ref,0,:2]-self.pose[0,:2]
   elif len(ok)>1:self.initial=affine_fit(self.pose[ok,:2],SP[ref,ok,:2])
  xy=np.array([(x,y) for y in range(-104,469,26) for x in range(-104,677,26)],np.float32)
  xy=np.vstack([xy,[[-104,-104],[679,-104],[679,468],[-104,468],[0,363],[575,363],[288,363]]]).astype(np.float32)
  if self.hasface:
   self.fc=self.face[:468,:2].mean(0);self.fr=np.maximum(np.ptp(self.face[:468,:2],axis=0)*.5,[15,20])
   # Face geometry owns the full interior; grid nodes animate hair and body.
   v=(xy-self.fc)/self.fr;xy=xy[(v*v).sum(1)>1.12]
   xy=np.vstack([xy,self.face[:468,:2]])
  else:
   self.fc=np.array([355,111],np.float32) if ref==302 else self.pose[0,:2].copy();self.fr=np.array([65,85],np.float32)
  self.points=np.unique(xy,axis=0);self.fixed_mesh=FixedMesh(self.points);self.registered=cv2.transform(self.points[None],self.initial[:2])[0]
  self.refgray=cv2.cvtColor(src(ref),cv2.COLOR_BGR2GRAY)
  self.premul=cv2.GaussianBlur(self.image,(0,0),.45)*self.alpha[...,None]
  self.pad=128
  self.texture=cv2.copyMakeBorder(self.premul,128,128,128,128,cv2.BORDER_REPLICATE)
  self.alpha_texture=cv2.copyMakeBorder(self.alpha,128,128,128,128,cv2.BORDER_REPLICATE)
  self.position_cache={}
  self.dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
  self.pose_registered=cv2.transform(self.pose[None,:,:2],self.initial[:2])[0]
 def positions(self,n):
  if n in self.position_cache:return self.position_cache[n]
  # Dense original motion for clothing; recomputed against the reference, so
  # errors do not accumulate and textures are never repeatedly resampled.
  flow=self.dis.calc(cv2.resize(self.refgray,(384,243)),cv2.resize(cv2.cvtColor(src(n),cv2.COLOR_BGR2GRAY),(384,243)),None)
  flow=cv2.resize(flow,(576,364));flow[...,0]*=1.5;flow[...,1]*=364/243;flow=cv2.GaussianBlur(flow,(0,0),5)
  q=self.registered.copy();d=cv2.remap(flow,q[:,0,None],q[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0]
  dst=q+d
  if self.ref==302:
   # Articulated arm bones preserve the armor instead of sliding a flat card.
   weights=[];moves=[]
   for a,b in [(11,23),(12,24),(11,13),(13,15),(12,14),(14,16)]:
    bp=self.pose_registered[[a,b]];sp0=SP[self.ref,[a,b],:2];sp1=SP[n,[a,b],:2]
    if not np.isfinite(sp1).all():continue
    desired=bp+sp1-sp0
    fit=affine_fit(bp,desired);new=cv2.transform(q[None],fit[:2])[0]
    axis=bp[1]-bp[0];t=np.clip((q-bp[0])@axis/max(axis@axis,1),0,1);dist=np.linalg.norm(q-bp[0]-t[:,None]*axis,axis=1)
    width=38 if b in [23,24] else 20
    weights.append(np.exp(-(dist/width)**2)+1e-5);moves.append(new)
   w=np.asarray(weights);dst=np.sum(np.asarray(moves)*w[...,None],axis=0)/w.sum(0)[:,None]
  if self.ref==302:
   tracked=np.load(D/'doom-face-track.npy');move=tracked[n-272]-tracked[self.ref-272]
   hfit=np.eye(3,dtype=np.float32);hfit[:2,2]=move;hfit=hfit@self.initial
   rigid=cv2.transform(self.points[None],hfit[:2])[0]
   dist=np.linalg.norm((self.points-self.fc)/self.fr,axis=1);influence=np.clip((1.65-dist)/.55,0,1)
   dst=dst*(1-influence[:,None])+rigid*influence[:,None]
  if self.ref>=39 and np.isfinite(SF[n]).all() and np.isfinite(SF[self.ref]).all():
   if self.hasface:
    current_to_ref=similarity(SF[n,STABLE],SF[self.ref,STABLE])
    cur_ref=apply(SF[n],current_to_ref)
    expr=cur_ref-SF[self.ref]
    # Preserve the identity's proportions and bound nonrigid skin motion.
    expr=np.clip(expr,-8,8)*.85
    base_face=apply(self.face,self.fit3)
    target_ref=base_face+expr
    # Relative aperture transfer closes the target's own eyes/lips fully;
    # additive landmark deltas otherwise leave its larger eyes half open.
    for upper,lower in [([246,161,160,159,158,157,173],[7,163,144,145,153,154,155]),([398,384,385,386,387,388,466],[382,381,380,374,373,390,249])]:
     original_gap=np.mean(np.linalg.norm(SF[self.ref,upper]-SF[self.ref,lower],axis=1))
     current_gap=np.mean(np.linalg.norm(cur_ref[upper]-cur_ref[lower],axis=1))
     ratio=float(np.clip((current_gap-1.5)/max(original_gap-1.5,2),.025,1.35))
     centers=(target_ref[upper]+target_ref[lower])*.5
     aperture=(base_face[upper]-base_face[lower])*.5*ratio
     target_ref[upper]=centers+aperture;target_ref[lower]=centers-aperture
    upper=[81,82,13,312,311];lower=[178,87,14,317,402]
    refgap=float(np.linalg.norm(SF[self.ref,13]-SF[self.ref,14]))
    if refgap>3:
     curgap=float(np.linalg.norm(cur_ref[13]-cur_ref[14]));ratio=float(np.clip((curgap-1)/max(refgap-1,2),.025,1.8))
     centers=(target_ref[upper]+target_ref[lower])*.5;aperture=(base_face[upper]-base_face[lower])*.5*ratio
     before=target_ref.copy();target_ref[upper]=centers+aperture;target_ref[lower]=centers-aperture
     target_ref[0]+=(target_ref[13]-before[13])*.6;target_ref[17]+=(target_ref[14]-before[14])*.6
    ss,rr,tt=current_to_ref
    animated=(target_ref-tt)@rr.T/ss
    headfit=affine_fit(self.face[STABLE,:2],animated[STABLE,:2])
   else:
    headmove=affine_fit(SF[self.ref,STABLE,:2],SF[n,STABLE,:2])
    headfit=headmove@self.initial
   rigid=cv2.transform(self.points[None],headfit[:2])[0]
   distance=np.linalg.norm((self.points-self.fc)/self.fr,axis=1)
   influence=np.clip((2.05-distance)/.65,0,1)
   # Hair above the forehead follows the skull with a soft neck transition.
   influence[self.points[:,1]>self.fc[1]+self.fr[1]*1.15]*=.45
   dst=dst*(1-influence[:,None])+rigid*influence[:,None]
   if self.hasface:
    from scipy.spatial import cKDTree
    _,ix=cKDTree(self.points).query(self.face[:468,:2]);dst[ix]=animated[:468,:2]
  # Bounds outside the actual isolated character are irrelevant; keeping them
  # fixed avoids foldovers at the outer image rectangle.
  border=(self.points[:,0]<=-100)|(self.points[:,0]>=675)|(self.points[:,1]<=-100)|(self.points[:,1]>=464)
  # Bottom fabric follows the torso, while empty border corners remain fixed.
  clear=np.array([self.alpha[int(np.clip(y,0,363)),int(np.clip(x,0,575))]<.05 for x,y in self.points])
  dst[border&clear]=self.points[border&clear]
  if self.ref!=302:
   bottom=np.clip((self.points[:,1]-225)/138,0,1)
   dst[:,1]=dst[:,1]*(1-bottom)+self.points[:,1]*bottom
  dst=dst.astype(np.float32);self.position_cache[n]=dst
  if len(self.position_cache)>6:del self.position_cache[next(iter(self.position_cache))]
  return dst
 def render(self,n):
  left=int(np.floor(n));right=int(np.ceil(n));t=n-left
  dst=self.positions(left) if left==right else (1-t)*self.positions(left)+t*self.positions(right)
  mapping=self.fixed_mesh.mapping(dst);missing=mapping[...,0]<-900
  p=cv2.remap(self.texture,mapping[...,0]+self.pad,mapping[...,1]+self.pad,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  a=cv2.remap(self.alpha_texture,mapping[...,0]+self.pad,mapping[...,1]+self.pad,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  a[missing]=0;p[missing]=0
  return p,a

def hulk_layer(cam):
 im=raw(210);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
 pp=matte(210) # Wanda matte is deliberately separate.
 # Color and a persistent source silhouette define Hulk once, not per frame.
 green=((hsv[...,0]>22)&(hsv[...,0]<95)&(hsv[...,1]>25)&(im[...,1]>im[...,2].astype(float)*1.015)).astype(np.uint8);green[:,285:]=0
 polygon=np.zeros((364,576),np.uint8);cv2.fillPoly(polygon,[np.array([[0,35],[75,48],[114,140],[156,141],[162,77],[192,59],[242,68],[266,103],[279,185],[266,255],[226,363],[0,363]])],1)
 labels=np.full((364,576),cv2.GC_BGD,np.uint8);labels[polygon>0]=cv2.GC_PR_FGD
 labels[(green>0)&(polygon>0)]=cv2.GC_FGD
 cv2.grabCut(im,labels,None,np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_MASK)
 mask=((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)).astype(np.float32);mask*=polygon;mask=cv2.GaussianBlur(mask,(5,5),.75)
 cv2.imwrite(str(V/'qa/hulk-alpha.png'),np.rint(mask*255).astype(np.uint8))
 return im.astype(np.float32)*mask[...,None],mask,np.linalg.inv(cam[210-173])

def encoder(path,size='576x854',fps=30,upscale=False):
 return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',size,'-r',str(fps),'-i','pipe:0',*(['-vf','scale=720:1068:flags=bicubic,setsar=1'] if upscale else []),'-an','-c:v','libx264','-preset','fast','-crf','17','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shots',default='0,2,3,4,5,6,7,8');ap.add_argument('--pilot',action='store_true');ap.add_argument('--stills',action='store_true');ap.add_argument('--fps',type=int,choices=[30,60],default=30);args=ap.parse_args();shots=[int(s) for s in args.shots.split(',')]
 baseline=cv2.VideoCapture(str(O/'video-marvel-continuidade-v2-60fps.mp4'))
 previous_index=-1;previous_frame=None
 def at(index):
  nonlocal previous_index,previous_frame
  if index==previous_index:return previous_frame.copy()
  if index==previous_index+2:baseline.grab()
  elif index!=previous_index+1:baseline.set(cv2.CAP_PROP_POS_FRAMES,index)
  ok,frame=baseline.read();assert ok
  previous_index=index;previous_frame=cv2.resize(frame,(576,854),interpolation=cv2.INTER_AREA)
  return previous_frame.copy()
 def old(n):return at(int(n)*2)
 changed={};puppets={};output=V/('pilot-native.mp4' if args.pilot else 'full-native.mp4')
 if args.fps==60:output=O/('teste-movimento-v3-60fps.mp4' if args.pilot else 'video-marvel-continuidade-v3-60fps.mp4')
 enc=None if args.stills else encoder(output,fps=args.fps,upscale=args.fps==60)
 windows={0:(0,39),2:(88,124),3:(173,241),4:(241,272),5:(272,349),6:(359,395),7:(510,606),8:(690,726)}
 for shot in (shots if args.pilot or args.stills else range(10)):
  first,last=CUTS[shot:shot+2];render=shot in shots
  if args.pilot:first,last=windows[shot]
  if not render:
   for n in range(first,last):
    for half in range(args.fps//30):
     enc.stdin.write(at(n*2+half).tobytes())
   continue
  cam=camera(shot);image,off=atlas(shot,cam)
  if shot==0:refs=[0,5,10,15,20,25,30,35,38]
  else:refs=[REFS[shot]]
  for ref in refs:
   if ref not in puppets:puppets[ref]=Puppet(ref)
  hulk=hulk_layer(cam) if shot==3 else None
  frames=[first,(first+last)//2,last-1] if args.stills else range(first,last)
  for native in frames:
   for half in range(1 if args.stills else args.fps//30):
    n=native+(.5*half if native<last-1 else 0)
    if shot==8 and n>=747:
     if enc:enc.stdin.write(old(int(n)).tobytes())
     continue
    local=native-CUTS[shot];frac=n-native;nc=min(local+1,len(cam)-1);camera_now=(1-frac)*cam[local]+frac*cam[nc];bg=background(image,off,camera_now)
    if hulk:
     hp,ha,inv=hulk;transform=(camera_now@inv)[:2]
     hp=cv2.warpAffine(hp,transform,(576,364));ha=cv2.warpAffine(ha,transform,(576,364));bg=hp+bg*(1-ha[...,None])
    j=bisect_left(refs,n)
    if j==0:a=b=refs[0]
    elif j==len(refs):a=b=refs[-1]
    else:a,b=refs[j-1:j+1]
    p,alpha=puppets[a].render(n)
    if b!=a:
     q,beta=puppets[b].render(n);t=(n-a)/(b-a);t=t*t*(3-2*t)
     p=p*(1-t)+q*t;alpha=alpha*(1-t)+beta*t
    crop=np.clip(p+bg*(1-alpha[...,None]),0,255).astype(np.uint8)
    full=old(native);full[230:594]=crop
    if enc:enc.stdin.write(full.tobytes())
    if args.stills or (half==0 and native%15==0):cv2.imwrite(str(V/'qa'/f'{native:04d}.jpg'),full)
    if half==0 and native%30==0:log(f'Shot {shot}, source frame {n}')
  log(f'Finished shot {shot}')
 if enc:
  enc.stdin.close();assert enc.wait()==0
 baseline.release();log(f'Written {output}')
 (V/'method.json').write_text(json.dumps({'method':'Persistent approved identity texture per dialogue shot; dense original body motion and 3D face landmark expression retargeting. Articulated Doom arms. Independent persistent Hulk. Recovered fixed source background atlas for shots 0 and 5.','source_frames':778,'native_fps':30,'new_generated_images':0,'fixed_mesh_topology':True,'fps60_method':'Interpolate source-measured vertex positions and camera transforms at half time; render original isolated texture once per output frame, without flow-warping the background.','limits':'2D texture retargeting cannot create unseen surfaces, new hand poses or exact mouth interiors. Opening retains nine approved views for the turn; standing and final portrait reuse v2.'},indent=2))
if __name__=='__main__':main()
