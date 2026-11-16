"""Targeted seam repair on the existing 2D reconstruction; keeps v3 intact."""
from pathlib import Path
from bisect import bisect_left
import argparse,json,time,subprocess
import cv2,numpy as np,torch
import render_continuity_v3 as m
from render_rife_backend import Rife
V=m.W/'continuity-v4';V.mkdir(exist_ok=True)
for d in ['trajectories','qa']:(V/d).mkdir(exist_ok=True)
class StablePuppet(m.Puppet):
 def prepare(self,first,last,shot):
  from scipy.ndimage import median_filter,gaussian_filter1d
  from scipy.spatial import cKDTree
  self.first=first;self.last=last;self.shot=shot;self.field=None;self.map_stats=[]
  cam=m.camera(shot);base=m.CUTS[shot];self.camera_sequence=cam;self.camera_start=base
  ir=np.linalg.inv(cam[self.ref-base]);self.to_world=ir@self.initial
  self.reference_world=cv2.transform(self.points[None],self.to_world[:2])[0]
  cache=V/'trajectories'/f'velocity-{self.ref:04d}-{first:04d}-{last:04d}.npz'
  if cache.exists():self.world=np.load(cache)['world']
  else:
   raw=np.stack([super(StablePuppet,self).positions(n) for n in range(first,last)])
   inv=np.linalg.inv(cam[first-base:last-base]);world=np.einsum('nij,nkj->nki',inv[:,:2,:2],raw)+inv[:,None,:2,2]
   clean=gaussian_filter1d(median_filter(world,size=(3,1,1),mode='nearest'),.8,axis=0,mode='nearest')
   if self.hasface:
    _,face_ids=cKDTree(self.points).query(self.face[:468,:2]);clean[:,face_ids]=gaussian_filter1d(world[:,face_ids],.35,axis=0,mode='nearest')
   self.world=clean;np.savez_compressed(cache,world=clean)
  if self.hasface:_,self.rigid_ids=cKDTree(self.points).query(self.face[m.STABLE,:2])
  else:
   xy=self.points.astype(int);a=self.alpha[np.clip(xy[:,1],0,363),np.clip(xy[:,0],0,575)];self.rigid_ids=np.flatnonzero(a>.8)
  self.summary={'method':'scene-space temporal filtering plus integrated continuous deformation; no per-frame global fallback','frames':last-first}
 def render(self,n):
  from continuous_warp_v4 import ContinuousWarp
  if self.field is None:self.field=ContinuousWarp(self.points)
  left=int(np.floor(n));right=int(np.ceil(n));t=n-left
  wa=self.world[int(np.clip(left-self.first,0,len(self.world)-1))];wb=self.world[int(np.clip(right-self.first,0,len(self.world)-1))];world=wa*(1-t)+wb*t
  rigid=m.affine_fit(self.reference_world[self.rigid_ids],world[self.rigid_ids]);template_to_world=rigid@self.to_world
  inv=np.linalg.inv(template_to_world);desired=cv2.transform(world[None],inv[:2])[0];velocity=desired-self.points
  displacement=self.field.inverse(velocity)
  j=left-self.camera_start;k=min(j+1,len(self.camera_sequence)-1);camera=self.camera_sequence[j]*(1-t)+self.camera_sequence[k]*t
  invall=np.linalg.inv(camera@template_to_world)
  uv=cv2.transform(m.GRID,invall[:2])+128
  d=cv2.remap(displacement,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);coords=uv+d
  rgb=cv2.remap(self.texture,coords[...,0],coords[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  alpha=cv2.remap(self.alpha_texture,coords[...,0],coords[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  if n==int(n):
   dx_x=cv2.Sobel(coords[...,0],cv2.CV_32F,1,0,ksize=3,scale=.125);dx_y=cv2.Sobel(coords[...,0],cv2.CV_32F,0,1,ksize=3,scale=.125)
   dy_x=cv2.Sobel(coords[...,1],cv2.CV_32F,1,0,ksize=3,scale=.125);dy_y=cv2.Sobel(coords[...,1],cv2.CV_32F,0,1,ksize=3,scale=.125)
   jac=dx_x*dy_y-dx_y*dy_x;region=alpha>.5;region[:3]=False;region[-3:]=False;region[:,:3]=False;region[:,-3:]=False
   self.map_stats.append({'frame':int(n),'nonpositive_pixels':int((jac[region]<=0).sum()),'minimum_jacobian':float(np.min(jac[region])) if region.any() else None})
  return rgb,alpha
class LayerTween:
 def __init__(self):self.rife=Rife(threads=2)
 def between(self,a,b,t):
  if t<=1e-6:return a
  if t>=1-1e-6:return b
  def tensor(layer):
   rgb,alpha=layer
   rgba=np.dstack([rgb[...,::-1]/255,alpha]).astype(np.float32)
   x=torch.from_numpy(np.ascontiguousarray(rgba)).permute(2,0,1)[None]
   return torch.nn.functional.pad(x,(0,(-576)%128,0,(-364)%128),mode='replicate')
  x=torch.cat([tensor(a),tensor(b)],1)
  from model.warplayer import warp
  with torch.inference_mode():
   flows,raw_mask,merged=self.rife.net(x,float(t),[32,16,8,4,1])
   out=merged[-1][0,:,:364,:576].permute(1,2,0).numpy()
   aa=warp(x[:,:4],flows[-1][:,:2])[0,3,:364,:576].numpy()
   bb=warp(x[:,4:],flows[-1][:,2:])[0,3,:364,:576].numpy()
   mask=torch.sigmoid(raw_mask)[0,0,:364,:576].numpy()
  alpha=np.clip(out[...,3],0,1)
  def distance(a):
   binary=(a>.5).astype(np.uint8)
   return cv2.distanceTransform(binary,cv2.DIST_L2,3)-cv2.distanceTransform(1-binary,cv2.DIST_L2,3)
  signed=distance(aa)*mask+distance(bb)*(1-mask)
  contour=np.clip(signed*.7+.5,0,1)
  strength=4*t*(1-t);final_alpha=alpha*(1-strength)+contour*strength
  color=np.clip(out[...,:3][...,::-1]/np.maximum(alpha[...,None],.0001),0,1)
  alpha=final_alpha;rgb=color*alpha[...,None]*255
  return rgb,alpha

def main():
 p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--opening',action='store_true');args=p.parse_args()
 output=m.O/('video-marvel-conexoes-v4-60fps.mp4' if args.full else 'teste-conexoes-v4-60fps.mp4')
 if args.opening:output=V/'opening-final-60fps.mp4'
 cap=cv2.VideoCapture(str(m.O/'video-marvel-continuidade-v3-60fps.mp4'));old_index=-1;old_frame=None
 def old(index):
  nonlocal old_index,old_frame
  if index==old_index:return old_frame.copy()
  if index==old_index+2:cap.grab()
  elif index!=old_index+1:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
  ok,im=cap.read();assert ok;old_index=index;old_frame=cv2.resize(im,(576,854),interpolation=cv2.INTER_AREA);return old_frame.copy()
 enc=m.encoder(output,fps=60,upscale=True);tween=LayerTween();report=[];frames=0
 segments=[(0,0,39)] if args.opening else ([(s,m.CUTS[s],m.CUTS[s+1]) for s in range(10)] if args.full else [(0,0,39),(5,272,349),(7,520,595)])
 for shot,first,last in segments:
  if shot in [1,9]:
   for n in range(first*2,last*2):enc.stdin.write(old(n).tobytes());frames+=1
   continue
  refs=[0,5,10,15,20,25,30,35,38] if shot==0 else [m.REFS[shot]]
  puppets={};cam=m.camera(shot);atlas,off=m.atlas(shot,cam)
  if shot==0:
   plate=V/'opening-clean-plate.png'
   if not plate.exists():cv2.imwrite(str(plate),m.clean(cv2.imread(str(m.D/'cleanplate-0000.png'))))
   atlas=cv2.imread(str(plate));off=np.linalg.inv(cam[0])
  for j,ref in enumerate(refs):
   puppet=StablePuppet(ref);start=m.CUTS[shot] if len(refs)==1 else refs[max(0,j-1)];end=m.CUTS[shot+1] if len(refs)==1 else min(m.CUTS[shot+1],refs[min(len(refs)-1,j+1)]+1)
   end=min(end,m.CUTS[shot]+len(cam));puppet.prepare(start,end,shot);puppets[ref]=puppet;report.append({'shot':shot,'reference':ref,**puppet.summary,'map_checks':puppet.map_stats});m.log(f'Regularized {ref}: {puppet.summary}')
  hulk=m.hulk_layer(cam) if shot==3 else None
  for native in range(first,last):
   for half in range(2):
    index=native*2+half
    if native in [747,748]:enc.stdin.write(old(index).tobytes());frames+=1;continue
    time=native+(.5*half if native<m.CUTS[shot+1]-1 else 0);local=native-m.CUTS[shot];fraction=time-native;camera=cam[local]*(1-fraction)+cam[min(local+1,len(cam)-1)]*fraction
    bg=m.background(atlas,off,camera)
    if hulk:
     hp,ha,inv=hulk;transform=(camera@inv)[:2];hp=cv2.warpAffine(hp,transform,(576,364));ha=cv2.warpAffine(ha,transform,(576,364));bg=hp+bg*(1-ha[...,None])
    j=bisect_left(refs,time)
    if j==0:a=b=refs[0]
    elif j==len(refs):a=b=refs[-1]
    else:a,b=refs[j-1:j+1]
    layer=puppets[a].render(time)
    if a!=b:layer=tween.between(layer,puppets[b].render(time),(time-a)/(b-a))
    rgb,alpha=layer;crop=np.clip(rgb+bg*(1-alpha[...,None]),0,255).astype(np.uint8);full=old(index);full[230:594]=crop
    enc.stdin.write(full.tobytes());frames+=1
    if half==0 and (native%15==0 or native in [533,534,535,536,580]):cv2.imwrite(str(V/'qa'/f'{native:04d}.jpg'),full)
   if native%30==0:m.log(f'v4 shot {shot} frame {native}')
  (V/f'completed-shot-{shot}.json').write_text(json.dumps([r for r in report if r['shot']==shot],indent=2))
 enc.stdin.close();assert enc.wait()==0;cap.release();expected=78 if args.opening else (1556 if args.full else 382);assert frames==expected,(frames,expected)
 (V/('opening-report.json' if args.opening else ('full-report.json' if args.full else 'pilot-report.json'))).write_text(json.dumps({'file':str(output),'frames':frames,'fps':60,'audio':False,'source_intervals':segments,'continuous_warp_checks':report,'background':'v3 persistent atlases; never sent to RIFE','opening_connections':'One shared RIFE field for premultiplied RGB and alpha; no double exposure crossfade.','limits':'Still based on 2D approved images; it cannot create physically correct unseen surfaces or exact new performances. Standing and ending remain v3.'},ensure_ascii=False,indent=2));m.log(f'Complete: {output}')
if __name__=='__main__':main()
